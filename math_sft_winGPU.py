# =========================================================
# [1] 环境与依赖初始化（云服务器优化版本）
# =========================================================
import os, json, csv, random, torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline,
    TrainingArguments, DataCollatorForLanguageModeling
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel

# 设置随机种子和CUDA
random.seed(42)
torch.manual_seed(42)

# CUDA优化设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 云服务器路径设置
WORK_DIR = "/home/vipuser/finetuneGPU"
os.makedirs(WORK_DIR, exist_ok=True)


# =========================================================
# [2] 工具函数优化（增加错误处理和进度显示）
# =========================================================
def load_jsonlines(path: str):
    """逐行读取 jsonl -> list[dict]"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in tqdm(f, desc=f"加载 {path}")]


def load_csv_prompts(path: str, field: str = "prompt_text"):
    """读取测试集（CSV）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return [row[field] for row in tqdm(rows, desc=f"加载 {path}")]


def extract_ans_from_response(text: str) -> str:
    """
    增强答案提取：处理更多格式变体
    """
    if "####" in text:
        seg = text.split("####")[-1].strip()
    else:
        # 如果没有####，尝试找最后一个数字
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            seg = numbers[-1]
        else:
            seg = text.strip()

    # 清理符号和单位
    for ch in [",", "$", "%", "g", "kg", "ml", "L", "meters", "m"]:
        seg = seg.replace(ch, "")

    # 提取数字（支持负数和浮点数）
    import re
    numbers = re.findall(r'-?\d+\.?\d*', seg)
    return numbers[0] if numbers else seg.strip()


def latest_checkpoint(dir_path: str) -> str | None:
    """找到最新的checkpoint"""
    if not os.path.isdir(dir_path):
        return None
    cks = [d for d in os.listdir(dir_path) if d.startswith("checkpoint-")]
    if not cks:
        return None
    cks = sorted(cks, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(dir_path, cks[-1])


# =========================================================
# [3] 加载基础模型（云服务器优化配置）
# =========================================================
sft_model_name = "./meta_llama"  # 确保这个路径在云服务器上存在

# 更激进的量化配置以节省显存
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

print("加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    sft_model_name,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map="auto",  # 使用auto device map
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    sft_model_name,
    trust_remote_code=True,
)

# 分词器设置优化
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# =========================================================
# [4] 配置 PEFT / LoRA（针对数学推理优化）
# =========================================================
peft_config = LoraConfig(
    r=16,  # 增加秩以提升数学推理能力
    lora_alpha=32,  # 增加alpha
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "up_proj", "down_proj", "gate_proj"],
)

print("应用LoRA配置...")
train_model = get_peft_model(base_model, peft_config)


# =========================================================
# [5] 构建 n-shot 对话（优化提示工程）
# =========================================================
def build_nshot_dialog(nshot_pool: list, n: int, q: str, a: str | None, mode: str):
    """
    构建数学推理对话
    """
    assert mode in ("train", "test")
    shots = random.sample(nshot_pool, k=min(n, len(nshot_pool)))

    chats = []
    for eg in shots:
        chats += [
            {"role": "user", "content": f"Question: {eg['question']}"},
            {"role": "assistant",
             "content": f"Reasoning: Let's think step by step. {eg.get('reasoning', '')} The final answer is: #### {extract_ans_from_response(eg['answer'])}"},
        ]

    # 增强的指令提示
    instruct = (
        "Please reason carefully about this question. Think step by step and provide your reasoning. "
        "At the end, you MUST write the final answer as '#### [number]'."
    )

    chats.append({"role": "user", "content": f"Question: {q}\n{instruct}"})

    if mode == "train" and a is not None:
        # 训练时提供完整的推理过程
        chats.append({"role": "assistant",
                      "content": f"Reasoning: Let's think step by step. The final answer is: #### {extract_ans_from_response(a)}"})

    return chats


# =========================================================
# [6] 加载并格式化训练数据（增加数据处理效率）
# =========================================================
print("加载训练数据...")
gsm8k_train = load_jsonlines("gsm8k_train.jsonl")
TRAIN_N_SHOT = 2  # 增加few-shot数量


def format_training_examples(qa_list):
    """批量格式化训练数据"""
    formatted = []
    max_len = 0

    for qa in tqdm(qa_list, desc="格式化训练数据"):
        dialog = build_nshot_dialog(gsm8k_train, TRAIN_N_SHOT, qa["question"], qa["answer"], "train")

        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=False
        )

        # 清理模板标记
        if "<|eot_id|>" in text:
            text = text[text.index("<|eot_id|>") + len("<|eot_id|>"):]

        token_len = len(tokenizer(text)["input_ids"])
        max_len = max(max_len, token_len)
        formatted.append({"text": text})

    print(f"最大序列长度: {max_len}")
    return formatted, max_len


# 格式化训练数据
formatted_data, max_seq_length = format_training_examples(gsm8k_train[:800])  # 使用更多数据
train_ds = Dataset.from_list(formatted_data)

# =========================================================
# [7] 训练参数优化（充分利用RTX 3090）
# =========================================================
training_args = TrainingArguments(
    output_dir=os.path.join(WORK_DIR, "sft_output"),
    per_device_train_batch_size=8,  # 大幅增加批次大小
    gradient_accumulation_steps=2,  # 减少梯度累积步数
    num_train_epochs=3,  # 增加训练轮数
    learning_rate=2e-4,  # 提高学习率
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=20,
    save_steps=200,
    eval_steps=200,
    save_total_limit=3,
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,  # 增加数据加载工作进程
    bf16=True,  # 使用bfloat16
    tf32=True,  # 启用TF32
    gradient_checkpointing=True,  # 梯度检查点节省显存
    report_to="none",
    group_by_length=True,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # 对齐优化
)

print("初始化训练器...")
trainer = SFTTrainer(
    model=train_model,
    train_dataset=train_ds,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    max_seq_length=min(max_seq_length, 2048),  # 限制最大长度
    dataset_text_field="text",
    packing=False,  # 数学推理任务不适合packing
)

# =========================================================
# [8] 训练执行
# =========================================================
print("开始训练...")
trainer.train()

# 保存最终模型
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)
print("训练完成!")

# =========================================================
# [9] 推理优化（使用训练好的模型）
# =========================================================
print("准备推理模型...")
ckpt = latest_checkpoint(training_args.output_dir)
if ckpt is None:
    print("使用最终模型进行推理")
    trained_model = train_model
else:
    print(f"加载检查点: {ckpt}")
    trained_model = PeftModel.from_pretrained(base_model, ckpt)

trained_model.eval()

# 优化生成管道
generator = pipeline(
    "text-generation",
    model=trained_model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device=device,
    max_new_tokens=512,  # 增加生成长度以容纳完整推理
    do_sample=False,  # 数学推理用贪婪解码更稳定
    temperature=0.1,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
)


def generate_answer(dialog: list) -> str:
    """优化的生成函数"""
    prompt = tokenizer.apply_chat_template(
        dialog, tokenize=False, add_generation_prompt=True
    )

    try:
        with torch.no_grad():
            outputs = generator(
                prompt,
                return_full_text=False,
                num_return_sequences=1,
                clean_up_tokenization_spaces=True
            )

        if isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], dict):
                text = outputs[0].get("generated_text", "")
            else:
                text = str(outputs[0])
        else:
            text = str(outputs)

        return text.strip()
    except Exception as e:
        print(f"生成错误: {e}")
        return ""


# =========================================================
# [10] 评估和推理（批量处理优化）
# =========================================================
TEST_N_SHOT = 2


def evaluate_gsm8k(test_file: str, is_public: bool = True):
    """评估GSM8K数据集"""
    test_data = load_jsonlines(test_file)
    predictions = []
    correct = 0

    desc = "GSM8K Public评估" if is_public else "GSM8K Private推理"
    bar = tqdm(total=len(test_data), desc=desc)

    for i, qa in enumerate(test_data):
        dialog = build_nshot_dialog(gsm8k_train, TEST_N_SHOT, qa["question"], None, "test")
        resp = generate_answer(dialog)
        pred = extract_ans_from_response(resp)

        predictions.append(pred)

        if is_public:
            gold = extract_ans_from_response(qa["answer"])
            if pred == gold:
                correct += 1
            bar.set_postfix(acc=f"{correct / (i + 1):.3f}")

        bar.update()

    bar.close()

    if is_public:
        accuracy = correct / len(test_data)
        print(f"GSM8K {desc} 准确率: {accuracy:.3f}")

    return predictions


print("开始评估...")
# 10.1 公测集
gsm8k_public_preds = evaluate_gsm8k("gsm8k_test_public.jsonl", is_public=True)

# 10.2 私测集
gsm8k_private_preds = evaluate_gsm8k("gsm8k_test_private.jsonl", is_public=False)

# 10.3 AILuminate推理
print("AILuminate推理...")
ailuminate_questions = load_csv_prompts("ailuminate_test.csv")
ailuminate_predictions = []

bar = tqdm(total=len(ailuminate_questions), desc="AILuminate推理")
for q in ailuminate_questions:
    dialog = [{"role": "user",
               "content": f"Question: {q}\nPlease provide step-by-step reasoning and end with #### [answer]."}]
    resp = generate_answer(dialog)
    ailuminate_predictions.append(resp)
    bar.update()
bar.close()

# =========================================================
# [11] 保存结果
# =========================================================
STUDENT_ID = "stu23"
results_file = os.path.join(WORK_DIR, f"{STUDENT_ID}_results.txt")

# 合并所有预测
all_predictions = gsm8k_public_preds + gsm8k_private_preds + ailuminate_predictions

with open(results_file, "w", encoding="utf-8") as f:
    for pred in all_predictions:
        f.write(str(pred) + "\n")

print(f"所有预测结果已保存到: {results_file}")

# 额外保存详细结果
detailed_results = {
    "gsm8k_public": gsm8k_public_preds,
    "gsm8k_private": gsm8k_private_preds,
    "ailuminate": ailuminate_predictions,
    "training_config": {
        "model": sft_model_name,
        "lora_r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "epochs": training_args.num_train_epochs,
    }
}

with open(os.path.join(WORK_DIR, "training_details.json"), "w", encoding="utf-8") as f:
    json.dump(detailed_results, f, ensure_ascii=False, indent=2)

print("训练和推理完成！")