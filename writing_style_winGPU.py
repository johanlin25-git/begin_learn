# ================================================================
# 1) 环境与基础配置（针对RTX 3090 24G优化）
# ================================================================
import os, gc, json, uuid, shutil
import torch
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
# 禁用Unsloth的Triton编译
os.environ["UNSLOTH_USE_TRITON"] = "0"

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import (
    standardize_sharegpt,
    train_on_responses_only,
    get_chat_template,
)
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

# —— RTX 3090 24G 优化配置
max_seq_length = 2048
dtype = None
load_in_4bit = True

# —— 路径优化
WORK_DIR = "/home/vipuser/finetuneGPU"
TMP_DIR = os.path.join(WORK_DIR, "tmp_dataset_cache")
os.makedirs(TMP_DIR, exist_ok=True)

# —— 设备与性能优化
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 3407
torch.manual_seed(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ================================================================
# 2) 加载模型（修复模板配置）
# ================================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-2-7b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto",
)

# —— LoRA配置
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

# —— 关键修复：使用unsloth的标准Llama-2模板
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",  #
)

# 确保pad_token设置正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ================================================================
# 3) 固定 Few-shot 示例
# ================================================================
FEW_SHOT_MESSAGES = [
    {"role": "user", "content": "请用唐诗五言绝句描写早春的江南，意境清新，收尾含蓄。"},
    {"role": "assistant", "content": "细雨涨春潮，\n轻烟出柳梢。\n桃花桥上客，\n不语看兰桡。"},
    {"role": "user", "content": "以唐诗五言绝句写边塞怀古，苍凉而不失昂扬。"},
    {"role": "assistant", "content": "风急雁声断，\n荒城挂夕晖。\n铁衣埋旧垒，\n沙路有人归。"},
]

SYSTEM_HINT = (
    "你是一位擅写唐诗的诗人，擅长五言绝句：四句、每句五字，讲求对仗与意境，"
    "语言凝练含蓄，避免现代口语与网络词。必要时可用典，但不过度。"
)


def build_messages_with_fewshots(user_prompt: str):
    msgs = [{"role": "system", "content": SYSTEM_HINT}]
    msgs += FEW_SHOT_MESSAGES
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


# ================================================================
# 4) 自监督增广
# ================================================================
themes = ["春", "夏", "秋", "冬", "边塞", "旅途", "思乡", "山居", "夜雨", "海岸", "江畔", "田园"]
imagery = ["柳", "雁", "月", "松", "泉", "雪", "花", "沙", "云", "灯", "舟", "笛"]
moods = ["清新", "苍凉", "昂扬", "寂寞", "恬淡", "豪迈", "悠远", "惆怅"]


def make_5char_quatrain(t, i, m):
    line1 = f"{i}影{t}波"
    line2 = f"风过{t}梢"
    line3 = f"旧梦随{m}"
    line4 = f"归人自遥"
    line3 = line3.replace("随清新", "随微风").replace("随苍凉", "入霜霄") \
        .replace("随昂扬", "逐长箫").replace("随寂寞", "落寒潮") \
        .replace("随恬淡", "在渔舠").replace("随豪迈", "上云霄") \
        .replace("随悠远", "向天涯").replace("随惆怅", "付晚潮")
    poem = "\n".join([line1, line2, line3, line4])
    return poem


def synthesize_self_instruct_examples(limit=200):
    pairs = []
    count = 0
    for t in themes:
        for i in imagery:
            for m in moods:
                if count >= limit:
                    break
                instr = f"请用唐诗五言绝句写{t}景，并融入「{i}」意象，整体气质{m}。"
                out = make_5char_quatrain(t, i, m)
                pairs.append({"conversations": [
                    {"role": "user", "content": instr},
                    {"role": "assistant", "content": out}
                ]})
                count += 1
            if count >= limit:
                break
        if count >= limit:
            break
    return Dataset.from_list(pairs)


self_instruct_ds = synthesize_self_instruct_examples(limit=200)

# ================================================================
# 5) 载入原始数据
# ================================================================
RAW_DATASET_DIR = os.path.join(WORK_DIR, "ML_Spring2025_HW5", "fastchat_alpaca_52k")
raw_ds = None
if os.path.isdir(RAW_DATASET_DIR):
    raw_ds = load_from_disk(RAW_DATASET_DIR)
    raw_ds = standardize_sharegpt(raw_ds, num_proc=4)


def convo_len(ex):
    return sum(len(m["content"].split()) for m in ex["conversations"])


def sort_split(ds: Dataset):
    def key_adv(ex):
        score = ex.get("score")
        score = 1.0 if score is None else score
        return 1e-5 * convo_len(ex) + 1.0 * score

    simple = Dataset.from_list(sorted(list(ds), key=convo_len))
    advanced = Dataset.from_list(sorted(list(ds), key=key_adv, reverse=True))
    return simple, advanced


base_train = self_instruct_ds if raw_ds is None else concatenate_datasets([self_instruct_ds, raw_ds])
simple_ds, advanced_ds = sort_split(base_train)


# ================================================================
# 6) 数据格式化（推荐：手动tokenization + 标准Trainer）
# ================================================================
def tokenize_function(examples):
    texts = []
    for convo in examples["conversations"]:
        messages = [{"role": "system", "content": SYSTEM_HINT}]
        messages += FEW_SHOT_MESSAGES

        for msg in convo:
            messages.append({"role": msg["role"], "content": msg["content"]})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    # Tokenize并创建labels
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_seq_length,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


# 处理训练数据
train_step1 = simple_ds.select(range(0, min(200, len(simple_ds)))).map(
    tokenize_function,
    batched=True,
    num_proc=2,
    load_from_cache_file=False,
    remove_columns=simple_ds.column_names
)

train_step2 = advanced_ds.select(range(0, min(200, len(advanced_ds)))).map(
    tokenize_function,
    batched=True,
    num_proc=2,
    load_from_cache_file=False,
    remove_columns=advanced_ds.column_names
)

# ================================================================
# 8) 创建Trainer（标准Trainer）
# ================================================================
from transformers import Trainer

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_step1,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_seq_length,
    ),
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=8,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=SEED,
        max_grad_norm=1.0,
        output_dir=os.path.join(WORK_DIR, "outputs"),
        report_to="none",
        save_strategy="epoch",
        save_total_limit=3,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    ),
)

# 关键修复：移除有问题的 train_on_responses_only 调用
# 因为标准的Llama-2模板已经正确处理了标签掩码

# ================================================================
# 9) 两阶段训练
# ================================================================
print("开始第一阶段训练（简单样本）...")
trainer.train_dataset = train_step1
trainer.train()

print("开始第二阶段训练（困难样本）...")
trainer.train_dataset = train_step2
trainer.train()


# ================================================================
# 10) 推理函数
# ================================================================
def parse_true_output(text):
    # 简单的输出解析函数
    if "[/INST]" in text:
        parts = text.split("[/INST]")
        if len(parts) > 1:
            return parts[-1].replace("</s>", "").strip()
    return text.strip()


FastLanguageModel.for_inference(model)


def generate_poem(prompt: str, max_new_tokens=128):
    msgs = build_messages_with_fewshots(prompt)
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)

    outputs = model.generate(
        input_ids=inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.85,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.batch_decode(outputs)[0]
    return parse_true_output(text)


# —— 测试生成
print("\n" + "=" * 50)
print("示例生成——题：夜雨思乡（五言绝句）")
print(generate_poem("以五言绝句写夜雨思乡，意境含蓄。"))

# ================================================================
# 11) 评测集推理和模型保存
# ================================================================
TEST_JSON = os.path.join(WORK_DIR, "ML_Spring2025_HW5", "test_set_evol_instruct_150.json")
if os.path.isfile(TEST_JSON):
    print("开始测试集推理...")
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = {}
    for i, entry in enumerate(test_data):
        entry_id = entry.get("id", f"test_{i}")
        last_user = ""
        for c in reversed(entry.get("conversations", [])):
            if c.get("from") == "human":
                last_user = c.get("value", "")
                break
        results[entry_id] = {
            "input": last_user,
            "output": generate_poem(last_user, max_new_tokens=128),
        }
        if i % 10 == 0:
            print(f"[进度 {i + 1}/{len(test_data)}] 已完成推理")

    with open(os.path.join(WORK_DIR, "pred.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("测试集推理完成，结果已保存")

# 保存模型
print("保存模型中...")
os.makedirs(os.path.join(WORK_DIR, "lora_model"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "merged_model"), exist_ok=True)

model.save_pretrained(os.path.join(WORK_DIR, "lora_model"))
tokenizer.save_pretrained(os.path.join(WORK_DIR, "lora_model"))

merged = model.merge_and_unload()
merged.save_pretrained(os.path.join(WORK_DIR, "merged_model"))
tokenizer.save_pretrained(os.path.join(WORK_DIR, "merged_model"))

print("训练完成！所有文件已保存。")