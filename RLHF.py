import torch
import os
import unsloth  # 必须在最前面！
import json
import sys, re, requests
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth import PatchDPOTrainer
from unsloth.chat_templates import get_chat_template

# 添加Windows兼容性设置
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置环境变量来配置代理（根据你的Clash实际地址修改）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"  # 解决Windows CUDA警告

# 应用补丁
PatchDPOTrainer()  # 修复 HuggingFace Trainer 与 DPO 的兼容性

if __name__ == "__main__":
    max_seq_length = 512
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,  # 权重和计算时数值精度类型
        load_in_4bit=load_in_4bit,
        cache_dir="./models"
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
    )

    with open(r"C:\AProjects\RLHF_pro\data\raw\train.json", encoding="utf-8") as jsonfile:
        full_data = json.load(jsonfile)
    with open(r"C:\AProjects\RLHF_pro\data\raw\test.json", encoding="utf-8") as jsonfile:
        test_data = json.load(jsonfile)


    # 把json格式转成对话格式
    def data_formulate(data):
        prompt_text = data["prompt"]
        messages = [
            {"role": "system", "content": "your entire response must be 100 characters or less"},
            {"role": "user", "content": data['prompt']},
        ]
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"formatted_text": text}


    def extract_assistant_response(text):
        # 分割文本获取助手部分
        parts = text.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(parts) > 1:
            assistant_part = parts[1]  # 有两部分，前是system和user
            response_parts = assistant_part.split("<|eot_id|>")  # 重复分割
            if len(response_parts) > 0:
                assistant_response = response_parts[0].strip()
                return assistant_response  # 添加返回值
        else:
            print("未找到助手部分")
        print("=" * 80)
        return ""  # 如果没有找到，返回空字符串


    # 第六模块
    original_model_response = []
    for data in test_data:
        id = data['id']
        prompt = data['prompt']
        formatted_result = data_formulate(data)
        # 从字典中获取格式化后的文本
        formatted_text = formatted_result["formatted_text"]
        # 添加模型生成步骤
        inputs = tokenizer([formatted_text], return_tensors="pt", padding=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # 控制生成长度
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        output = tokenizer.batch_decode(outputs)[0]
        output = extract_assistant_response(output)
        original_model_response.append(output)
        print(f"基线测试，{output}")

    num_epoch = 3
    data_size = 10
    support_ratio = 0

    training_data = full_data[:data_size]
    support_data_size = int(data_size * support_ratio)

    # 修正：获取字符串而不是字典
    prompt_list = [data_formulate(data)["formatted_text"] for data in training_data]
    chosen_list = [data['support'] for data in training_data[:support_data_size]] + \
                  [data['oppose'] for data in training_data[support_data_size:]]
    rejected_list = [data['oppose'] for data in training_data[support_data_size:]] + \
                    [data['support'] for data in training_data[:support_data_size]]

    train_dataset = Dataset.from_dict({'prompt': prompt_list,
                                       'chosen': chosen_list,
                                       'rejected': rejected_list})

    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    from transformers import TrainingArguments
    from trl import DPOTrainer, DPOConfig
    from unsloth import is_bfloat16_supported

    # 关键修改：使用标准的DPOConfig，但明确设置长度参数
    dpo_config = DPOConfig(
        max_length=512,  # 明确设置
        max_prompt_length=128,  # 明确设置
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=num_epoch,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="./models",
        report_to="none",
        remove_unused_columns=False,  # 关键：必须设置为False
        dataloader_pin_memory=False,  # Windows兼容性
    )

    # 使用标准DPOTrainer，但添加错误处理
    dpo_trainer = DPOTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        ref_model=None,
        beta=0.1,
        args=dpo_config,
    )

    try:
        dpo_trainer.train()
    except Exception as e:
        print(f"DPO训练错误: {e}")
        print("尝试使用更小的batch size...")

    # 第12模块
    aligned_model_response = []
    for data in test_data:
        id = data['id']
        prompt = data['prompt']
        print(f"处理第 {data['id']} 条数据:")
        formatted_result = data_formulate(data)
        # 从字典中获取格式化后的文本
        formatted_text = formatted_result["formatted_text"]
        # 添加模型生成步骤
        inputs = tokenizer([formatted_text], return_tensors="pt", padding=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = dpo_trainer.model.generate(
                **inputs,
                max_new_tokens=128,  # 控制生成长度
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        output = tokenizer.batch_decode(outputs)[0]
        output = extract_assistant_response(output)
        aligned_model_response.append(output)
        print(f"对齐测试，{output}")

    ############################################################
    # (13) 保存结果到 JSON 文件 (包含对齐前 vs 对齐后)
    ############################################################
    student_id = "B12345678"
    dir_name = "./data"
    file_name = f"{dir_name}/{student_id}_hw7_epoch{num_epoch}_ratio{support_ratio}_size{data_size}.json"

    output_list = []
    for data in test_data:
        original_response = original_model_response.pop(0)
        aligned_response = aligned_model_response.pop(0)
        output_list.append({
            "id": data["id"],
            "prompt": data["prompt"],
            "original_response": original_response,
            "aligned_response": aligned_response
        })

    output_data = {"num_epoch": num_epoch,
                   "data_size": data_size,
                   "support_ratio": support_ratio,
                   "results": output_list}

    with open(file_name, "w") as output_file:
        json.dump(output_data, output_file, indent=4)


    ############################################################
    # (14) 手动测试新的 Prompt (可修改 system + prompt)
    ############################################################
    def make_prompt(system, prompt):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt


    system = "Your entire response must be 100 characters or less."
    prompt = "回答我，你是LORA适配器还是原始MODEL"  # TODO: 自定义问题

    inputs = make_prompt(system, prompt)
    outputs = model.generate(
        **tokenizer(inputs, return_tensors="pt").to("cuda"),
        max_new_tokens=512,
        do_sample=False,
    )
    output = tokenizer.batch_decode(outputs)[0]
    output = extract_assistant_response(output)
    print(output)


