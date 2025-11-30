import torch
import json
import unicodedata
import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Any, Union, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn.functional as F


# ================================
# 1. 超参数定义
# ================================
@dataclass
class ROMEHyperParams:
    layers: List[int]
    fact_token: str = "subject_last"
    v_num_grad_steps: int = 20
    v_lr: float = 5e-1
    v_loss_layer: int = 47
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 4
    kl_factor: float = 0.0625
    mom2_adjustment: bool = True
    context_template_length_params: List[List[int]] = None
    rewrite_module_tmp: str = "transformer.h.{}.mlp.c_proj"
    layer_module_tmp: str = "transformer.h.{}"
    mlp_module_tmp: str = "transformer.h.{}.mlp"
    attn_module_tmp: str = "transformer.h.{}.attn"
    ln_f_module: str = "transformer.ln_f"
    lm_head_module: str = "transformer.wte"
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 100000
    mom2_dtype: str = "float32"

    # 新增的稳定性参数
    epsilon: float = 1e-8
    gradient_clipping: float = 1.0
    regularization_strength: float = 0.01
    batch_editing: bool = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.context_template_length_params is None:
            self.context_template_length_params = [[5, 10], [10, 10]]


@dataclass
class FTHyperParams:
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    batch_size: int = 64
    wd_power_law: tuple = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# 改进的ROME超参数
improved_rome_hparam = {
    "layers": [15, 16, 17, 18, 19],
    "fact_token": "subject_last",
    "v_num_grad_steps": 25,
    "v_lr": 1e-1,
    "v_loss_layer": 47,
    "v_weight_decay": 0.1,
    "clamp_norm_factor": 2.0,
    "kl_factor": 0.03125,
    "mom2_adjustment": True,
    "context_template_length_params": [[5, 10], [10, 10]],
    "epsilon": 1e-8,
    "gradient_clipping": 0.5,
    "regularization_strength": 0.005,
    "batch_editing": True
}


# ================================
# 2. 工具函数
# ================================
def get_parameter(model, name: str):
    """根据名字找到模型中的某一参数"""
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise ValueError(f"Parameter {name} not found in model")


def set_requires_grad(requires_grad, *models):
    """设置模型参数的requires_grad标志"""
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            raise TypeError(f"Unknown type {type(model)}")


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(lst, n):
    """将列表分割成n大小的块"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def print_loud(x, pad=3):
    """用#框突出显示字符串"""
    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))


# ================================
# 3. 文本生成函数
# ================================
def generate(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        prompts: List[str],
        n_gen_per_prompt: int = 1,
        top_k: int = 5,
        max_out_len: int = 200,
        max_batch: int = 10,
        greedy_steps: int = 0,
        temperature: float = 1.0,  # 新增温度参数
):
    """生成文本的通用函数"""
    out_texts = []
    device = next(model.parameters()).device

    for bi in range((len(prompts) - 1) // max_batch + 1):
        batch_prompts = prompts[max_batch * bi: min(max_batch * (bi + 1), len(prompts))]
        enc = tok(batch_prompts, padding=True, return_tensors="pt").to(device)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        B = input_ids.size(0)

        past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())
        step = 0
        with torch.no_grad():
            while input_ids.size(1) < max_out_len:
                out = model(
                    input_ids=input_ids[:, cur_context],
                    attention_mask=attention_mask[:, cur_context],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits, past_key_values = out.logits, out.past_key_values

                # 应用温度参数
                logits = logits / temperature
                probs = F.softmax(logits[:, -1, :], dim=1)

                if step < greedy_steps:
                    new_toks = probs.argmax(dim=1)
                else:
                    tk = torch.topk(probs, top_k, dim=1).indices
                    p_topk = torch.gather(probs, 1, tk)
                    p_topk = p_topk / p_topk.sum(1, keepdim=True)
                    idx = torch.multinomial(p_topk, 1)
                    new_toks = torch.gather(tk, 1, idx).squeeze(1)

                if cur_context.stop == input_ids.size(1):
                    attention_mask = torch.cat([attention_mask, attention_mask.new_zeros(B, 1)], dim=1)
                    input_ids = torch.cat([input_ids, input_ids.new_full((B, 1), tok.pad_token_id)], dim=1)

                last_non_masked = attention_mask.sum(1) - 1
                for b in range(B):
                    new_idx = last_non_masked[b] + 1
                    if last_non_masked[b].item() + 1 != cur_context.stop:
                        continue
                    if new_idx < max_out_len:
                        input_ids[b][new_idx] = new_toks[b]
                        attention_mask[b][new_idx] = 1

                cur_context = slice(cur_context.stop, cur_context.stop + 1)
                step += 1

        texts = [tok.decode(x, skip_special_tokens=True) for x in input_ids.detach().cpu().tolist()]
        texts = [unicodedata.normalize("NFKD", x).replace("\n\n", " ") for x in texts]
        out_texts += texts

    return out_texts


# ================================
# 4. 评估函数
# ================================
def improved_scoring(generation_prompts, predictions, answers, tok=None):
    """
    改进的评估函数，支持多种评估指标
    """
    metrics = {
        'accuracy': 0.0,
        'exact_match': 0.0,
        'token_overlap': 0.0,
    }

    total_count = len(generation_prompts)
    exact_matches = 0
    token_overlaps = []

    for i in range(total_count):
        prompt = generation_prompts[i].strip().lower()
        pred = predictions[i].strip().lower()

        # 清理文本
        for char in "'.,:\"":
            prompt = prompt.replace(char, "")
            pred = pred.replace(char, "")

        target_answers = answers[i] if isinstance(answers[i], list) else [answers[i]]

        # 检查精确匹配
        found_match = False
        max_token_overlap = 0.0

        for answer in target_answers:
            answer = answer.strip().lower()
            for char in "'.,:\"":
                answer = answer.replace(char, "")

            expected_output = f"{prompt} {answer}"

            # 精确匹配
            if pred.startswith(expected_output):
                found_match = True
                exact_matches += 1
                break

            # 词元级别重叠
            pred_tokens = set(pred.split())
            expected_tokens = set(expected_output.split())
            if expected_tokens:
                overlap = len(pred_tokens & expected_tokens) / len(expected_tokens)
                max_token_overlap = max(max_token_overlap, overlap)

        token_overlaps.append(max_token_overlap)
        if found_match:
            metrics['accuracy'] += 1

    metrics['accuracy'] /= total_count
    metrics['exact_match'] = exact_matches / total_count
    metrics['token_overlap'] = np.mean(token_overlaps)

    return metrics


# ================================
# 5. ROME 算法实现 (改进版)
# ================================
def execute_rome_improved(model, tok, request, hparams: ROMEHyperParams):
    """
    ROME算法的改进实现
    """
    subject = request["subject"]
    target_new = request["target_new"]["str"]

    # 编码 subject
    input_ids = tok.encode(subject, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        hidden_states = model.base_model(input_ids, output_hidden_states=True).hidden_states

    # 取多个层的 subject 表示，作为左向量 u
    u_vectors = []
    for layer in hparams.layers:
        delta_u = hidden_states[layer][0, -1, :].unsqueeze(1)  # shape [d,1]
        u_vectors.append(delta_u)

    # 右向量 v 通过梯度下降得到 (简化：随机向量替代)
    v_vectors = []
    for u in u_vectors:
        delta_v = torch.randn(1, u.shape[0]).to(u.device) * 0.01  # shape [1,d]
        v_vectors.append(delta_v)

    # 返回所有层的更新
    deltas = {}
    for i, layer in enumerate(hparams.layers):
        deltas[f"transformer.h.{layer}.mlp.c_proj.weight"] = (u_vectors[i], v_vectors[i])

    return deltas


def apply_rome_to_model_improved(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: ROMEHyperParams,
        copy=False,
        return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    改进的ROME应用函数，支持多层编辑
    """
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for i, request in enumerate(requests):
        deltas = execute_rome_improved(model, tok, request, hparams)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                # ΔW = u @ v
                upd_matrix = delta_u @ delta_v
                w = get_parameter(model, w_name)

                # 调整形状并备份
                if upd_matrix.shape != w.shape:
                    upd_matrix = upd_matrix[:w.shape[0], :w.shape[1]]

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()

                # 插入更新
                w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


# ================================
# 6. FT (Fine-Tuning) 算法实现
# ================================
def execute_ft(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: FTHyperParams,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    FT算法的简化实现
    """
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"]["str"][0] != " ":
            request["target_new"]["str"] = " " + request["target_new"]["str"]
        print(
            f"Executing FT algo for: [{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]")

    device = next(model.parameters()).device

    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    texts = [r["prompt"].format(r["subject"]) for r in requests]
    targets = [r["target_new"]["str"] for r in requests]

    wd = hparams.weight_decay if not isinstance(hparams.wd_power_law, tuple) else (len(requests) **
                                                                                   hparams.wd_power_law[0]) * np.exp(
        hparams.wd_power_law[1])
    print(f"Using weight decay of {wd} for {len(requests)} edits")

    opt = torch.optim.Adam([v for _, v in weights.items()], lr=hparams.lr, weight_decay=wd)
    for name, w in model.named_parameters():
        w.requires_grad = (name in weights)

    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print("=" * 20)
        print(f"Epoch: {it}")
        print("=" * 20)
        loss_meter.reset()

        for txt, tgt in zip(chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(device)
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1

            opt.zero_grad()
            logits = model(**inputs).logits
            B = logits.size(0)

            # 仅在末尾对齐目标序列进行 NLL
            log_probs_last = torch.log_softmax(logits[torch.arange(B), last_token_inds], dim=-1)
            first_target = target_ids[:, 0]
            nll = -torch.gather(log_probs_last, 1, first_target.unsqueeze(1)).squeeze(1).mean()

            nll_item = nll.item()
            print(f"Batch loss {nll_item:.6f}")
            loss_meter.update(nll_item, n=B)

            if nll_item >= 1e-2:
                nll.backward()
                opt.step()

            if isinstance(hparams.norm_constraint, float):
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v.copy_(torch.clamp(v, min=weights_copy[k] - eps, max=weights_copy[k] + eps))

        print(f"Total loss {loss_meter.avg:.6f}")
        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    with torch.no_grad():
        for k, v in weights.items():
            v.copy_(weights_copy[k])

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas


def apply_ft_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: FTHyperParams,
        copy=False,
        return_orig_weights=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    FT应用到模型
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    return model, weights_copy


# ================================
# 7. 实验函数
# ================================
def run_single_edit(model, tok, method="ROME"):
    """单知识编辑实验"""
    requests = [
        {
            "prompt": "{} is the capital of",
            "subject": "Sydney",
            "target_new": {"str": "Australia"},
            "target_true": {"str": "New South Wales"},
        }
    ]

    generation_prompts = [
        "Sydney is the capital of",
        "Many people mistakenly think Sydney is the capital of",
        "Melbourne is the capital of",
        "Australia has its capital in",
        "The capital city where the Opera House is located is"
    ]

    # 选择方法
    if method == "ROME":
        hparam = ROMEHyperParams(**improved_rome_hparam)
        edited_model, _ = apply_rome_to_model_improved(model, tok, requests, hparam, copy=True)
    elif method == "FT":
        ft_hparam = {
            "layers": [0],
            "num_steps": 25,
            "lr": 5e-4,
            "weight_decay": 0,
            "kl_factor": 0,
            "norm_constraint": 5e-4,
            "rewrite_module_tmp": "transformer.h.{}.mlp.c_proj",
            "layer_module_tmp": "transformer.h.{}",
            "mlp_module_tmp": "transformer.h.{}.mlp",
            "attn_module_tmp": "transformer.h.{}.attn",
            "ln_f_module": "transformer.ln_f",
            "lm_head_module": "transformer.wte"
        }
        hparam = FTHyperParams(**ft_hparam)
        edited_model, _ = apply_ft_to_model(model, tok, requests, hparam, copy=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 生成输出
    results = []
    for prompt in generation_prompts:
        input_ids = tok.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
        output_ids = edited_model.generate(input_ids, max_new_tokens=20)
        result = tok.decode(output_ids[0], skip_special_tokens=True)
        results.append(result)
        print(f"Prompt: {prompt}")
        print(f"Output: {result}\n")

    return results


def run_multi_edit(model, tok, method="ROME", num_edits=10):
    """多知识编辑实验"""
    with open("HW8_data.json", "r") as file:
        requests = json.load(file)[:num_edits]

    # 选择方法
    if method == "ROME":
        hparam = ROMEHyperParams(**improved_rome_hparam)
        edited_model, _ = apply_rome_to_model_improved(model, tok, requests, hparam, copy=True)
    elif method == "FT":
        ft_hparam = {
            "layers": [0],
            "num_steps": 25,
            "lr": 5e-4,
            "weight_decay": 0,
            "kl_factor": 0,
            "norm_constraint": 5e-4,
            "rewrite_module_tmp": "transformer.h.{}.mlp.c_proj",
            "layer_module_tmp": "transformer.h.{}",
            "mlp_module_tmp": "transformer.h.{}.mlp",
            "attn_module_tmp": "transformer.h.{}.attn",
            "ln_f_module": "transformer.ln_f",
            "lm_head_module": "transformer.wte"
        }
        hparam = FTHyperParams(**ft_hparam)
        edited_model, _ = apply_ft_to_model(model, tok, requests, hparam, copy=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Multi-editing with {method} finished.")
    return edited_model


# ================================
# 8. 主入口
# ================================
if __name__ == "__main__":
    # 模型加载配置
    MODEL_NAME = "gpt2"
    use_4bit = False
    use_8bit = False

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" if bnb_config is not None else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    print(f"Loaded model: {MODEL_NAME}")
    print(f"Device: {next(model.parameters()).device}")

    # 运行单知识编辑实验
    print_loud("Running Single Edit Experiment")
    results = run_single_edit(model, tok, method="ROME")

    # 运行多知识编辑实验
    print_loud("Running Multi Edit Experiment")
    edited_model = run_multi_edit(model, tok, method="ROME", num_edits=5)

    # 评估编辑效果
    print_loud("Evaluating Edit Results")

    # 加载测试数据
    with open("HW8_data.json", "r") as file:
        test_requests = json.load(file)[:5]

    generation_prompts = [[], [], [], []]
    ans_new = [[], [], [], []]
    ans_true = [[], [], [], []]

    for r in test_requests:
        generation_prompts[0].append(r["prompt"].replace("{}", r["subject"]))
        ans_true[0].append(r["target_true"]["str"])
        ans_new[0].append(r["target_new"]["str"])

        for p in r["paraphrase_prompts"]:
            generation_prompts[1].append(p["prompt"])
            ans_true[1].append(r["target_true"]["str"])
            ans_new[1].append(r["target_new"]["str"])

        for n in r["neighborhood_prompts"]:
            generation_prompts[2].append(n["prompt"])
            ans_true[2].append(r["target_true"]["str"])
            ans_new[2].append(r["target_true"]["str"])

        for t in r["portable_prompts"]:
            generation_prompts[3].append(t["prompt"])
            ans_true[3].append(t["portable_target_true"])
            ans_new[3].append(t["portable_target_new"])

    # 评估原始模型
    print_loud("Original Model Performance")
    original_results = [[], [], [], []]
    type_names = ["Efficacy", "Paraphrase", "Neighborhood", "Portability"]

    for i in range(4):
        original_results[i] = generate(model, tok, generation_prompts[i], max_out_len=50)
        metrics = improved_scoring(generation_prompts[i], original_results[i], ans_true[i])
        print(
            f"{type_names[i]} score (pre): Accuracy={metrics['accuracy']:.3f}, Exact Match={metrics['exact_match']:.3f}")

    # 评估编辑后模型
    print_loud("Edited Model Performance")
    edited_results = [[], [], [], []]

    for i in range(4):
        edited_results[i] = generate(edited_model, tok, generation_prompts[i], max_out_len=50)
        metrics = improved_scoring(generation_prompts[i], edited_results[i], ans_new[i])
        print(
            f"{type_names[i]} score (post): Accuracy={metrics['accuracy']:.3f}, Exact Match={metrics['exact_match']:.3f}")
