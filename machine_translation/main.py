import sys
import pdb
import pprint
import logging
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils
import matplotlib.pyplot as plt
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast
import shutil
import sacrebleu
import subprocess
import Model_Architecture as Model

# region 随机种子设置
seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# endregion

#region 配置信息
# 配置训练参数
config = Namespace(
    # 数据和模型路径配置
    datadir="./DATA/data-bin/ted2020",           # 训练数据目录
    savedir="./checkpoints/rnn",                # 模型检查点保存目录
    source_lang="en",                           # 源语言（如英语）
    target_lang="zh",                           # 目标语言（如中文）

    # 数据处理配置
    num_workers=2,                              # 数据加载的CPU线程数
    max_tokens=8192,                            # 每个批次的最大token数量
    accum_steps=2,                              # 梯度累积步数，用于模拟更大的批次大小

    # 学习率调度配置（Noam算法）
    lr_factor=2.0,                              # 学习率缩放因子
    lr_warmup=4000,                             # 学习率预热步数

    # 训练稳定性配置
    clip_norm=1.0,                              # 梯度裁剪阈值，防止梯度爆炸

    # 训练周期配置
    max_epoch=30,                               # 最大训练轮数
    start_epoch=1,                              # 起始训练轮数（用于恢复训练）

    # 推理配置（生成翻译结果）
    beam=5,                                     # 束搜索宽度，用于解码时的多路径探索
    max_len_a=1.2,                              # 生成序列最大长度参数a（max_len = a*src_len + b）
    max_len_b=10,                               # 生成序列最大长度参数b
    post_process="sentencepiece",               # 后处理方式，如移除sentencepiece标记

    # 检查点配置
    keep_last_epochs=5,                         # 保留的最近检查点数量
    resume=None,                                # 恢复训练的检查点名称

    # 日志和可视化配置
    use_wandb=False,                            # 是否使用WandB进行实验跟踪
)

# 配置日志系统
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",  # 日志格式：时间|级别|模块|消息
    datefmt="%Y-%m-%d %H:%M:%S",                     # 日期时间格式
    level="INFO",                                     # 日志级别：INFO, DEBUG, WARNING, ERROR
    stream=sys.stdout,                                # 输出到标准输出
)

# 创建项目日志记录器
proj = "hw5.seq2seq"
logger = logging.getLogger(proj)

# 初始化WandB实验跟踪（如果启用）
if config.use_wandb:
    import wandb
    # 初始化WandB项目，使用保存目录名作为实验名称
    wandb.init(
        project=proj,
        name=Path(config.savedir).stem,
        config=vars(config)  # 将Namespace对象转换为字典
    )

# 检查CUDA环境并获取设备信息
cuda_env = utils.CudaEnvironment()  # 创建CUDA环境对象
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])  # 打印CUDA环境信息
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 设置计算设备

# endregion

#region 数据集处理
# setup task
task_cfg = TranslationConfig( #    创建任务的配置对象
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap",
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg) # 初始化任务

logger.info("loading data for epoch 1")
task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
task.load_dataset(split="valid", epoch=1)

sample = task.dataset("valid")[1] #从获取索引为 1 的样本
# pprint.pprint(sample) #打印原始样本数据（通常是 token ID 列表）
#
# pprint.pprint(
#     "Source: " + \
#     task.source_dictionary.string(
#         sample['source'],
#         config.post_process,
#     ) #将 token ID 列表转换为文本字符串
# )
# pprint.pprint(
#     "Target: " + \
#     task.target_dictionary.string(
#         sample['target'],
#         config.post_process,
#     )
# )

#Dataset Iterator
"""
GPU 内存优化：通过控制每个 batch 的 token 数量（而非句子数量）动态调整 batch size
训练稳定性：每个 epoch 进行不同的 shuffling 防止模型记忆顺序
计算效率：统一 batch 内序列长度，利用 GPU 并行能力
自回归训练：通过位移处理实现 teacher forcing机制(强制使用真实的标签序列 token 作为输入)
"""
def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(dataset=task.dataset(split),max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),#删除超过 max_positions 的超长序列
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
    )
    return batch_iterator

# if __name__=='__main__':
#     demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
#     demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True) #调用next_epoch_itr()会生成下一个全新的迭代器
#     sample = next(demo_iter) #取出下一个批次（batch）的数据
#     print("11111111111",sample) #包含输入张量、目标张量和其他元数据的字典。
#endregion

#region 模型初始化
# # HINT: transformer architecture
def build_model(args, task):
    """ build a model instance based on hyperparameters"""
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
#
    # 输入阶段 token embeddings 将输入的 token 转换为连续向量表示
    # 初始化嵌入层
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())

    # encoder decoder
    # HINT: TODO: switch to TransformerEncoder & TransformerDecoder
    encoder = Model.RNNEncoder(args, src_dict, encoder_embed_tokens)
    decoder = Model.RNNDecoder(args, tgt_dict, decoder_embed_tokens)
    model = Model.Seq2Seq(args, encoder, decoder) # 将编码器和解码器组合

    # 针对不同类型module的采用不同的初始化策略
    def init_params(module):
        from fairseq.modules import MultiheadAttention
        if isinstance(module, nn.Linear): #正态分布
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase): #均匀分布
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)

    # weight initialization
    model.apply(init_params)
    return model
#endregion

#region 架构配置
arch_args = Namespace(
    encoder_embed_dim=256,
    encoder_ffn_embed_dim=512,
    encoder_layers=2,
    decoder_embed_dim=256,
    decoder_ffn_embed_dim=1024,
    decoder_layers=2,
    share_decoder_input_output_embed=True,
    dropout=0.3,
)
# # HINT: these patches on parameters for Transformer
# def add_transformer_args(args):
#     args.encoder_attention_heads=4
#     args.encoder_normalize_before=True

#     args.decoder_attention_heads=4
#     args.decoder_normalize_before=True

#     args.activation_fn="relu"
#     args.max_source_positions=1024
#     args.max_target_positions=1024

#     # patches on default parameters for Transformer (those not set above)
#     from fairseq.models.transformer import base_architecture
#     base_architecture(arch_args)

# add_transformer_args(arch_args)
if config.use_wandb:
    wandb.config.update(vars(arch_args))
model = build_model(arch_args, task)
logger.info(model)
#endregion

#region Optimization
#损失函数
class LabelSmoothedCrossEntropyCriterion(nn.Module):
    """
    标签平滑的交叉熵损失函数
    参数:
        smoothing (float): 标签平滑因子，范围[0,1]，0表示不使用平滑
        ignore_index (int, optional): 忽略的目标索引(如填充符)，损失计算中会忽略
        reduce (bool, optional): 是否对损失求和，默认为True
    """

    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, lprobs, target):
        """
        前向传播计算损失

        参数:
            lprobs (torch.Tensor): 对数概率，形状为[batch_size, seq_len, vocab_size]
            target (torch.Tensor): 目标标签，形状为[batch_size, seq_len]

        返回:
            torch.Tensor: 计算得到的损失值
        """
        # 确保目标维度与对数概率匹配
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)

        # NLL 损失计算 计算负对数似然损失(NLL)，相当于one-hot标签下的交叉熵
        nll_loss = -lprobs.gather(dim=-1, index=target)

        # 计算平滑损失，即对所有标签的对数概率求和
        # 这表示模型应该对非目标标签也有一定的置信度
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        # 处理填充符号，将填充位置的损失置为0
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)

        # 对损失进行求和(如果需要)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        # 计算最终损失：原始损失与平滑损失的加权和
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss
# 示例：创建一个标签平滑损失函数实例
# 通常，smoothing=0.1是一个不错的选择
criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),  # 忽略填充符号
)

#优化器
class NoamOpt:
    """
    优化器包装器，实现学习率的动态调整
    Adam + lr 调度
    参数:
        model_size (int): 模型的嵌入维度大小
        factor (float): 缩放因子，调整整体学习率大小
        warmup (int): 预热步数，在此步数内学习率逐渐增加
        optimizer (torch.optim.Optimizer): 基础优化器，如Adam
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0  # 当前步数
        self.warmup = warmup  # 预热步数
        self.factor = factor  # 缩放因子
        self.model_size = model_size  # 模型维度
        self._rate = 0  # 当前学习率

    @property #将类的方法转换为属性，使得我们可以像访问属性一样访问方法
    def param_groups(self):
        """访问优化器的参数组"""
        return self.optimizer.param_groups

    def multiply_grads(self, c):
        """
        将梯度乘以一个常数c
        参数:
            c (float): 梯度缩放系数
        """
        for group in self.param_groups:
            for p in group['params']: #group['params'] 是该组中的参数（即模型的 nn.Parameter 对象
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def step(self):
        """
        更新模型参数和学习率
        """
        self._step += 1  # 增加步数
        rate = self.rate()  # 计算当前学习率

        # 更新所有参数组的学习率
        for p in self.param_groups:
            p['lr'] = rate

        self._rate = rate  # 记录当前学习率
        self.optimizer.step()  # 执行参数更新

    def rate(self, step=None):
        """
        计算学习率，实现如下公式:
        lr = factor * (model_size^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)))

        参数:
            step (int, optional): 指定步数，默认使用当前步数

        返回:
            float: 计算得到的学习率
        """
        if step is None:
            step = self._step
        return 0 if not step else self.factor * \
                                  (self.model_size ** (-0.5) *
                                   min(step ** (-0.5), step * self.warmup ** (-1.5)))
#优化过程可视化
optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim,
    factor=config.lr_factor,
    warmup=config.lr_warmup,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))
plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
#endregion

# 训练过程
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
        """
        训练一个完整的轮次(epoch)

        参数:
            epoch_itr: 数据迭代器
            model: 训练的模型
            task: 任务对象
            criterion: 损失函数(如标签平滑交叉熵)
            optimizer: 优化器(如NoamOpt包装的AdamW)
            accum_steps: 梯度累积步数，默认为1
        """
        # 获取下一个轮次的数据迭代器并打乱
        itr = epoch_itr.next_epoch_itr(shuffle=True)

        # 梯度累积: 每accum_steps个样本更新一次参数
        itr = iterators.GroupedIterator(itr, accum_steps)

        # 统计信息
        stats = {"loss": []}

        # 自动混合精度训练工具
        scaler = GradScaler()

        # 设置模型为训练模式
        model.train()

        # 进度条显示
        progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)

        # 遍历批次
        for samples in progress:
            # 梯度清零
            model.zero_grad()
            accum_loss = 0
            sample_size = 0

            # 梯度累积: 处理每个批次中的样本
            for i, sample in enumerate(samples):
                # 第一个样本后清空CUDA缓存，减少OOM风险
                if i == 1:
                    torch.cuda.empty_cache()

                # 将样本移至GPU
                sample = utils.move_to_cuda(sample, device=device)
                target = sample["target"]
                sample_size_i = sample["ntokens"]
                sample_size += sample_size_i

                # 混合精度训练
                with autocast():
                    # 前向传播
                    net_output = model.forward(**sample["net_input"])
                    lprobs = F.log_softmax(net_output[0], -1)

                    # 计算损失
                    loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))

                    # 记录累积损失
                    accum_loss += loss.item()

                    # 反向传播
                    scaler.scale(loss).backward()

            # 梯度处理和参数更新
            scaler.unscale_(optimizer)

            # 梯度归一化，防止梯度爆炸
            optimizer.multiply_grads(1 / (sample_size or 1.0))

            # 梯度裁剪，限制梯度范数
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)

            # 优化器更新参数
            scaler.step(optimizer)
            scaler.update()

            # 日志记录
            loss_print = accum_loss / sample_size
            stats["loss"].append(loss_print)
            progress.set_postfix(loss=loss_print)

            # 使用wandb记录训练指标
            if config.use_wandb:
                wandb.log({
                    "train/loss": loss_print,
                    "train/grad_norm": gnorm.item(),
                    "train/lr": optimizer.rate(),
                    "train/sample_size": sample_size,
                })

        # 计算平均损失并记录
        loss_print = np.mean(stats["loss"])
        logger.info(f"training loss: {loss_print:.4f}")
        return stats

    # Fairseq的束搜索生成器
    # 给定模型和输入序列，通过束搜索生成翻译假设
    sequence_generator = task.build_generator([model], config)

    def decode(toks, dictionary):
        """
        将模型生成的token张量转换为人类可读的文本句子

        参数:
            toks (torch.Tensor): token索引张量
            dictionary: Fairseq词典对象，用于映射token索引到词表中的词

        返回:
            str: 解码后的文本句子
        """
        # 将张量转换为人类可读的句子
        s = dictionary.string(
            toks.int().cpu(),  # 将token张量转为CPU上的整数类型
            config.post_process,  # 后处理方式，如添加空格等
        )
        return s if s else "<unk>"  # 如果解码结果为空，返回未知标记

    def inference_step(sample, model):
        """
        执行一个推理步骤，生成翻译结果并收集输入、假设和参考句子

        参数:
            sample: 包含输入和目标数据的样本字典
            model: 训练好的模型

        返回:
            tuple: 包含源句子列表、生成的假设句子列表和参考句子列表的元组
        """
        # 使用束搜索生成翻译结果
        gen_out = sequence_generator.generate([model], sample)

        srcs = []  # 源语言句子列表
        hyps = []  # 生成的假设句子列表
        refs = []  # 参考句子列表

        # 处理每个样本
        for i in range(len(gen_out)):
            # 对于每个样本，收集输入、假设和参考，稍后用于计算BLEU等指标

            # 源语言句子解码
            srcs.append(decode(
                utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
                task.source_dictionary,
            ))

            # 生成的假设句子解码（取束搜索中的最优结果）
            hyps.append(decode(
                gen_out[i][0]["tokens"],  # 0表示使用束搜索中的top-1假设
                task.target_dictionary,
            ))

            # 参考句子解码
            refs.append(decode(
                utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
                task.target_dictionary,
            ))

        return srcs, hyps, refs

    # 描述模型的实际性能
    def validate(model, task, criterion, log_to_wandb=True):
        """
        模型验证函数：在验证集上评估模型性能，计算损失和BLEU分数

        参数:
            model: 待验证的模型
            task: 任务对象（包含数据加载、词典等信息）
            criterion: 损失函数（与训练时一致）
            log_to_wandb (bool): 是否将结果记录到wandb，默认True

        返回:
            dict: 包含验证指标的字典，包括损失、BLEU分数、源句子、假设句子、参考句子
        """
        logger.info('begin validation')

        # 加载验证集数据迭代器（不打乱顺序，保证结果可复现）
        itr = load_data_iterator(
            task, "valid", 1, config.max_tokens, config.num_workers
        ).next_epoch_itr(shuffle=False)

        # 初始化统计信息字典
        stats = {"loss": [], "bleu": 0, "srcs": [], "hyps": [], "refs": []}
        srcs = []  # 存储源语言句子
        hyps = []  # 存储模型生成的假设句子
        refs = []  # 存储参考（真实）句子

        # 设置模型为评估模式（关闭dropout等训练特有的层）
        model.eval()

        # 进度条显示验证过程
        progress = tqdm.tqdm(itr, desc=f"validation", leave=False)

        # 关闭梯度计算（节省显存，加速验证）
        with torch.no_grad():
            for i, sample in enumerate(progress):
                # 1. 计算验证集损失
                sample = utils.move_to_cuda(sample, device=device)  # 移至GPU
                net_output = model.forward(**sample["net_input"])  # 模型前向传播

                # 计算对数概率和损失（与训练时一致的流程）
                lprobs = F.log_softmax(net_output[0], -1)
                target = sample["target"]
                sample_size = sample["ntokens"]  # 样本中有效token数量
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size

                # 记录损失并更新进度条
                progress.set_postfix(valid_loss=loss.item())
                stats["loss"].append(loss)

                # 2. 执行推理，生成翻译结果
                s, h, r = inference_step(sample, model)  # 调用之前定义的推理函数
                srcs.extend(s)  # 累加源句子
                hyps.extend(h)  # 累加模型生成的假设
                refs.extend(r)  # 累加参考句子

        # 3. 计算验证集整体指标
        # 确定BLEU计算的分词方式（中文用'zh'，其他语言用'13a'标准分词）
        tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'

        # 计算平均损失
        stats["loss"] = torch.stack(stats["loss"]).mean().item()

        # 计算BLEU分数（使用sacrebleu库，支持多语言分词）
        stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)

        # 存储句子数据
        stats["srcs"] = srcs
        stats["hyps"] = hyps
        stats["refs"] = refs

        # 4. 日志记录
        if config.use_wandb and log_to_wandb:
            wandb.log({
                "valid/loss": stats["loss"],
                "valid/bleu": stats["bleu"].score,  # BLEU分数（0-100）
            }, commit=False)

        # 随机打印一个样本的结果，直观展示模型性能
        showid = np.random.randint(len(hyps))
        logger.info("example source: " + srcs[showid])
        logger.info("example hypothesis: " + hyps[showid])
        logger.info("example reference: " + refs[showid])

        # 打印验证集整体结果
        logger.info(f"validation loss:\t{stats['loss']:.4f}")
        logger.info(stats["bleu"].format())  # 格式化显示BLEU分数及细节

        return stats

    #保存模型权重
    def validate_and_save(model, task, criterion, optimizer, epoch, save=True):
        """
        验证模型性能并保存检查点

        参数:
            model: 待验证的模型
            task: 任务对象
            criterion: 损失函数
            optimizer: 优化器
            epoch: 当前轮次
            save: 是否保存检查点，默认为True

        返回:
            dict: 包含验证统计信息的字典
        """
        # 调用验证函数获取性能指标
        stats = validate(model, task, criterion)
        bleu = stats['bleu']
        loss = stats['loss']

        if save:
            # 创建保存目录
            savedir = Path(config.savedir).absolute()
            savedir.mkdir(parents=True, exist_ok=True)

            # 构建检查点数据
            check = {
                "model": model.state_dict(),  # 模型参数
                "stats": {"bleu": bleu.score, "loss": loss},  # 验证指标
                "optim": {"step": optimizer._step}  # 优化器步数
            }

            # 保存当前轮次的检查点
            torch.save(check, savedir / f"checkpoint{epoch}.pt")
            shutil.copy(savedir / f"checkpoint{epoch}.pt", savedir / f"checkpoint_last.pt")
            logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")

            # 保存当前轮次的样本
            with open(savedir / f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w",encoding="utf-8") as f:
                for s, h in zip(stats["srcs"], stats["hyps"]):
                    f.write(f"{s}\t{h}\n")

            # 保存最佳BLEU分数的模型
            if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
                validate_and_save.best_bleu = bleu.score
                torch.save(check, savedir / f"checkpoint_best.pt")
                logger.info(f"saved best checkpoint: {savedir}/checkpoint_best.pt with BLEU: {bleu.score}")

            # 删除旧的检查点，只保留最近的几个
            del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
            if del_file.exists():
                del_file.unlink()
                logger.info(f"deleted old checkpoint: {del_file}")

        return stats
    #加载模型权重
    def try_load_checkpoint(model, optimizer=None, name=None):
        """
        尝试加载模型检查点

        参数:
            model: 待加载参数的模型
            optimizer: 优化器(可选)
            name: 检查点名称(可选)，默认为"checkpoint_last.pt"

        返回:
            None
        """
        # 构建检查点路径
        name = name if name else "checkpoint_last.pt"
        checkpath = Path(config.savedir) / name

        if checkpath.exists():
            # 加载检查点数据
            check = torch.load(checkpath)

            # 加载模型参数
            model.load_state_dict(check["model"])

            # 获取统计信息
            stats = check["stats"]

            # 恢复优化器状态(如果提供)
            step = "unknown"
            if optimizer != None:
                optimizer._step = step = check["optim"]["step"]

            logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
        else:
            logger.info(f"no checkpoints found at {checkpath}!")

    # region 训练模型
    # 将模型和损失函数移至指定设备（如GPU）
    model = model.to(device=device)
    criterion = criterion.to(device=device)

    # 日志记录关键组件信息
    logger.info("task: {}".format(task.__class__.__name__))  # 任务类型
    logger.info("encoder: {}".format(model.encoder.__class__.__name__))  # 编码器类型
    logger.info("decoder: {}".format(model.decoder.__class__.__name__))  # 解码器类型
    logger.info("criterion: {}".format(criterion.__class__.__name__))  # 损失函数类型
    logger.info("optimizer: {}".format(optimizer.__class__.__name__))  # 优化器类型

    # 日志记录模型参数数量
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),  # 总参数数量
            sum(p.numel() for p in model.parameters() if p.requires_grad),  # 可训练参数数量
        )
    )

    # 日志记录训练配置
    logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")

    # 加载训练数据迭代器（从指定epoch开始）
    epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)

    # 尝试加载检查点（若有），支持断点续训
    try_load_checkpoint(model, optimizer, name=config.resume)

    # 训练主循环：直到达到最大epoch数
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # 1. 训练一个epoch
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)

        # 2. 验证并保存模型
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)

        # 3. 记录当前epoch结束信息
        logger.info("end of epoch {}".format(epoch_itr.epoch))

        # 4. 加载下一个epoch的训练数据迭代器
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)
    #endregion 模型训练结束

    # region 生成可供分析或应用的模型预测文本
    # 第一步：执行 average_checkpoints.py 命令
    checkdir = config.savedir
    command = [
        'python',
        './fairseq/scripts/average_checkpoints.py',
        '--inputs', checkdir,
        '--num-epoch-checkpoints', '5',
        '--output', os.path.join(checkdir, 'avg_last_5_checkpoint.pt')
    ]
    subprocess.run(command, check=True)

    # 第二步：加载模型并验证
    try_load_checkpoint(model, name="checkpoint_best.pt")
    validate(model, task, criterion, log_to_wandb=False)

    # 第三步：生成预测结果
    def generate_prediction(model, task, split="test", outfile="./prediction.txt"):
        """
        在指定数据集上生成预测结果并保存

        参数:
            model: 训练好的模型
            task: 任务对象（包含数据加载、词典等信息）
            split: 数据集名称，默认为"test"（测试集）
            outfile: 预测结果保存路径，默认为"./prediction.txt"
        """
        # 加载指定数据集（如test集）
        task.load_dataset(split=split, epoch=1)

        # 创建数据集迭代器（不打乱顺序，确保结果顺序可复现）
        itr = load_data_iterator(
            task, split, 1, config.max_tokens, config.num_workers
        ).next_epoch_itr(shuffle=False)

        idxs = []  # 存储样本索引，用于恢复原始顺序
        hyps = []  # 存储模型生成的预测结果

        # 设置模型为评估模式（关闭dropout等训练层）
        model.eval()

        # 进度条显示预测过程
        progress = tqdm.tqdm(itr, desc=f"prediction")

        # 关闭梯度计算（节省显存，加速预测）
        with torch.no_grad():
            for i, sample in enumerate(progress):
                # 将样本移至GPU（若使用）
                sample = utils.move_to_cuda(sample, device=device)

                # 执行推理，生成预测结果（仅关注假设句子hyps）
                _, h, _ = inference_step(sample, model)  # 忽略源句子和参考句子

                # 累积预测结果和样本索引
                hyps.extend(h)
                idxs.extend(list(sample['id']))  # sample['id']记录原始数据的索引

        # 按原始数据顺序排序预测结果（因批次处理可能打乱顺序）
        # 通过样本索引恢复预处理前的原始顺序
        hyps = [x for _, x in sorted(zip(idxs, hyps))]

        # 将预测结果写入文件
        with open(outfile, "w", encoding="utf-8") as f:
            for h in hyps:
                f.write(h + "\n")

    # 调用函数生成预测结果（默认使用test集，保存到./prediction.txt）
    generate_prediction(model, task)
    # endregion 生成预测结果结束



