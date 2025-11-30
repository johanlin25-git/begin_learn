import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (FairseqEncoder,FairseqIncrementalDecoder,FairseqEncoderDecoderModel)

# 训练时可并行计算，推理时必须串行生成
# 把输入序列转成有用的编码表示
# 最终需要的是词表中每个 token 的概率
class RNNEncoder(FairseqEncoder):
    """
    基于双向GRU的序列编码器，用于将源语言序列转换为连续表示
    编码器负责将输入序列（如源语言句子）编码为包含全局语义的向量
    主要功能：
    1. 词嵌入：将输入token转换为向量表示
    2. 双向GRU编码：捕获序列中的上下文信息
    3. 状态合并：处理双向编码的隐藏状态
    输入：
        src_tokens：表示英语的整数序列，例如 1、28、29、205、2
    输出：
        outputs：每个时间步长的 RNN 输出，可进一步处理 Attention
        final_hiddens：每个时间步长的隐藏状态，将传递给解码器进行解码
        encoder_padding_mask：这告诉解码器要忽略哪个位置
    """
    def __init__(self, args, dictionary, embed_tokens):
        """
        初始化编码器
        参数:
            args: 命令行参数，包含模型配置信息
            dictionary: 词典对象，用于获取填充索引
            embed_tokens: 词嵌入层，通常由模型外部传入
        """
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens  # 词嵌入层

        # 获取模型配置参数
        self.embed_dim = args.encoder_embed_dim  # 嵌入维度
        self.hidden_dim = args.encoder_ffn_embed_dim  # 隐藏层维度
        self.num_layers = args.encoder_layers  # GRU层数

        # 正则化模块
        self.dropout_in_module = nn.Dropout(args.dropout)  # 层归一化 (LayerNorm)

        # 双向GRU编码器 多头注意力（Multi-Head Attention)
        self.rnn = nn.GRU(   # 每个时间步产生两个隐藏状态 directions=2
            self.embed_dim,  # 输入维度
            self.hidden_dim,  # 隐藏层维度
            self.num_layers,  # 层数
            dropout=args.dropout,  # dropout率
            batch_first=False,  # 输入格式：[seq_len, batch, embed_dim]
            bidirectional=True  # 使用双向GRU
        )

        self.dropout_out_module = nn.Dropout(args.dropout)  # 输出dropout
        self.padding_idx = dictionary.pad()  # 填充token的索引

    def combine_bidir(self, outs, bsz: int):
        """
        合并双向GRU的隐藏状态
        将[num_layers*directions, batch, hidden]格式的张量
        转换为[num_layers, batch, directions*hidden]格式
        """
        # 合并双向状态 将前向和后向状态在批次维度后相邻排列
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def forward(self, src_tokens, **unused):
        """
        编码器前向传播函数
        参数:
            src_tokens: 源语言token序列，形状为[batch_size, seq_len]
        返回:
            元组(outputs, final_hiddens, encoder_padding_mask):
                - outputs: 各时间步的输出，形状为[seq_len, batch, hidden*directions]
                - final_hiddens: 最终隐藏状态，形状为[num_layers, batch, hidden*directions]
                - encoder_padding_mask: 填充位置掩码，形状为[seq_len, batch]
        """
        bsz, seqlen = src_tokens.size()  # 获取批次大小和序列长度

        # Input Embedding（输入嵌入层）
        x = self.embed_tokens(src_tokens)

        # Input Embedding 后的预处理
        x = self.dropout_in_module(x)  # 应用输入dropout
        x = x.transpose(0, 1)  # 转换为时序优先格式 [seq_len, batch, embed_dim]

        # Multi-Head Attention + Feed Forward
        h0 = x.new_zeros(2 * self.num_layers, bsz, self.hidden_dim) # 初始化双向GRU的隐藏状态
        # x: 各时间步的输出，形状为[seq_len, batch, hidden*directions]
        # final_hiddens: 最终隐藏状态，形状为[num_layers*directions, batch, hidden]
        x, final_hiddens = self.rnn(x, h0)

        # Add & Norm 等后处理
        outputs = self.dropout_out_module(x)
        # 合并双向GRU的隐藏状态
        final_hiddens = self.combine_bidir(final_hiddens, bsz)

        # Padding Mask 处理 忽略无效字符（如 填充含义<PAD>）
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t() # 创建填充位置掩码 [seq_len, batch]

        # 返回编码器输出
        return tuple(
            (
                outputs,  # 序列各位置的编码表示
                final_hiddens,  # 最终隐藏状态
                encoder_padding_mask,  # 填充位置掩码
            )
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        束搜索调整批次顺序的场景
        候选序列 (candidates)：当前步骤得分最高的部分序列
        原始输入: [A, B, C]   (batch_size=3)
        候选序列可能:
          候选1: 来自A的扩展
          候选2: 来自C的扩展
          候选3: 来自B的扩展
        此时需要将编码器输出重排为 [A, C, B] 的顺序
        """
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),  # 重新排序输出序列
                encoder_out[1].index_select(1, new_order),  # 重新排序最终隐藏状态
                encoder_out[2].index_select(1, new_order),  # 重新排序填充掩码
            )
        )

# 捕捉序列间的依赖关系
class AttentionLayer(nn.Module):
    """
    简化版注意力机制层（类似Transformer中的Scaled Dot-Product Attention）
    核心公式：Attention(Q,K,V) = softmax(Q·K^T/√d)·V
    本实现特点：
        - 未显式分离Q、K、V，而是直接用input_proj处理输入 查询矩阵变换 键矩阵变换 值矩阵变换
        - 用encoder_outputs同时作为K和V
        - 缺少√d缩放因子（简化实现）
        - 支持padding mask以忽略填充位置
    """

    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        """
        参数:
            input_embed_dim: 输入(query)的维度 作为Q
            source_embed_dim: 源序列(encoder_outputs)的维度 作为K和V
            output_embed_dim: 输出的维度
            bias: 是否使用偏置项
        """
        super().__init__()

        # 线性投影层，将输入转换到与源序列相同的维度空间
        # 相当于标准注意力机制中的Q = X·W^q 和 K = X·W^k 的合并（简化实现）
        self.input_proj = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)

        # 输出投影层，将注意力输出与原始输入拼接后投影到目标维度
        # 对应公式中的最终线性变换：output = W^o·[Attention;X]
        self.output_proj = nn.Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )

    def forward(self, inputs, encoder_outputs, encoder_padding_mask):
        """
        参数:
            inputs: 查询的张量，形状为 [T, B, input_embed_dim] (T=输入的序列长度, B=批量大小)
            encoder_outputs: 键和值的张量，形状为 [S, B, source_embed_dim] (S=源序列长度)
            encoder_padding_mask: 填充掩码，形状为 [S, B]，值为True的位置表示需要忽略

        返回:
            output: 注意力输出，形状为 [T, B, output_embed_dim]
            attn_weights: 注意力权重，形状为 [B, T, S]
        """
        # 1. 转换为batch-first格式 [B, T, dim] 和 [B, S, dim]
        inputs = inputs.transpose(1, 0)  # [B, T, input_embed_dim]
        encoder_outputs = encoder_outputs.transpose(1, 0)  # [B, S, source_embed_dim]
        encoder_padding_mask = encoder_padding_mask.transpose(1, 0)  # [B, S]

        # 2. 线性投影：将输入转换到与源序列相同的维度空间
        # 相当于标准注意力中的Q = X·W^q 注意：这里用encoder_outputs直接作为K和V，未进行额外投影
        x = self.input_proj(inputs)  # Q = [B, T, source_embed_dim]

        # 3. 计算注意力得分 (Q·K^T)
        # ^a = [B, T, source_embed_dim] x [B, source_embed_dim, S] -> [B, T, S]
        # 这里的x相当于Q，encoder_outputs相当于K
        attn_scores = torch.bmm(x, (encoder_outputs.transpose(1, 2))) #先实现K的结合

        # 4. “应用”填充“掩码”：将填充位置的注意力得分“设为负无穷”
        # 确保softmax后这些位置的权重接近0，不参与后续计算
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1)  # [B, 1, S]
            attn_scores = (
                attn_scores.float()  # 转为float32以支持masked_fill_操作（FP16兼容性）
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)  # 转回原始数据类型
            )

        # 5. 应用softmax生成注意力权重
        # 将得分归一化为概率分布，dim=-1表示在源序列S维度上进行softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, “S”]

        # 6. 加权求和b^1 = Σ α'_{1,i} v^i  获取上下文向量 Attention(Q,K,V) = softmax(Q·K^T)·V
        # [B, T, S] x [B, S, source_embed_dim] -> [B, T, source_embed_dim] 这里encoder_outputs相当于V
        x = torch.bmm(attn_weights, encoder_outputs)

        # 7. 拼接注意力输出与原始输入，增强表达能力
        # 特征融合:将注意力信息与原始输入特征结合，类似Residual Connection的思想
        x = torch.cat((x, inputs), dim=-1)  # [B, T, (source_embed_dim + input_embed_dim)]

        # 8. 最终线性变换并应用tanh激活函数
        # 将拼接后的特征投影到output_embed_dim维度，并限制输出范围在[-1, 1]
        x = torch.tanh(self.output_proj(x))  # [B, T, output_embed_dim]

        # 9. 恢复原始维度顺序 [T, B, output_embed_dim]
        return x.transpose(1, 0), attn_weights

# 接收编码器输出和先前生成的token，逐步生成目标序列
class RNNDecoder(FairseqIncrementalDecoder):
    """
    基于RNN序列解码器的输出的全局语义和历史生成的 token，通过自回归方式逐步生成符合目标任务的输出序列
    RNNDecoder 是 Seq2Seq 模型的 “生成核心”，其角色类似人类的 “写作过程”：
    先理解输入（通过编码器输出），
    再结合已写内容（历史生成 token），
    聚焦关键信息（注意力机制），
    逐步写出完整内容（自回归生成）
    Vocab 一个映射表 "the" → 0、"cat" → 1、"sat" → 2
    """
    def __init__(self, args, dictionary, embed_tokens):
        """
        参数:
            args: 模型配置参数
            dictionary: 词典对象
            embed_tokens: 词嵌入层
        """
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens

        # 验证编码器和解码器"层数匹配" 避免参数冲突
        assert args.decoder_layers == args.encoder_layers, f"""
            seq2seq rnn要求编码器和解码器具有相同层数的RNN。
            得到: {args.encoder_layers, args.decoder_layers}
        """
        # 当编码器使用双向 RNN 时，其输出维度为 2 * hidden_size
        assert args.decoder_ffn_embed_dim == args.encoder_ffn_embed_dim * 2, f"""
            seq2seq-rnn要求解码器隐藏层维度为编码器隐藏层维度的2倍。
            得到: {args.decoder_ffn_embed_dim, args.encoder_ffn_embed_dim * 2}
        """

        # 基础配置
        self.embed_dim = args.decoder_embed_dim
        self.hidden_dim = args.decoder_ffn_embed_dim
        self.num_layers = args.decoder_layers

        # 正则化层
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.dropout_out_module = nn.Dropout(args.dropout)

        # RNN层 - 使用单向 GRU，处理序列依赖（ decoder 核心“记忆”模块 ）
        # 必须使用单向 RNN，因为生成过程是顺序的，未来 token 尚未生成。
        self.rnn = nn.GRU(
            self.embed_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=args.dropout if self.num_layers > 1 else 0,
            batch_first=False,
            bidirectional=False
        )

        # 注意力层 - 解码器关注编码器的相关部分
        self.attention = AttentionLayer(
            self.embed_dim, self.hidden_dim, self.embed_dim, bias=False
        )

        # 输出维度投影(如果需要)
        if self.hidden_dim != self.embed_dim:
            self.project_out_dim = nn.Linear(self.hidden_dim, self.embed_dim)
        else:
            self.project_out_dim = None

        # 输出投影层 - 将隐藏状态映射到词表大小
        if args.share_decoder_input_output_embed: # 权重是否共享
            self.output_projection = nn.Linear(  # nn.Linear(embed_dim, vocab_size)
                self.embed_tokens.weight.shape[1],  # 输入维度：词嵌入向量的维度（如 512）
                self.embed_tokens.weight.shape[0],  # 输出维度：词表大小（如 30000）=vocab_size
                bias=False,
            )
            # 关键：将输出投影层的权重（vocab_size, embed_dim）与词嵌入层的权重（vocab_size, embed_dim）绑定
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.embed_dim,  # 输入维度：解码器输出的嵌入维度（如 512）
                len(dictionary),  # 输出维度：词表大小
                bias=False,
            )
            # 初始化权重：正态分布，均值0，标准差为嵌入维度的倒数平方根
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.embed_dim ** -0.5
            )

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **unused):
        """
        解码器前向传播过程
        增量状态的作用:避免重复计算，将推理阶段的时间复杂度从 O(T²) 降至 O(T)
        参数:
            prev_output_tokens: 先前生成的token序列 [batch_size, seq_len]
            encoder_out: 编码器输出元组(encoder_outputs, encoder_hiddens, encoder_padding_mask)
            incremental_state: 增量解码状态(用于推理时缓存历史信息)——自回归生成
            encoder_hiddens:历史信息压缩到固定维度的向量 h_t 中
        返回:
            x: 解码器输出 [batch_size, seq_len, vocab_size]
            x[b, t, v] 批次中第 b 个样本 在时间步 t（生成第 t 个词时） 词表中第 v 个 token 的得分（logits）
            extra: 额外信息(这里为None)
        """
        # 从编码器输出中提取信息
        """
         encoder_outputs: [encoder_seq_len, batch_size, num_directions*hidden]
         encoder_hiddens: [num_layers, batch_size, num_directions*encoder_hidden]
         encoder_padding_mask: [encoder_seq_len, batch_size]
        """
        encoder_outputs, encoder_hiddens, encoder_padding_mask = encoder_out

        # 处理增量解码状态(用于推理时) 复用历史 hidden state——
        if incremental_state is not None and len(incremental_state) > 0:
            # 如果有增量状态，只需处理最新的token
            prev_output_tokens = prev_output_tokens[:, -1:]
            # 从增量状态中获取之前的"隐藏状态"
            cache_state = self.get_incremental_state(incremental_state, "cached_state")
            prev_hiddens = cache_state["prev_hiddens"]
        else:
            # “训练”或首次解码时，使用编码器的最终隐藏状态初始化解码器
            prev_hiddens = encoder_hiddens

        bsz, seqlen = prev_output_tokens.size() # [batch_size, seq_len]

        # 词嵌入
        x = self.embed_tokens(prev_output_tokens) # [batch_size, seq_len, embed_dim]
        x = self.dropout_in_module(x)

        # 调整维度从 [batch_size, seq_len, embed_dim] 到 [seq_len, batch_size, embed_dim]
        x = x.transpose(0, 1)

        # 应用注意力机制
        if self.attention is not None:
            x, attn = self.attention(x, encoder_outputs, encoder_padding_mask)

        # 通过单向RNN处理序列
        x, final_hiddens = self.rnn(x, prev_hiddens)
        # x: [seq_len, batch_size, hidden_dim]
        # final_hiddens: [num_layers, batch_size, hidden_dim]
        x = self.dropout_out_module(x)

        # 投影到嵌入维度(如果需要)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        # 将隐藏状态"映射"到词表大小; hidden_dim 维度被映射为 vocab_size
        x = self.output_projection(x) #通过矩阵乘法将每个位置的隐藏向量从 hidden_dim 维映射到 vocab_size 维

        # 调整维度回 [batch_size, seq_len, vocab_size]
        x = x.transpose(1, 0)

        # 保存增量状态(用于下一次解码步骤)
        cache_state = {
            "prev_hiddens": final_hiddens,
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        return x, None

    def reorder_incremental_state(
            self,
            incremental_state,
            new_order,
    ):
        """
        重新排序增量状态(用于束搜索等需要改变批次顺序的操作)
        对多个候选路径的状态进行重排序
        参数:
            incremental_state: 增量状态
            new_order: 新的排序索引
        """
        # 从增量状态中获取缓存的隐藏状态
        cache_state = self.get_incremental_state(incremental_state, "cached_state")
        prev_hiddens = cache_state["prev_hiddens"]

        # 按照新顺序重新排列隐藏状态 沿着批次维度（第 0 维），根据 new_order 重新排列张量
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]

        # 更新增量状态
        cache_state = {
            "prev_hiddens": torch.stack(prev_hiddens),
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        return

# Seq2Seq 模型通过编码器将输入序列转换为固定维度的表示，然后解码器根据这个表示和之前生成的输出，逐步生成目标序列
class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        """
        参数:
            args: 包含模型配置的命名空间
            encoder: 编码器实例
            decoder: 解码器实例
        """
        super().__init__(encoder, decoder)
        self.args = args

    def forward(self, src_tokens, src_lengths, prev_output_tokens, return_all_hiddens: bool = True,):
        """
        参数:
            src_tokens: 源语言token的索引张量 [batch_size, src_len]
            src_lengths: 源序列长度 [batch_size]
            prev_output_tokens: 前一个输出的token索引 [batch_size, tgt_len] 标签
            return_all_hiddens: 是否返回所有隐藏层状态

        返回:
            tuple: (logits, extra)
                logits: 预测的logits [batch_size, tgt_len, vocab_size]
                extra: 包含额外信息的字典
        """
        # 将输入序列src_tokens和长度信息src_lengths传入编码器
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )

        # 解码器上一个生成的输出prev_output_tokens、编码器的输出encoder_out和源序列长度src_lengths传入
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        return logits, extra




