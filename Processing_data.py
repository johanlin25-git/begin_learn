import re
import random
import os
import sentencepiece as spm
import subprocess
from pathlib import Path

#文件内容打印
data_dir = './DATA'
dataset_name = 'ted2020'
src_lang = 'en'
tgt_lang = 'zh'

prefix = Path(data_dir).absolute() / dataset_name

prefix.mkdir(parents=True, exist_ok=True)

data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'

# 定义函数，用于读取并打印文件前几行内容（这里示例打印前5行）
def print_file_content(file_path, num_lines=5):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: # with 确保在使用资源后，自动执行释放资源
            content = [next(f).strip() for _ in range(num_lines)] #逐行读取文件内容，f 是文件对象
            print('\n'.join(content)) #列表中的字符串（一行多列）元素连接成一列多行的字符串
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")

#region 文本预处理和语料库清洗
#全角转半角函数（一个字母所占的字符只有一个）
def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s: # 字符编码转换
            inside_code = ord(uchar)  # 单个字符的 Unicode 码点  ord('A') → 65
            if inside_code == 12288:  # 检测全角空格
                inside_code = 32      # 转半角空格
            elif (inside_code >= 65281 and inside_code <= 65374):  # 其他全角字符（除空格）根据关系转化
                inside_code -= 65248  # 全角字符与半角字符之间恰好相差 65248
            rstring += chr(inside_code) # chr(65) → 'A'
        ss.append(rstring)
    return ''.join(ss) # 连接成一个字符串

#文本清洗（标准）
def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s)  # "Hello (world)" → "Hello "
        s = s.replace('-', '')  # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s)  # 前后添加空格 "Hello!" → "Hello ! "
    elif lang == 'zh':
        s = strQ2B(s)  # 全角转半角
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = ' '.join(s.strip().split())  # 移除多余空格
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s)
    s = ' '.join(s.strip().split()) #strip() 移除首尾空格，split() 所有空白字符分割成列表，最后用单个空格重新连接
    return s

def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split()) #按空格分词后计算词数

#语料库清洗与筛选
def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return # 如果清洗存在，跳出函数

    with open(f'{prefix}.{l1}', 'r', encoding='utf-8') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r', encoding='utf-8') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w', encoding='utf-8') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w', encoding='utf-8') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0:  # 移除短句子
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0:  # 移除长句子
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0:  # 按长度比例移除
                            if s1_len / s2_len > ratio or s2_len / s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)

#执行清洗数据集
clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)

print("file cleaned data.")
print_file_content(f"{data_prefix}.clean.{src_lang}")
print_file_content(f"{data_prefix}.clean.{tgt_lang}")
# endregion

#region 生成一定比例的训练集与验证集
valid_ratio= 0.01 # 3000~4000 would suffice
train_ratio = 1 - valid_ratio

if (prefix/f'train.clean.{src_lang}').exists() \
and (prefix/f'train.clean.{tgt_lang}').exists() \
and (prefix/f'valid.clean.{src_lang}').exists() \
and (prefix/f'valid.clean.{tgt_lang}').exists():
    print(f'train/valid splits exists. skipping split.')
else:
    with open(f'{data_prefix}.clean.{src_lang}', 'r', encoding='utf-8') as f:
        line_num = sum(1 for line in f)
    labels = list(range(line_num)) #生成列表
    random.shuffle(labels)
    for lang in [src_lang, tgt_lang]: # 对列表里的每一种语言各执行一回 先en再zh
        train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w', encoding='utf-8')
        valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w', encoding='utf-8')
        count = 0 #索引 0到line_num-1
        with open(f'{data_prefix}.clean.{lang}', 'r', encoding='utf-8') as f:
            for line in f:
                if labels[count]/line_num < train_ratio:  # 约 99% 的数据（索引值较小的部分）
                    train_f.write(line)
                else:
                    valid_f.write(line)
                count += 1
        train_f.close()
        valid_f.close()
#endregion

#region 分词处理
#单词的单位，把文本转换为子词序列 提高模型对未知词汇的处理能力
vocab_size = 8000
if (prefix/f'spm{vocab_size}.model').exists():
    print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
else:
    spm.SentencePieceTrainer.train(
        input=','.join([f'{prefix}/train.clean.{src_lang}',
                        f'{prefix}/valid.clean.{src_lang}',
                        f'{prefix}/train.clean.{tgt_lang}',
                        f'{prefix}/valid.clean.{tgt_lang}']),
        model_prefix=prefix/f'spm{vocab_size}', #设置模型文件的前缀
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 'bpe' works as well
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',) # 训练一个分词器模型 生成两个主要文件
spm_model = spm.SentencePieceProcessor()# 加载模型
in_tag = { # 映射关系
    'train': 'train.clean',
    'valid': 'valid.clean',
    'test': 'test.raw.clean',
}
for split in ['train', 'valid', 'test']:
    for lang in [src_lang, tgt_lang]:
        out_path = prefix/f'{split}.{lang}'
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(prefix / f'{split}.{lang}', 'w', encoding='utf-8') as out_f:
                with open(prefix / f'{in_tag[split]}.{lang}', 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str) # 将文本编码转为子词列表，out_type=str表示输出字符串类型的子词
                        print(' '.join(tok), file=out_f) # 接成字符串

print("file spm_encoded.")
print_file_content(f"{data_dir}/{dataset_name}/{split}.{src_lang}")
print_file_content(f"{data_dir}/{dataset_name}/{split}.{tgt_lang}")
#endregion

#region 数据集二进制 生成data-bin
binpath = Path('./DATA/data-bin', dataset_name)
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    # 构造命令列表，避免字符串拼接的转义问题
    cmd = [
        'python', '-m', 'fairseq_cli.preprocess',
        '--source-lang', src_lang,
        '--target-lang', tgt_lang,
        '--trainpref', f'{prefix}/train',
        '--validpref', f'{prefix}/valid',
        '--testpref', f'{prefix}/test',
        '--destdir', str(binpath),
        '--joined-dictionary',
        '--workers', '2'
    ]
    # 执行命令，捕获输出（可选）
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print(f"预处理失败：{result.stderr}")
    else:
        print(f"预处理完成：{result.stdout}")
#endregion



