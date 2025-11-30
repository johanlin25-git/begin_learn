import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset,Dataset
from torchvision.datasets import DatasetFolder
import os
import time

# 定义自定义数据集类---半监督数据集功能
class PseudoLabelDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):   #通过索引获取元素
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# 定义CNN模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()  #调用父类模型初始化
        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(    #卷积层
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(    #全连接层
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        x = self.cnn_layers(x)  # 特征提取
        # 此时 x 形状: [batch_size, 256, 8, 8] （256通道的8x8特征图）
        x = x.flatten(1)  # 空间展平
        # 现在 x 形状: [batch_size, 256*8*8] = [batch_size, 16384]
        x = self.fc_layers(x)  # 分类决策
        # 最终 x 形状: [batch_size, 11] （11个类别的预测分数）
        return x

# 伪标签生成函数
def get_pseudo_labels(dataset, model, threshold=0.65):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    softmax = nn.Softmax(dim=-1)#输入数据转换为概率分布，转换后所有元素的数值都处于 0 到 1 之间，并且它们的总和为 1。

    # 存储伪标签样本的路径和标签
    pseudo_image_paths = []
    pseudo_labels = []
    #变量
    batch_size = 128
    current_index = 0
    total_samples = 0
    accepted_samples = 0
    confidence_sum = 0.0

    # 从DatasetFolder中获取所有图像路径
    all_image_paths = [path for path, _ in dataset.samples]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True,drop_last=False)
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="生成伪标签"): #img==[batch_size张图片, channels, height, width]。
            batch_size_actual = imgs.size(0)              #实际批次大小==batch_size
            total_samples += batch_size_actual

            logits = model(imgs.to(device))
            probs = softmax(logits)

            # 获取每个样本的最大概率和对应类别
            max_probs, preds = torch.max(probs, dim=1) #从列中取最高类

            batch_paths = all_image_paths[current_index:current_index + batch_size_actual] #切片获取所有索引
            current_index += batch_size_actual

            # 筛选高置信度样本
            for i in range(batch_size_actual):  #索引的i位置与batch_size密切相关
                confidence = max_probs[i].item()  #取最大值的数值 将这个单元素张量转换为 Python 的浮点数（float）便于后续使用（例如打印、保存到文件等
                # 最大值大于阈值才接受伪标签
                if confidence >= threshold:
                    pseudo_image_paths.append(batch_paths[i])  #最大值添加到列表中
                    pseudo_labels.append(preds[i].item())
                    accepted_samples += 1
                    confidence_sum += confidence

    # 计算平均置信度
    avg_confidence = confidence_sum / accepted_samples if accepted_samples > 0 else 0

    # 打印统计信息
    acceptance_rate = accepted_samples / total_samples * 100 if total_samples > 0 else 0
    print(f"伪标签生成完成: {accepted_samples}/{total_samples} 样本被接受 ({acceptance_rate:.2f}%)")
    print(f"平均置信度: {avg_confidence:.4f}, 阈值: {threshold:.2f}")

    # 构建带伪标签的新数据集
    pseudo_dataset = PseudoLabelDataset(
        pseudo_image_paths,
        pseudo_labels,
        transform=train_tfm
    )

    model.train()
    return pseudo_dataset

# 数据增强
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试和验证使用的变换
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 图像加载器
def image_loader(path):
    return Image.open(path).convert('RGB')

# 训练函数
def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{total_epochs}"):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)

        # 前向传播
        logits = model(imgs)
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        # 计算准确率
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc.item())

    # 计算平均训练损失和准确率
    avg_train_loss = sum(train_loss) / len(train_loss)
    avg_train_acc = sum(train_accs) / len(train_accs)

    return avg_train_loss, avg_train_acc

# 验证函数
def validate(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader, desc="验证"):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(imgs)
            loss = criterion(logits, labels)

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc.item())

    # 计算平均验证损失和准确率
    avg_valid_loss = sum(valid_loss) / len(valid_loss)
    avg_valid_acc = sum(valid_accs) / len(valid_accs)

    return avg_valid_loss, avg_valid_acc

# 测试函数
def test(model, test_loader, device):
    model.eval()
    predictions = []

    for batch in tqdm(test_loader, desc="测试"):
        imgs, _ = batch
        imgs = imgs.to(device)

        with torch.no_grad():
            logits = model(imgs)

        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    return predictions

def main():
    # 设置超参数
    batch_size = 128
    n_epochs = 10
    start_semi_epoch = 2  # 从第5个epoch开始半监督学习
    base_threshold = 0.8  # 基础伪标签阈值
    min_threshold = 0.55  # 最小伪标签阈值
    threshold_decay = 0.02  # 每个epoch降低0.02
    do_semi = False
    # 添加是否恢复训练的选项
    resume_training = True  # 设置为 True 以恢复训练

    # 确保输出目录存在
    os.makedirs("results", exist_ok=True)

    # 构造数据集
    print("加载数据集...")
    train_set = DatasetFolder("../data/food-11/training/labeled",
                              loader=image_loader, extensions=("jpg",), transform=train_tfm)
    valid_set = DatasetFolder("../data/food-11/validation",
                              loader=image_loader, extensions=("jpg",), transform=test_tfm)
    # 创建未标记数据集（用于伪标签生成）
    unlabeled_set_pseudo = DatasetFolder("../data/food-11/training/unlabeled",
                                         loader=image_loader, extensions=("jpg",), transform=test_tfm)
    test_set = DatasetFolder("../data/food-11/testing",
                             loader=image_loader, extensions=("jpg",), transform=test_tfm)

    # 构造初始数据加载器
    print("创建数据加载器...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 初始化模型
    print("初始化模型...")
    model = Classifier().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # **检查是否需要恢复训练**
    if resume_training:
        try:
            print("加载最佳模型...")
            best_checkpoint = torch.load("results/best_model.pth")
            model.load_state_dict(best_checkpoint['model_state_dict'])
            optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
            start_epoch = best_checkpoint['epoch']
            best_valid_acc = best_checkpoint['valid_acc']
            history = best_checkpoint['history']
            print(f"从 epoch {start_epoch} 继续训练，最佳验证准确率: {best_valid_acc:.4f}")
        except Exception as e:
            print(f"恢复训练失败: {e}")
            print("将从头开始训练")
            start_epoch = 0
            best_valid_acc = 0.0
            history = {
                'train_loss': [],
                'train_acc': [],
                'valid_loss': [],
                'valid_acc': [],
                'pseudo_samples': [],
                'threshold': []
            }
    else:
        start_epoch = 0
        best_valid_acc = 0.0
        history = {
            'train_loss': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_acc': [],
            'pseudo_samples': [],
            'threshold': []
        }

    print("\n开始训练...")
    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        # 计算当前阈值（动态调整）
        current_threshold = max(base_threshold - epoch * threshold_decay, min_threshold)
        if do_semi:
            print(f"\n[Epoch {epoch + 1}/{n_epochs}] 开始半监督学习 (阈值={current_threshold:.2f})...")

            # 生成伪标签
            pseudo_set = get_pseudo_labels(unlabeled_set_pseudo, model, threshold=current_threshold)
            history['pseudo_samples'].append(len(pseudo_set))
            history['threshold'].append(current_threshold)

            # 合并原始标签数据和伪标签数据
            concat_dataset = ConcatDataset([train_set, pseudo_set])

            # 新的训练数据加载器
            train_loader = DataLoader(concat_dataset,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)
        else:
            history['pseudo_samples'].append(0)
            history['threshold'].append(0.0)

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, n_epochs
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 学习率调度
        scheduler.step()

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        # 基于验证集性能启动
        if valid_acc > min_threshold and epoch >= start_semi_epoch:
            print("启动半监督学习")
            do_semi = True

        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        # 保存最佳模型
        if valid_acc > best_valid_acc and epoch>=start_semi_epoch:
            best_valid_acc = valid_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_acc': valid_acc,
                'history': history
            }, "results/best_model.pth")
            print(f"保存最佳模型，验证准确率: {best_valid_acc:.4f}")

        # 打印结果
        epoch_time = time.time() - epoch_start_time
        print(f"\n[Epoch {epoch + 1:03d}/{n_epochs:03d}] "
              f"训练损失: {train_loss:.5f}, 训练准确率: {train_acc:.4f} | "
              f"验证损失: {valid_loss:.5f}, 验证准确率: {valid_acc:.4f} | "
              f"学习率: {optimizer.param_groups[0]['lr']:.6f} | "
              f"时间: {epoch_time:.1f}秒")

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"results/checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_acc': valid_acc,
                'history': history
            }, checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")
    # 训练完成
    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {total_time / 60:.1f}分钟")

    # ---------- 测试阶段 ----------
    print("\n开始测试...")
    # 加载最佳模型
    best_checkpoint = torch.load("results/best_model.pth")
    model.load_state_dict(best_checkpoint['model_state_dict'])

    predictions = test(model, test_loader, device)

    # 保存预测结果
    result_path = "results/predictions.csv"
    with open(result_path, "w") as f:
        f.write("Id,Category\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")

    print(f"测试完成! 结果已保存到 {result_path}")

    # 保存训练历史
    history_path = "results/training_history.npy"
    np.save(history_path, history)
    print(f"训练历史已保存到 {history_path}")

if __name__ == "__main__":
    main() # Windows系统上必须的保护措施



