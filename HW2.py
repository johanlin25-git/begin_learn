import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

#preparing data
print('Loading data ...')
data_root='../data/timit_11/'
train_data = np.load(data_root+'train_11.npy')
label_data=np.load(data_root+'train_label_11.npy')
test_data=np.load(data_root+'test_11.npy')
print("# train.shape = (样本数, 特征数) size of training data{}".format(train_data.shape))

#自定义数据集结构——封装数据为PyTorch可处理的格式
class TIMITDataset(Dataset):
    def __init__(self,X,y=None):   # 实际有接收三个参数：self + X + y
        self.data=torch.from_numpy(X).float()
        if y is not None:         #数据存在标签？
            y = y.astype(int)  # 标签转为整数
            self.label = torch.LongTensor(y)  # 标签转为长整型张量
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None: #返回
            return self.data[idx],self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

#划分数据集的作用域
VAL_RATIO=0.2
percent=int(train_data.shape[0]*(1-VAL_RATIO))
train_x,train_y=train_data[:percent],label_data[:percent]
val_x,val_y=train_data[percent:],label_data[percent:]    #从percent开始到最后的所有行和列

#创建数据加载器
BATCH_SIZE=64
train_set=TIMITDataset(train_x,train_y) #self是自动生成的实例
val_set=TIMITDataset(val_x,val_y)
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=False)

#内存清理
del train_data,label_data,train_x,train_y,val_x,val_y
gc.collect()#垃圾回收

#定义模型
#神经网络架构   输入层(429) → 隐藏层1(1024) → 隐藏层2(512) → 隐藏层3(128) → 输出层(39)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(429,1024)
        self.layer2=nn.Linear(1024,512)
        self.layer3=nn.Linear(512,128)
        self.out=nn.Linear(128,39)
        self.act_fn = nn.Sigmoid()
    def forward(self,x):#向前传播
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
        return x
#线性回归
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)
#
class ImprovedSequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 39)  # 添加输出层，映射到39个类别
        )

    def forward(self, x):
        return self.net(x)

#定义损失函数
criterion = nn.CrossEntropyLoss()  # 分类任务常用损失函数

def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

def same_seeds(seed):
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True
# fix random seed for reproducibility
same_seeds(0)
# get device
device = get_device()
print(f'DEVICE: {device}')

#训练过程
def train_epoch_ch(model,train_loader,criterion,optimizer,num_epoch):
    model.train()  # 设置为训练模式（影响Dropout、BatchNorm等层）
    for epoch in range(num_epoch):
        # 初始化指标
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        for i, data in enumerate(train_loader):
            # 数据准备
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            optimizer.zero_grad()  # 清除梯度
            outputs = model(inputs)  # 模型预测

            # 损失计算
            batch_loss = criterion(outputs, labels)

            # 预测结果
            _, train_pred = torch.max(outputs, 1)  # 获取预测dim=1的类别——返回两个值data与index

            # 反向传播
            batch_loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 累计指标
            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # 验证阶段
        if len(val_set) > 0:  # 如果有验证集
            model.eval()  # 设置为评估模式
            with torch.no_grad():  # 禁用梯度计算
                for i, data in enumerate(val_loader):
                    # 类似训练阶段的处理，但不更新参数
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                    _, val_pred = torch.max(outputs, 1)

                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                    val_loss += batch_loss.item()

            # 打印每轮结果
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch,
                train_acc / len(train_set),  # 训练准确率
                train_loss / len(train_loader),  # 平均训练损失
                val_acc / len(val_set),  # 验证准确率
                val_loss / len(val_loader)  # 平均验证损失
            ))
            best_acc = 0.0  # 跟踪最佳验证准确率
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)  # 保存模型权重
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))

        else:  # 如果没有验证集
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch,
                train_acc / len(train_set),
                train_loss / len(train_loader))
            )

            # 如果没有验证集，保存最后一个epoch的模型
            if len(val_set) == 0:
                torch.save(model.state_dict(), model_path)
            print('saving model at last epoch')

# training parameters
num_epoch = 10                # number of training epoch
learning_rate = 0.0005       # learning rate

# the path where checkpoint saved
model_path = 'model.ckpt'
model = ImprovedSequential().to(device)  # 创建模型并移至设备
#定义优化器——梯度下降算法
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器
train_epoch_ch(model,train_loader,criterion,optimizer,num_epoch)

# # create testing dataset
# test_set = TIMITDataset(test_data, None)
# test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
#
# # create model and load weights from checkpoint
# model = Classifier().to(device)
# model.load_state_dict(torch.load(model_path))
#
# predict = []
# model.eval() # set the model to evaluation mode
# with torch.no_grad():
#     for i, data in enumerate(test_loader):
#         inputs = data
#         inputs = inputs.to(device)
#         outputs = model(inputs)
#         _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
#
#         for y in test_pred.cpu().numpy():
#             predict.append(y)
#
# with open('prediction.csv', 'w') as f:
#     f.write('Id,Class\n')
#     for i, y in enumerate(predict):
#         f.write('{},{}\n'.format(i, y))