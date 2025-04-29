import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn
import torch.optim as optim
import time

# 数据集路径
dataset_dir = "dataset/Caltech256"

# 预处理，增加数据增强操作
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小至224x224像素
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图片
    transforms.RandomRotation(10),  # 随机旋转图片，角度范围为 -10 到 10 度
    transforms.RandomCrop(224, padding=10),  # 随机裁剪图片
    transforms.ToTensor(),  # 将图片转换为 PyTorch 中的张量（tensor）类型
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化处理
])

# 加载数据集
train_data = datasets.ImageFolder(root=dataset_dir + '/256_ObjectCategories', transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

train_num = len(train_data)
print("训练数据集数量：{}".format(train_num))

# 加载训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print('使用GPU训练')
else:
    print('使用CPU训练')

# 加载模型
model = torch.load("123.pth", weights_only=False)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)  # 使用 Adagrad 优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # 学习率调度器
criterion.to(device)

# 设置训练网络的参数
epoch = 10  # 训练轮数

start_time = time.time()  # 开始时间
temp_time = start_time

# 开始训练
for i in range(epoch):
    print("--------------第{}轮训练开始：---------------".format(i))
    # 将模型设置为训练模式
    model.train()
    # 初始化训练损失和正确率
    train_loss = 0.0
    train_acc = 0.0
    for data in train_loader:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播，计算损失
        outputs = model(images)
        loss = criterion(outputs, targets)
        # 优化模型，反向传播，更新模型参数
        loss.backward()
        optimizer.step()
        # 统计训练损失和正确率
        train_loss += loss.item() * images.size(0)
        preds = torch.max(outputs, 1)
        train_acc += (preds[1] == targets).sum().item()

    # 调整学习率
    scheduler.step()

    # 计算平均训练损失和正确率
    train_loss = train_loss / train_num
    train_acc = train_acc / train_num
    print("第{}轮训练平均损失值为{}".format(i, train_loss))
    print("第{}轮训练正确率为{}".format(i, train_acc))

    end_time = time.time()
    print("第{}轮训练用时{}秒".format(i, end_time - temp_time))
    temp_time = end_time

    # 保存模型
    torch.save(model, "123.pth")
    print("本轮模型已保存")
    print("--------------第{}轮训练结束：---------------".format(i))

end_time = time.time()  # 结束时间
print("共用时{}秒".format(end_time - start_time))