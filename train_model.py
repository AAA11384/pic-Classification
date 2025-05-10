import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# 数据集路径
dataset_dir = "dataset/Caltech256"

# 预处理，增加更多数据增强操作
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小至224x224像素
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图片
    transforms.RandomRotation(15),  # 随机旋转图片，角度范围为 -15 到 15 度
    transforms.RandomCrop(224, padding=10),  # 随机裁剪图片
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 颜色抖动
    transforms.ToTensor(),  # 将图片转换为 PyTorch 中的张量（tensor）类型
    # 对图片进行标准化处理
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载数据集
train_data = datasets.ImageFolder(root=dataset_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(root=dataset_dir + '/test', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

train_num = len(train_data)
test_num = len(test_data)
print("训练数据集数量：{}".format(train_num))
print("测试数据集数量：{}".format(test_num))

# 加载训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print('使用GPU训练')
else:
    print('使用CPU训练')

print(torch.version.cuda)

# 加载模型
model = torch.load("123.pth", weights_only=False)
# 假设模型有 classifier 层，添加 Dropout 层
if hasattr(model, 'classifier'):
    dropout_layer = nn.Dropout(p=0.5)
    new_classifier = nn.Sequential()
    for i, layer in enumerate(model.classifier):
        new_classifier.add_module(str(i), layer)
        if i == 1:  # 在合适的位置添加 Dropout 层
            new_classifier.add_module(str(i + 1), dropout_layer)
    model.classifier = new_classifier
model.to(device)

# 定义损失函数和优化器，添加 L2 正则化
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
criterion.to(device)

# 设置训练网络的参数
epoch = 50  # 训练轮数

# 创建 SummaryWriter 对象，用于写入 TensorBoard 数据
writer = SummaryWriter('runs/image_classification')

start_time = time.time()  # 开始时间
temp_time = start_time

# 提前停止相关参数
best_test_acc = 0
early_stopping_patience = 3
early_stopping_counter = 0

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
        outputs = model(images)  # 将数据放到网络中训练
        loss = criterion(outputs, targets)  # 用损失函数得到差异值
        # 优化模型，反向传播，更新模型参数
        loss.backward()
        optimizer.step()
        # 统计训练损失和正确率
        train_loss += loss.item() * images.size(0)
        preds = torch.max(outputs, 1)[1]
        train_acc += (preds == targets).sum().item()

    # 计算平均训练损失和正确率
    train_loss = train_loss / train_num
    train_acc = train_acc / train_num

    # 将模型设置为评估模式
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in test_loader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * images.size(0)
            preds = torch.max(outputs, 1)[1]
            test_acc += (preds == targets).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 计算平均测试损失和正确率
    test_loss = test_loss / test_num
    test_acc = test_acc / test_num

    # 计算精确率、召回率和 F1 score
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    # 将训练损失和正确率写入 TensorBoard
    writer.add_scalar('Training Loss', train_loss, i)
    writer.add_scalar('Training Accuracy', train_acc, i)
    writer.add_scalar('Testing Loss', test_loss, i)
    writer.add_scalar('Testing Accuracy', test_acc, i)
    writer.add_scalar('Testing Precision', precision, i)
    writer.add_scalar('Testing Recall', recall, i)
    writer.add_scalar('Testing F1 Score', f1, i)

    print("第{}轮训练平均损失值为{}".format(i, train_loss))
    print("第{}轮训练正确率为{}".format(i, train_acc))
    print("第{}轮测试平均损失值为{}".format(i, test_loss))
    print("第{}轮测试正确率为{}".format(i, test_acc))
    print("第{}轮测试精确率为{}".format(i, precision))
    print("第{}轮测试召回率为{}".format(i, recall))
    print("第{}轮测试 F1 score 为{}".format(i, f1))

    end_time = time.time()
    print("第{}轮训练用时{}秒".format(i, end_time - temp_time))
    temp_time = end_time

    # 调整学习率
    scheduler.step()

    # 提前停止策略
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model, "123_best.pth")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping at epoch {}".format(i))
            break

    print("--------------第{}轮训练结束：---------------".format(i))

end_time = time.time()  # 结束时间
print("共用时{}秒".format(end_time - start_time))

# 关闭 SummaryWriter
writer.close()