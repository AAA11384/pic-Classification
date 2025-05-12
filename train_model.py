import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# === 配置选项 ===
use_label_smoothing = True  # True 使用 Label Smoothing，False 使用 CrossEntropy
use_adagrad = False         # True 使用 Adagrad，False 使用 Adam
epoch = 50                  # 训练轮数

# 数据集路径
dataset_dir = "dataset/Caltech256"

# 数据增强
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(224, padding=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_data = datasets.ImageFolder(root=dataset_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(root=dataset_dir + '/test', transform=test_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
train_num = len(train_data)
test_num = len(test_data)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
print(torch.version.cuda)

# 模型加载
model = torch.load("123.pth", weights_only=False)
if hasattr(model, 'classifier'):
    dropout_layer = nn.Dropout(p=0.5)
    new_classifier = nn.Sequential()
    for i, layer in enumerate(model.classifier):
        new_classifier.add_module(str(i), layer)
        if i == 1:
            new_classifier.add_module(str(i + 1), dropout_layer)
    model.classifier = new_classifier
model.to(device)

# 损失函数和优化器
if use_label_smoothing:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
else:
    criterion = nn.CrossEntropyLoss()

if use_adagrad:
    optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.0001)
else:
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
criterion.to(device)

# TensorBoard
writer = SummaryWriter('runs/image_classification')

best_test_acc = 0
early_stopping_patience = 3
early_stopping_counter = 0

train_accuracies = []
test_accuracies = []

start_time = time.time()
temp_time = start_time

for i in range(epoch):
    print(f"--------------第{i}轮训练开始：---------------")
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        preds = torch.max(outputs, 1)[1]
        train_acc += (preds == targets).sum().item()

    train_loss /= train_num
    train_acc /= train_num

    model.eval()
    test_loss, test_acc = 0.0, 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * images.size(0)
            preds = torch.max(outputs, 1)[1]
            test_acc += (preds == targets).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss /= test_num
    test_acc /= test_num
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    writer.add_scalar('Training Loss', train_loss, i)
    writer.add_scalar('Training Accuracy', train_acc, i)
    writer.add_scalar('Testing Loss', test_loss, i)
    writer.add_scalar('Testing Accuracy', test_acc, i)
    writer.add_scalar('Testing Precision', precision, i)
    writer.add_scalar('Testing Recall', recall, i)
    writer.add_scalar('Testing F1 Score', f1, i)

    print(f"第{i}轮训练损失：{train_loss:.4f}，准确率：{train_acc:.4f}")
    print(f"第{i}轮测试损失：{test_loss:.4f}，准确率：{test_acc:.4f}")
    print(f"精确率：{precision:.4f}，召回率：{recall:.4f}，F1：{f1:.4f}")
    print(f"训练耗时：{time.time() - temp_time:.2f}s")
    temp_time = time.time()

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    scheduler.step()

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model, "123_best.pth")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {i}")
            break

    print(f"--------------第{i}轮训练结束---------------")

print(f"总耗时：{time.time() - start_time:.2f}s")
writer.close()

# 准确率绘图
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
plt.plot(range(len(test_accuracies)), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train/Test Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()
