import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from utils.directory import DocumentManager

# 启用 MPS
torch.backends.mps.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = False
# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

        # 设置权重的数据类型为 float32
        self.conv1.weight = nn.Parameter(self.conv1.weight.to(torch.float32))
        self.conv2.weight = nn.Parameter(self.conv2.weight.to(torch.float32))
        self.fc1.weight = nn.Parameter(self.fc1.weight.to(torch.float32))
        self.fc2.weight = nn.Parameter(self.fc2.weight.to(torch.float32))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # 准备数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    train_dataset = datasets.CIFAR10(root=DocumentManager.default_train_data_dir(), train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleCNN()
    model = model.to("mps")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 使用混合精度训练
    scaler = GradScaler()

    # 训练模型
    epochs = 5
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # 使用 autocast 进行混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

if __name__ == '__main__':
    train()
