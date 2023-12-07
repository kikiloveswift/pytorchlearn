import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

dir = 'FashionMNIST'

# 下载训练数据
train_data = datasets.FashionMNIST(
    root=dir,
    train=True,
    download=True,
    transform=ToTensor()
)

# 下载测试数据
test_data = datasets.FashionMNIST(
    root=dir,
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# 加载数据
# 加载数据有两个关键的类， torch.utils.data.DataLoader torch.utils.data.Dataset.Dataset
# Dataset 数据集，存储样本和相关的标签
# DataLoader 围绕这个数据集包装了一个可以迭代的对象

# 创建 loader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

for x, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# 定义网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # print(f"forward: x = {x}")
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# 定义损失函数和优化器，损失函数选用交叉熵，优化器使用 SGD
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # 计算预测误差
        pred = model(x)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f} [{current:>5d} / {size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(correct * 100):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1} \n ----------------------")
    train(dataloader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    test(dataloader=test_loader, model=model, loss_fn=loss_fn)
print("Done")

torch.save(model.state_dict(), f"{dir}/model.pth")
print("Saved Pytorch model state to model")