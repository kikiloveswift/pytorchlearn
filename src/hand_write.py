import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import logging

_logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个转换，将图像转换为 PyTorch 张量
transform = transforms.Compose([transforms.ToTensor()])

# data
train_data = dataset.MNIST(root="data/MNIST", train=True, transform=transform, download=True)
test_data = dataset.MNIST(root="data/MNIST", train=False, transform=transform, download=False)

# batchsize 当数据量特别大的时候，没法一次性把所有数据都放到内存中，这里的 batchsize 概念就是分批次丢入
batch_size = 64
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# net
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 因为原始图是 28 * 28 的灰度图，MaxPool2d 那里写的是2，因此最终结果是 14 * 14 * out_channel（32）
        # 10 是因为想要输出当前的数字在 0 ~ 9 上的概率分布
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    # PyTorch 的顺序是 NCHW
    """
    N (Batch Size): 批量大小，代表一次处理的图像数量。在训练和推理时，通常会将多个图像打包成一个批次，以便并行处理，提高效率。

    C (Channels): 通道数，表示图像的颜色通道。对于标准的RGB图像，C 为 3（分别代表红、绿、蓝）。对于灰度图像，C 为 1。

    H (Height): 图像的高度，单位通常是像素。

    W (Width): 图像的宽度，也是以像素为单位。
    """
    def forward(self, x):
        out = self.conv(x)
        # 这行没懂
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

"""
在您的卷积神经网络（CNN）中，`out = out.view(out.size()[0], -1)` 这一行代码执行的是一个重塑（reshaping）操作，这是在准备将多维的卷积层输出转换为全连接层（`self.fc`）的输入时非常重要的步骤。

让我们分解这行代码：

1. `out.size()[0]`：这部分获取的是批量大小（`N`）的值。在您的网络中，`out` 在通过卷积层和池化层后，其形状是 `[N, C, H, W]`。其中 `N` 是批量大小，`C` 是输出通道数，`H` 和 `W` 分别是卷积操作后的高度和宽度。`out.size()[0]` 就是提取这个批量大小 `N`。

2. `-1`：在 PyTorch 中，将 `-1` 作为 `view` 方法的参数表示自动计算该维度的大小，以保证总元素数量与原始张量相同。在这个情况下，它会自动计算卷积层输出的所有元素（除了批量大小维度）并将它们展平成一个长向量。

综上，`out.view(out.size()[0], -1)` 的作用是将每个样本在卷积层的输出从 `[C, H, W]` 的形状变换成一个一维向量。这个一维向量然后被用作全连接层（`self.fc`）的输入。这是因为全连接层需要一维的输入向量，而不是多维的卷积输出。

在你的例子中，由于输入图像是 28x28 的灰度图，经过带有 2x2 最大池化层的卷积操作后，每个特征图的尺寸变为 14x14。由于有 32 个输出通道，因此每个样本在卷积层的输出尺寸为 [32, 14, 14]。`out.view(out.size()[0], -1)` 会将这个输出转换成一个 [N, 32*14*14] 的矩阵，即每个样本都变成了一个 6272（32*14*14）元素的一维向量。这个向量接着被送入全连接层进行进一步的处理。
"""

cnn = CNN().to(device)
# loss
# 损失函数选择，分类问题一般使用交叉熵
loss_func = torch.nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# training
for epoch in range(100):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _logger.info(f"epoch is {epoch + 1}, ite = {i}/{len(train_data) // batch_size} loss = {loss.item()} ")


# eval/test

# save

# load

# inference 推理