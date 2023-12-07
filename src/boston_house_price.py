import torch

# 选择 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data 解析数据
import numpy as np
# 打开文件并读取数据
with open('data/boston/housing.data', 'r') as file:
    data_lines = file.readlines()

# 将每一行的数据转换为浮点数
data_values = [list(map(float, line.split())) for line in data_lines]

# 将数据转换为 NumPy 数组
data_array = np.array(data_values).astype(np.float32)
print(data_array.shape)

# Y 为提取最后一列的数据，因为最后一列是价格
Y = data_array[:, -1]

# X 是提取从第 0 ~ 倒数第二列的所有数据，也即是，除去最后 1 列的所有数据
X = data_array[:, 0:-1]

# 划分训练集
Y_train = Y[0:496, ...]
X_train = X[0:496, ...]

Y_test = Y[496:, ...]
X_test = X[496:, ...]

# net 搭建网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_out):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_out)

    def forward(self, x):
        out = self.hidden(x)
        # 加入 relu 非线性运算，使我们的模型表达能力更强
        out = torch.relu(out)
        out = self.predict(out)
        return out

net = Net(13, 1).to(device)

# loss 定义 loss
loss_func = torch.nn.MSELoss()

# optimize 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# training 训练
for i in range(10000):
    x_data = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_data = torch.tensor(Y_train, dtype=torch.float32).to(device)

    pred = net(x_data)
    # squeeze 去掉张量为 1 的维度
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001

    # 调用优化器，梯度置为 0，然后反向传播
    optimizer.zero_grad()
    loss.backward()
    # 更新网络中的参数
    optimizer.step()
    print(f"item:{i}, loss:{loss}")

    # 打印计算出来的前 10 个值
    print(pred[0:10])

    # 打印实际的前 10 个值
    print(y_data[0:10])

    x_data = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_data = torch.tensor(Y_test, dtype=torch.float32).to(device)
    pred = net(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 0.001
    print(f"test item:{i}, loss_test: {loss_test}")

torch.save(net, "model/model.pkl")
torch.save(net.state_dict(), "model/params.pkl")