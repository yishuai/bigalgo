# 安装 PyTorch
# https://pytorch.org/get-started/locally/

# GPT4 Prompt:
# 请用 PyTorch 的梯度下降方法，写出一个最简单的 Batch gradient descent 的 SVM 代码示例，请包括训练集和测试集的划分。在训练集训练后，打印出获得的模型参数 W 和 b 的值。最后在测试集上进行测试，并画图展示结果

# 下面是一个使用 PyTorch 实现的线性支持向量机（SVM）模型的最简单的 Batch Gradient Descent 代码示例。代码包括训练集和测试集的划分，训练后的模型参数 `W` 和 `b` 的打印，以及在测试集上的结果预测和可视化展示。

# PyTorch 实现 SVM 的代码：

### 代码解释：
# 1. **数据生成与分割**：
#    - 使用`train_test_split`将数据集分为训练集和测试集，测试集占20%。
#    - 数据转换为 PyTorch 张量以便进行计算。

# 2. **模型定义**：
#    - SVM 模型是通过 `nn.Linear` 创建的简单线性模型。
#    - 损失函数使用 Hinge Loss（SVM 常用的损失函数），并加入L2正则化以避免过拟合。

# 3. **训练过程**：
#    - 使用批量梯度下降 (`SGD`) 进行优化。
#    - 每 100 次迭代会打印一次损失值。

# 4. **模型参数**：
#    - 打印训练后模型的权重 `W` 和偏置 `b`。

# 5. **结果可视化**：
#    - 使用`matplotlib`绘制测试集的数据点和模型的决策边界。

# ### 结果：
# - **训练集和测试集**上的分类效果可以通过绘图可视化，决策边界将在图上显示。
# - 模型的权重 `W` 和偏置 `b` 将在终端打印。

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 生成一些示例数据
X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [3, 3], [2, 1], [6, 4]])
y = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, -1])

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# 定义 SVM 模型
class SVM(nn.Module):
    def __init__(self, n_features):
        super(SVM, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)

# 超参数
learning_rate = 0.01
n_iters = 1000
lambda_param = 0.01  # 正则化参数

# 实例化模型
model = SVM(n_features=2)

# 定义损失函数（Hinge Loss）和优化器（SGD）
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 定义 Hinge Loss (SVM 损失函数)
def hinge_loss(output, target):
    return torch.mean(torch.clamp(1 - output * target, min=0))

# 训练模型
for epoch in range(n_iters):
    # 前向传播
    outputs = model(X_train)
    
    # 计算损失，包括 L2 正则化
    loss = hinge_loss(outputs, y_train) + lambda_param * torch.norm(model.linear.weight, 2)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{n_iters}], Loss: {loss.item():.4f}')

# 打印模型参数 W 和 b
W = model.linear.weight.detach().numpy()
b = model.linear.bias.detach().numpy()
print(f'获得的模型参数 W: {W}, b: {b}')

# 在测试集上进行预测
with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred = torch.sign(y_test_pred).numpy()

# 可视化结果
def plot_svm(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)

    # 绘制决策边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    xy_tensor = torch.tensor(xy, dtype=torch.float32)
    Z = model(xy_tensor).detach().numpy().reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,
               linestyles=['-'])
    
    plt.title("SVM Classification Result")
    plt.show()

# 绘制训练集的结果
plot_svm(X_train.numpy(), y_train.numpy(), model)

# 绘制测试集的结果
# plot_svm(X_test.numpy(), y_test.numpy(), model)