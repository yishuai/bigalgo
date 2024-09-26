import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 构造数据集
# x = np.random.rand(100, 1) * 10
# x = np.array([[1],[2]])
# y = 2 * x
# y = 1 * x + 2
x = np.random.rand(20, 1) * 10
y = 2 * x + 3 
# np.random.randn(100, 1)
# 转换为PyTorch张量
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print("x_tensor:",x_tensor)
print("y_tensor:",y_tensor)

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度为1，输出维度为1

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# 定义损失函数和优化器
criterion = nn.MSELoss()

params = [{'params': model.linear.weight, 'lr': 0.01},  # 设置线性层参数的学习率为0.01
          {'params': model.linear.weight, 'lr': 0.01},  # 设置线性层参数的学习率为0.01
          {'params': model.parameters()}]  # 默认学习率为0.001

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for param_group in optimizer.param_groups:
    print("Parameters in this group:", param_group['params'])
    print("Learning rate:", param_group['lr'])

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):

    # 获取权重参数
    weight_param = model.linear.weight.data.item()
    print("Weight parameter: w = ", weight_param)

    # 获取偏置参数
    bias_param = model.linear.bias.data.item()
    print("Bias parameter: b = ", bias_param)

    # 前向传播
    outputs = model(x_tensor)
    print("outputs:", outputs)
    print("outputs-y_tensor:", outputs-y_tensor)

    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 获取模型参数的梯度
    for param in model.parameters():
        print(param.grad)

    # 获取权重参数
    weight_param = model.linear.weight.data.item()
    print("Weight parameter: w = ", weight_param)

    # 获取偏置参数
    bias_param = model.linear.bias.data.item()
    print("Bias parameter: b = ", bias_param)

    # if (epoch+1) % 100 == 0:
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # 获取模型参数
    # for param_tensor in model.parameters():
    #     print(param_tensor.data)

    # 可视化结果
    predicted = model(x_tensor).detach().numpy()

plt.scatter(x, y, color='blue')
plt.plot(x, predicted, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.show()
print("OK")