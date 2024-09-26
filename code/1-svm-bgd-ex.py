# GPT4 Prompt：
# 请给出一个最简单的 Batch gradient descent 的 SVM 代码示例，请包括训练集和测试集的划分。在训练集训练后，打印出获得的模型参数 W 和 b 的值。最后在测试集上进行测试，并画图展示结果

# 为了展示SVM模型在训练集和测试集上的表现，可以将数据集分为训练集和测试集，并使用`matplotlib`库绘制分类结果的图形。

# 下面是完整的代码示例，包括训练、测试集的划分以及结果的可视化：

### 代码解释：
# 1. **数据集分割**：使用`train_test_split`将数据集分为训练集和测试集。训练集占80%，测试集占20%。
# 2. **分类准确率**：通过`np.mean()`计算训练集和测试集的预测准确率。
# 3. **结果可视化**：
#    - `plot_svm()` 函数用来绘制数据点以及SVM的决策边界。
#    - 使用`matplotlib`库绘制散点图，并在其上叠加SVM的决策边界和支持向量。

### 结果展示：
# 1. **训练集** 和 **测试集** 的准确率会在终端显示。
# 2. 会生成两幅图：一幅展示训练集数据和决策边界，另一幅展示测试集数据和决策边界。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 线性核的 SVM 模型类
class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # 将标签转换为 -1 和 1

        # 初始化权重和偏差
        self.w = np.zeros(n_features)
        self.b = 0

        # 批量梯度下降
        for n in range(self.n_iters):
            print(n)
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * INPUT1)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * INPUT1 - INPUT2)
                    self.b -= self.learning_rate * INPUT3

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# 生成一些示例数据
X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [3, 3], [2, 1], [6, 4]])
y = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, -1])

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 SVM
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

print(svm.w)
print(svm.b)

# 预测训练集和测试集
y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)

# 计算分类准确率
train_accuracy = np.mean(y_train == y_train_pred) * 100
test_accuracy = np.mean(y_test == y_test_pred) * 100

print(f"训练集准确率: {train_accuracy:.2f}%")
print(f"测试集准确率: {test_accuracy:.2f}%")

# 可视化结果
def plot_svm(X, y, svm):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)
    
    # 绘制决策边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm.predict(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    plt.title("SVM Classification Result")
    plt.show()

# 绘制训练集的结果
plot_svm(X_train, y_train, svm)

# 绘制测试集的结果
plot_svm(X_test, y_test, svm)
