# 以下是使用 **SVD（奇异值分解）** 的最简单 Python 代码示例，使用 `NumPy` 来进行 SVD 操作。


# ### 代码解释：
# 1. **创建矩阵**：我们生成了一个 \(4 \times 3\) 的随机矩阵 `A`。
   
# 2. **进行 SVD 分解**：
#    - 使用 `np.linalg.svd()` 进行奇异值分解，得到三个矩阵 `U`、`S`、`Vt`：
#      - `U`：左奇异向量构成的矩阵，形状为 \( m \times m \)。
#      - `S`：奇异值向量，包含分解中非负的奇异值。
#      - `Vt`：右奇异向量的转置矩阵，形状为 \( n \times n \)。

# 3. **重构原始矩阵**：
#    - 将奇异值 `S` 扩展为对角矩阵后，通过矩阵乘法 \( U \cdot S \cdot V^T \) 重构原始矩阵。

# ### 输出：
# - 原始矩阵 `A`。
# - 分解得到的 `U`、`S` 和 `Vt` 矩阵。
# - 通过 `U`、`S`、`Vt` 重构后的矩阵，与原始矩阵 `A` 非常接近。

# SVD 可以应用于很多场景，包括数据降维、噪声过滤等。

import numpy as np

# 创建一个随机矩阵
A = np.random.rand(4, 3)
print("原始矩阵 A：")
print(A)

# 使用 NumPy 的 SVD 函数进行奇异值分解
U, S, Vt = np.linalg.svd(A)

# 输出分解结果
print("\n矩阵 U：")
print(U)

print("\n奇异值向量 S：")
print(S)

print("\n矩阵 V 的转置 Vt：")
print(Vt)

# 通过 U, S, Vt 重构原始矩阵
S_full = np.zeros((U.shape[0], Vt.shape[0]))
S_full[:S.shape[0], :S.shape[0]] = np.diag(S)
A_reconstructed = np.dot(U, np.dot(S_full, Vt))

print("\n重构的矩阵 A：")
print(A_reconstructed)