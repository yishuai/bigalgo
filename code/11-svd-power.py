# 以下是使用 **Power Iteration** 方法手动实现 **SVD（奇异值分解）** 的最简单 Python 代码示例。Power Iteration 是一种迭代方法，用于计算矩阵的主奇异值及其对应的奇异向量。

# ### 代码解释：
# 1. **Power Iteration**：我们定义了 `power_iteration` 函数，它通过迭代的方法来逼近矩阵 \(A^T A\) 的最大奇异值及其对应的右奇异向量 `v`。使用 `v`，我们还可以计算对应的左奇异向量 `u`。
   
# 2. **输入矩阵**：我们创建了一个随机的矩阵 `A`，然后通过 `power_iteration` 方法来计算它的最大奇异值 `sigma` 及其对应的左右奇异向量 `u` 和 `v`。

# 3. **重构矩阵**：我们使用最大奇异值 `sigma` 和奇异向量 `u`、`v` 来重构矩阵 `A` 的近似值。这只利用了最大的奇异值及其对应的奇异向量。

# ### 输出：
# - 原始矩阵 `A`。
# - 计算出的最大奇异值 `sigma`。
# - 对应的左奇异向量 `u` 和右奇异向量 `v`。
# - 用这一个奇异值和奇异向量重构出的矩阵 `A` 的近似值。

# ### Power Iteration 的作用：
# Power Iteration 是计算矩阵主奇异值的一个简单有效方法。虽然在这里我们只计算了最大的奇异值和其对应的奇异向量，但通过进一步迭代，我们可以得到更多的奇异值及其对应的奇异向量，用来进行矩阵分解。

import numpy as np

# Power iteration to find the dominant singular value and corresponding singular vectors
def power_iteration(A, num_simulations=100, epsilon=1e-6):
    # Randomly initialize a vector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A.T @ A, b_k)

        # Re-normalize the vector
        b_k1_norm = np.linalg.norm(b_k1)
        if b_k1_norm < epsilon:
            break
        b_k = b_k1 / b_k1_norm

    # Calculate the singular value
    sigma = np.linalg.norm(np.dot(A, b_k))

    # Calculate the left singular vector u
    u = np.dot(A, b_k) / sigma

    return sigma, u, b_k

# Example matrix
A = np.random.rand(4, 3)
print("Original Matrix A:")
print(A)

# Find the largest singular value and vectors using power iteration
sigma, u, v = power_iteration(A)

print("\nDominant Singular Value (Sigma):")
print(sigma)

print("\nLeft Singular Vector (u):")
print(u)

print("\nRight Singular Vector (v):")
print(v)

# Reconstruct an approximation of the matrix A using the largest singular value and vectors
A_approx = sigma * np.outer(u, v)
print("\nApproximated Matrix A (using dominant singular value/vector):")
print(A_approx)