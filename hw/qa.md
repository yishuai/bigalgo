# 推荐

- 举例说明 Utility 矩阵的物理意义

- 举例说明 Explicit rating 和 Implicit Rating 的 物理意义

- 举例说明 “冷启动” 难题

- 举例说明 基于内容的推荐 的 基本原理

- （计算题）给定 电影 和 用户的 内容特征向量，计算其匹配度

- 分析 基于内容的推荐 的 优缺点

- 举例说明 基于 合作过滤 的推荐 的 基本原理

- （计算题）给定 Utility 矩阵，计算一个用户对一个 Item 的可能评分

- 实际中， 比较 基于 Item-Item 的 算法 和 基于 User-User 的算法，哪个 性能 好？为什么？

- 分析 基于 合作过滤 的推荐 的 优缺点

- 关于 Latent Factor 推荐模型

a）写出它的优化目标函数，解释其中各项的物理意义。

b）写出 通过 “梯度下降方法” 求解该优化问题的步骤。写出“梯度”的数学表达式。

c）写出 SGD 的英文全称，解释它和 GD 的差别，比较它们的优点

# 通过试验学习

- 给出 MAB 问题的 最优策略 的 数学表达式，介绍其物理意义

- 写出 Epsilon-Greedy 算法，分析其优点和不足。它是 最优策略 吗？

- 写出 UCB 算法，分析其优点和不足。它是 最优策略 吗？

- 举例说明，什么是 Contextual Bandit 问题

# 语言模型

- 给出语言模型的数学表达式，解释其物理意义

- 给出 Unigram、Bigram、Trigram 语言模型的数学形式，解释其物理意义

- （计算题）给定 Bigram 的 单词统计表，估计一句话 的 概率

- 给出 Bigram 的 Perplexity 的 数学表达式，解释其物理意义

- （是非题）在一个 数据集上 评估一个语言模型的性能时，其 Perplexity 越高越好

- 给出 语言模型 估计时的 加1 Laplace 平滑 的数学表达式，解释其物理意义

- 举例说明 Linear Interpolation 平滑 的基本原理，要求用数学公式解释

- 举例说明 ”Stupid Backoff“ 平滑 的基本原理，要求用数学公式解释

- 比较上述三种平滑方法的性能，说明其应用场合

# Logistic 回归（LR）

- 为什么说 Logistic 回归 是 Discriminative 分类器？ 写出其优化目标函数的数学表达式，然后解释

- 写出 LR 模型的数学表达式，解释其物理意义

- （案例分析题）给定一段文本，请设计对其进行情感分类的特征

- 写出 0/1 分类问题 的 交叉熵 （Cross-Entropy）Loss 函数 的 数学表达式

- 推导 LR 模型下， Cross-Entropy Loss 的 梯度。要求给出推导过程。给出推导结果的物理意义

# 神经元网络（NN）

- 画出 ReLU 激活函数的形状，给出其数学公式

- 给出 Softmax 的数学表达式

- 画图说明 NN 计算时的 Forward Pass 和 Backward Differentiation 的基本原理

# Transformer

- 画图说明 Self-Attention 的基本原理，写出其数学形式。解释为什么需要它

- 画图说明 Positional encoding 的基本原理，写出其数学形式。解释为什么需要它

- 画图说明 多头注意力 Multi-Head Attention 的基本原理，写出其数学形式。解释为什么需要它

- 画图说明 Masked attention 的基本原理，写出其数学形式。解释为什么需要它

- 画出 Transfomer 的结构图，解释每一个模块的作用

# 预训练

- 简述有监督学习和无监督学习的区别

- 给出 word2vec 算法的数学描述，解释其工作原理

- 分析 word2vec 是有监督学习，还是无监督学习

- 简述 word2vec 得到的表征的不足。为什么 Contextual representation (表征) 可以改进它？

- 简述采用 BERT 等 预训练模型 进行  Contextual representation (表征) 学习 的 原理

- 简述 BERT 预训练 时 采用 的 MLM 方法

- 画图说明，用 BERT 预训练模型 实现 文本分类 的 输入 和 输出 配置方法

- 简述 如何 利用 BERT 预训练 模型 提取 文本 特征

- 简述 GPT 和 BERT 的网络结构区别。结合该区别，解释 为什么 只有 GPT 可以做 文本生成

