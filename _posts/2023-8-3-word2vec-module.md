---
title: Word2vec module study
tags: Computer-Science
---

​	word2vec模型能够根据语料库生成相应的词向量，并实现提供上下文预测相应单词的能力。

#### 词向量

​	wiki的定义是这样的：
​	词嵌入（Word embedding）是自然语言处理（NLP）中语言模型与表征学习技术的统称。概念上而言，它是指把一个维数为所有词的数量的高维空间嵌入到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量。
​	为什么要使用词向量呢？这里不得不提另一种编码——独热编码(one-hot-code)，独热码被用来表示一种特殊的位元组或向量，该位元组或向量里仅容许其中一位为1，其他位都必须为0。举个例子有apple,orange,banana编码为独热码即为（1,0,0），（0,1,0），（0,0,1），这里可以发现使用独热码无法区分两个单词的关系如:$\lVert apple-orange \rVert=\lVert apple-banada \rVert$ ，而采用词向量的方法便可以加以区分。

### 连续词袋模型（Continuous Bag-of-Word Model）

#### One-word context

先从最简单的情况考虑context只有一个单词的情况，模型目标是通过一个单词来预测另一个单词。
<img src="/assets/blog_pics/20180501155805140.png" />

$x_1,x_2...x_v$表示一个单词的独热编码,W<sub>N×V</sub> 表示输入层矩阵, W<sup>'</sup><sub>N×V</sub>表示输出层矩阵。
矩阵W<sub>V×N</sub>每一行代表的是一个与输入层相关的单词的N维向量表示形式$v_{\omega}$在经过训练后可以表示为单词所对应的词向量，N的大小由训练者确定。可得公式 : 
$$
\begin{equation}
h = W^Tx=V^T_{\omega I}
\end{equation}
$$
其中$v_{\omega I}$表示输入单词$\omega I $的对应向量，训练后可作为词向量。由于$x_1,x_2...x_v$为独热码，实际上h向量是由W矩阵的第i行复制而来($x_i=1$)，得到向量$\mathbf{h}$后可以计算每个单词的得分情况
$$
\begin{equation}
\mu_j={V^{'}_{\omega j}}^T h
\end{equation}
$$
$V^{'}_{\omega j}$为矩阵$W^{'}$的第j列向量(因为使用的是W'的转置矩阵),使用softmax函数进行归一化计算单词的后验分布(貌似是个归一化函数，具体细节没深入了解)，
$$
\begin{align}
p(\omega j|\omega I)=y_j=\frac{exp(\mu_j)}{\sum^{V}_{j'=1}exp(\mu_{j'})}
\end{align}
$$
将(1)(2)带入(3)即可得最终公式(4)(懒得写了)

- #### 隐藏层权重矩阵更新

模型训练最终目标是求公式(4)的最大值，即在训练中给定(contex,world)对，根据contex模型输出的值的最大概率的项经过不断训练调参后与给定world一致。
论文中给出损失函数为
$$
\begin{align}
max p(\omega o|\omega I) & = max y_{j*}\\
&=maxlog y_{j*}   \nonumber\\
&=\mu_{j*}-log\sum^{V}_{j'=1}exp(\mu_{j'}):=-E \nonumber
\end{align}
$$
没看懂。。。不过我看后面大概就是
$$
\begin{align}
E = \frac{1}{2}(y_j-t_j)^2 \nonumber
\end{align}
$$
当理想输出world的独热编码的第j个向量分量为1时$t_j=1$，否则$t_j=0$
得到
$$
\begin{align}
 \frac{\partial \mu_j}{\partial \omega^{'}_{ij}} &= h_i \\ 
\frac{\partial E}{\partial u_j}& = y_j - t_j:=e_j \\
\frac{\partial E}{\partial \omega^{'}_{ij}}& = \frac{\partial E}{\partial \mu_j}\cdot\frac{\partial \mu_j}{\partial \omega^{'}_{ij}}=e_j \cdot h_i
\end{align}
$$
(6)是由于$\omega$矩阵经过转置后 $\omega^{'}_{ij}$与$h_i$相乘

得到隐藏层到输出层矩阵权重更新公式：
$$
\begin{align}
{\omega^{'}_{ij}}^{new}={\omega^{'}_{ij}}^{old}-\eta\cdot e_j\cdot h_i
\end{align}
$$
迭代即可。

- #### 输入层矩阵更新

  由于$h_i$与$\mu_1,\mu_2...\mu_v$均有关系，故得

$$
\frac{\partial E}{\partial h_i}=\sum^V_{j=1}\frac{\partial E}{\partial \mu_j}\cdot \frac{\partial \mu_j}{\partial h_i}=\sum^V_{j=1}e_j\cdot\omega^{'}_{ij}:=EH_i
$$

这里$\mathbf{EH}$是N维向量
$$
\begin{align}
\frac{\partial E}{\partial \omega_{ki}}=\frac{\partial E}{\partial h_i}\cdot \frac{\partial h_i}{\partial \omega_{ki}}=EH_i \cdot x_k
\end{align}
$$
论文中给出了张量乘积的表达形式：（不太懂）
$$
\begin{align}
\frac{\partial E}{\partial W}=x \otimes EH=xEH^T
\end{align}
$$
故矩阵的更新公式为
$$
\begin{align}
{V_{\omega I}}^{new}={V_{\omega I}}^{old}-\eta \cdot EH^T
\end{align}
$$

#### Multi-word context

<img src="/assets/blog_pics/20180501151808331.png"/>

其隐藏层的输出值的计算过程为：首先将输入的上下文单词（context words）的向量叠加起来并取其平均值，接着与$input \rightarrow hidden$的权重矩阵相乘，作为最终的结果，公式如下：
$$
\begin{align}
h&=\frac{1}{C}W^T(x_1+x_2+...+x_c) \nonumber \\
&=\frac{1}{C}{(V_{\omega 1}+V_{\omega 2}+...+V_{\omega c})}^{T}
\end{align}
$$
其与公式与上文one-world模型基本一致

### Skip-Gram Model

<img src="/assets/blog_pics/20180426181210689.png"/>

该模型根据中心词汇来预测context

需要根据预期输出的每个单词的损失来更新矩阵$W^{'}$，其余貌似和上个模型差不多（懒得想了，有空看看源码再说）

### Optimizing Computational Efficiency

对于输出层矩阵更新，需要迭代的参数过多（具体怎么多还没细看）下面介绍两种优化算法：

#### Hierarchical Softmax

<img src="/assets/blog_pics/20180427145827697.png"/>

用二叉树的叶子结点表示叶子上单词的概率
$n(\omega ,j)$表示从根节点到单词$\omega$的路径上的第j个节点
二叉树的每一个内部节点有输出向量$v^{'}_{(\omega,j)}$，可以作为单词的概率计算单元
引入sigmoid函数来将每个节点的输出值限定在(0,1)范围内。
大致过程如下：对于每个节点输入参数为该节点当前的向量值，从根节点开始经过激活函数signoid确定其路径指向左孩子节点还是右孩子节点，公式表达为：
$$
\begin{align}
p(n,left)&=\sigma({v^{'}_n}^T \cdot h) \\
p(n,right)&=1-\sigma({v^{'}_n}^T \cdot h)
\end{align}
$$
h是输入单词的词向量表示，多个单词加权取平均即可，从根节点到对应单词叶子结点概率累乘即得最终概率。
可证（论文作者说的，我也没证过）：
$$
\sum^{V}_{i=1}p(\omega i=\omega o)=1
$$
更新公式为（没细看论文，不过直观感知应该是对的):
$$
{V^{'}_{j}}^{new}={V^{'}_{j}}^{old}-\eta(\sigma({V^{'}_{j}}^{T}h)-t_j)\cdot h,for j=1,2,...,L(\omega)-1
$$
使用该模型可以将计算复杂度冲O(V)降低到O(log(V))

### Negative Sampling

没细看，论文说就是只挑选一部分输出向量，减小规模，日后补坑~

参考论文<<word2vec Parameter Learning Explained>>

PS:第一次写blog，latex公式好多都显示不出来好麻烦www