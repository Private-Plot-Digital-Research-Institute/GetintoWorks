职位描述

负责公司大模型的研发和应用，研究相关技术在搜索、推荐、广告、创作、对话和客服等领域的全新应用和解决方案，满足用户不断增长的智能交互需求。主要工作方向包括: 1、探索超大规模模型,并进行极致系统优化; 2、数据建设、指令微调、偏好对齐、模型优化; 3、相关应用落地,包括生成创作、逻辑推理、代码生成等; 4、就大模型使用场景进行深入研究和探索。

职位要求

1、2024届获得硕士及以上学历，计算机、软件工程等相关专业； 2、优秀的代码能力、数据结构和基础算法功底，熟练C/C++或Python，ACM/ICPC、NOI/IOI、Top Coder、Kaggle等比赛获奖者优先； 3、熟悉NLP、CV相关的算法和技术，熟悉大模型训练、RL算法者优先； 4、在大模型领域，主导过大影响力的项目或论文者优先； 5、出色的问题分析和解决能力，能深入解决大模型训练和应用存在的问题； 6、良好的沟通协作能力，能和团队一起探索新技术，推进技术进步。


### RL

#### 基础概念

[深度强化学习面试题目总结-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1773320)

[强化学习算法面试问题 & 解答_强化学习开题答辩-CSDN博客](https://blog.csdn.net/ao1886/article/details/118424792)

值迭代(Value-based)和策略迭代(Policy-based)的区别？

  价值迭代采用了Bellman最优算子，策略迭代采用的是Bellman期望算子。价值迭代是策略迭代的一种特殊情况，是每进行一次策略评估就更新一次策略。(?)

**有哪些方法可以使得RL训练稳定？**  
1. 选用方差较小的方法，采用合理的奖励函数
2. 梯度裁剪

 简述时间差分(TD)算法

  TD算法和MC算法一样可以从和环境互动的经验中学习策略而不依赖环境的动态特性，TD和DP一样都采用的自举法，是采样更新。

**怎么处理稀疏奖励（sparse reward）**
奖励重塑：reward shaping
exploration调整
好奇心：好奇心驱动是使用内在奖励鼓励agent探索更陌生的状态，平衡探索与利用，本质上是提高了样本

[强化学习及深度强化学习面试题-CSDN博客](https://blog.csdn.net/u010705932/article/details/105727130)

**手工推导策略梯度过程？**


$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} [\sum_t r(s_t, a_t)]
$$

求log反导，重要性采样，

[《Easy RL》面试题汇总 - 凯鲁嘎吉 - 博客园 (cnblogs.com)](https://www.cnblogs.com/kailugaji/p/16140474.html)

**分别写出基于状态值函数的贝尔曼方程以及基于状态-动作值函数的贝尔曼方程**

状态值函数： 对某状态(s)下不同策略(\pi)值函数

$$
V^\pi(s) = \mathbb{E}_{a\sim \pi(a|s)} \mathbb{E}_{s' \sim p(s' | s, a)}[r(s, a, s') + \gamma V^\pi(s')]
$$

第一个期望是动作在策略的分布，第二个期望是动作转移状态的分布，期望内的参数是奖励和未来步骤的参数

总结： 给定状态下策略的价值期望，等于从策略中得出的动作使得状态转移价值的期望

状态-动作值函数（Q函数）： 对某状态、动作下不同

$$
Q^\pi(s, a) = \mathbb{E}_{s' \sim p(s' | s, a)} [r(s, a, s') + \gamma \mathbb{E}_{a' \sim \pi(a' | s')}[Q^\pi(s', a')]]
$$

第一个期望： 状态转移期望

第二个期望： 采样动作期望



#### DQN

**DQN的原理是什么？**  
DQN对Q-learning的修改主要体现在以下三个方面：DQN利用深度卷积神经网络逼近值函数（表达能力增强，可以处理状态空间动作空间较大的问题）；DQN利用了经验回放对强化学习的学习过程进行训练（打破数据关联性，保证输入神经网络的数据独立同分布）；DQN独立设置了目标网络来单独处理时间差分算法中的TD偏差（打破关联性，提高收敛能力）



#### SAC


#### TRPO

#### PPO


### NLP

[BERT相关面试题（不定期更新） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/151412524)

词嵌入，向量检索

#### Bytepair Encoding

[理解NLP最重要的编码方式 — Byte Pair Encoding (BPE)，这一篇就够了 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/424631681)

BPE每次把相邻出现频率最高的两个字符编码为一个新token，且这个过程可以迭代产生。 总token数量一般先增加后减少


#### Word Embedding

词嵌入（Word Embedding）是自然语言处理（NLP）中的一种技术，它将每个单词表示为预定义向量空间中的实值向量。以下是关于词嵌入的20%关键要点，可以帮助你回答80%的相关问题：

1. **词嵌入的定义**：词嵌入是一种将单词表示为实值向量的技术，这些向量通常具有数十到数百的维度。这种表示方法允许具有相似含义的单词具有相似的表示[1][3][4]。

2. **词嵌入的目标**：词嵌入的目标是捕获单词的上下文，语义和句法相似性，以及单词之间的关系。这种方法使得在向量空间中，相似含义的单词在空间上更接近[2][3]。

3. **词嵌入的生成**：词嵌入可以通过多种方法生成，包括神经网络、对单词共现矩阵进行降维、概率模型、可解释的知识库方法，以及根据单词出现的上下文进行显式表示[4]。

4. **主要的词嵌入算法**：主要的词嵌入算法包括Word2Vec和GloVe。Word2Vec有两种模型，一种是CBOW（Continuous Bag-of-Words），另一种是Skip-Gram。CBOW模型通过上下文预测当前单词，而Skip-Gram模型则通过当前单词预测周围的单词[1][2]。GloVe（Global Vectors for Word Representation）是一种全局对数双线性回归模型，用于无监督学习单词表示，它在单词类比、单词相似性和命名实体识别任务上表现优秀[1]。

5. **词嵌入的使用**：你可以选择为你的问题学习一个词嵌入，这将需要大量的文本数据，例如数百万或数十亿的单词。你也可以选择重用一个预训练的词嵌入，例如Word2Vec和GloVe词嵌入都可以免费下载[1]。

以上就是关于词嵌入的关键要点，希望对你有所帮助。

Citations:
[2] https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa
[3] https://www.turing.com/kb/guide-on-word-embeddings-in-nlp
[4] https://en.wikipedia.org/wiki/Word_embedding
[5] https://research-information.bris.ac.uk/en/studentTheses/concepts-in-word-embeddings


N-Gram是一种基于统计语言模型的算法。它的基本思想是将文本里面的内容按照字节进行大小为N的滑动窗口操作，形成了长度是N的字节片段序列。

该模型基于这样一种假设，第N个词的出现只与前面N-1个词相关，而与其它任何词都不相关，整句的概率就是各个词出现概率的乘积。这些概率可以通过直接从语料中统计N个词同时出现的次数得到。常用的是二元的Bi-Gram和三元的Tri-Gram。

 

跳字模型（skip-gram）：通过中心词来推断上下文一定窗口内的单词。

## What are embeddings?

](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)

OpenAI’s text embeddings measure the relatedness of text strings. Embeddings are commonly used for:

- **Search** (where results are ranked by relevance to a query string)
- **Clustering** (where text strings are grouped by similarity)
- **Recommendations** (where items with related text strings are recommended)
- **Anomaly detection** (where outliers with little relatedness are identified)
- **Diversity measurement** (where similarity distributions are analyzed)
- **Classification** (where text strings are classified by their most similar label)

An embedding is a vector (list) of floating point numbers. The [distance](https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use) between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.

#### Vector Indexing

#### 

**如何评估文本生成模型的性能？**  
评估文本生成模型的性能通常涉及到两个方面：一是生成文本的质量，二是生成文本的多样性。质量可以通过人工评估或者使用自动评估指标（如BLEU、ROUGE、METEOR等）来评估。多样性可以通过计算生成文本的不同n-gram的数量或者使用诸如自我重复率这样的指标来评估


### LLM

#### Context Length


[Dao-AILab/flash-attention: Fast and memory-efficient exact attention (github.com)](https://github.com/Dao-AILab/flash-attention)

We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method.

发现的问题： 

1. 对Attention的优化不会带来太多墙上时间（整体耗时）的增益。原因：性能瓶颈在IO一侧

[[2306.15595] Extending Context Window of Large Language Models via Positional Interpolation (arxiv.org)](https://arxiv.org/abs/2306.15595)

grouped-query attention

#### PEFT

[[PEFT]]


#### Cross-modality

align, fuse


#### transformer

针对transformer架构：可调参数有: d_model (也就是context-length),  d_ff(全连接层, 会按照d_model生成的两次投影), n_hea

不可调有： position_embedding(位置编码，是d_model的函数)

W_Q, W_K, W_V 都是d_model * d_model / n_head  * n_head(每一个head用不同的W)

W_O： d_model * d_model

原版一个block：( 3 * 512* 512 + 512 * 512 ) + (512 * 2048 + 2048 + 2048* 512 + 512) = 3m


params:  -> 一个解码器有两个attention 和一个ff block

编码器: 3m * 6 = 18m 解码器: (2m + 2m) * 6 = 24m  

Complexity per layer(n^2 d) n : sequance length, d : representation dimension

1.对于一个文本，我们希望找到某张图片中和文本描述相关的局部图像，怎么办？文本作query(查询），图像做value（数据库）

2.对于一个图像，想要找一个文本中和图像所含内容有关的局部文本，如何设计？图像作query，文本作value.

3.自注意力（我查我自己）:我们想知道句子中某个词在整个句子中的分量（或者相关文本），怎么设计？句子本身乘以三个矩阵得到Q,K,V，每个词去查整个句子。

4.交叉注意力（查别人）:transformer模型的decoder中，由decoder的输入经过变换作为query，由encoder的输出作为key和value（数据库）。value和query来自不同的地方，就是交叉注意力。可以看到key和value一定是代表着同一个东西。即:[Q,(K,V)]。如果用encoder的输出做value，用decoder的输入做key和query 那就完完全全不make sense了。
  
  
作者：匿名用户  
链接：https://www.zhihu.com/question/325839123/answer/2718310467  
来源：知乎  
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


scaled- pairwise dot-product attention

#### 调参经验



### 过往项目

#### Self-Evaluator

跟随Andrea karpathy搭建一套大模型反强化学习框架，使用了ChatGLM6B开源模型和Huggingface PEFT。 难点： 如何让大模型的评估形成梯度优化自身的参数

解决方案： 

1. 用少量数据集（种子问题）生成领域知识
2. 用随机差和不同beam-search参数生成不同问题
3. 扔掉大模型不确定的数据

效果：在8卡机上训练微调6B模型（3个peft模型），用PPO做RL算法

同时塞三个大模型： 打分模型、生成模型，生成模型梯度又是两倍内存

定量： Self Evaluator在测试集上能够以96%的概率识别出更好的回复

RL后的大模型能够生成更好的回复（虽然不稳定）

RL的LOSS为啥不放上来？ -> 不好看（起伏比较大）

为啥不用Prefix: Limited Context Length

遇到的困难？

1. 显存 -> 别的同学也在用
2. RL框架 -> 不能直接用现有框架，单卡训练，但是模型部署在不同卡上
3. LoRA（精度转换，模型适配）

### HPC


[CS267 Spring 2021 (google.com)](https://sites.google.com/lbl.gov/cs267-spr2021)


## 面试记录
TRPO, PPO

会问推导， KLDiv

Transformer的改进，attention的最新工作

Let's verify step by step


FP16, BF16,FP32

PPO CLIP

RoPE


