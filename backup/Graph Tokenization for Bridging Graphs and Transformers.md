# Graph Tokenization for Bridging Graphs and Transformers

- 会议/期刊：ICLR
- 年份：2026
- 论文链接：https://arxiv.org/abs/2603.11099
- 代码链接：https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers


## 一句话总结

这篇论文提出 GraphTokenizer，通过“可逆图序列化 + 结构频率引导 + BPE”的方式，将带标签图转换成离散 token 序列，使 BERT、GTE 等标准 Transformer 无需图专用结构修改即可处理图分类和图回归任务，并在 14 个 benchmark 上取得优于多数 GNN 和 Graph Transformer 的结果。

## 研究问题

### 背景

Transformer 在 NLP、CV、音频等领域取得成功，很大程度上依赖 tokenizer 将原始输入转换成离散 token 序列。文本天然具有线性顺序，因此 tokenization 相对自然；但图数据没有固定节点顺序，邻域结构是分支状的，而且同构图在节点编号变化后应被视为同一个图。

因此，如何把图结构数据转换成标准 Transformer 能直接处理的 token 序列，是连接图学习与大模型生态的关键问题。

### 核心问题

本文要解决的问题是：如何设计一个通用的 graph tokenizer，使其能够把任意带标签图转换为离散 token 序列，同时尽可能满足以下要求：

1. 保留图的拓扑结构和节点/边标签信息；
2. 对节点排列变化具有稳定性；
3. 生成适合 Transformer 处理的序列；
4. 不依赖图专用模型结构；
5. 能从数据中自动学习有意义的图子结构 token。

### 现有方法不足

- GNN 主要依赖局部消息传递，长程依赖建模能力有限，且难以直接复用标准 Transformer 的训练和扩展生态。
- Graph Transformer 通常需要加入图结构偏置、位置编码或特殊 attention 设计，模型架构不再是标准 Transformer。
- 将图映射为连续 embedding 再输入 Transformer 的方法可能产生信息损失、重构歧义或表示不稳定。
- Random Walk、BFS、DFS 等图序列化方法通常不可逆，容易丢失完整边连接信息。
- 普通 Eulerian circuit / CPP 虽然可以保留边结构，但遍历时存在任意选择，导致序列化结果不确定。
- SMILES 虽然适合分子图，但依赖化学领域规则，不能自然推广到任意带标签图。

## 核心思路

1. 先用可逆图序列化方法把图转换成符号序列，保证原始图结构可以从序列中恢复。
2. 再用训练集中的局部子结构频率引导遍历顺序，使常见子结构在序列中更容易相邻出现。
3. 最后在序列化后的图语料上使用 BPE，迭代合并高频相邻符号，学习出具有结构意义的 graph token。
4. 将得到的 token 序列输入标准 Transformer，例如 BERT 或 GTE，用于图分类、图回归，甚至图生成任务。



## 方法要点

### 1. 模块 A：结构频率统计与频率引导序列化

- 做什么：

统计训练集中局部图结构的出现频率，并用这些频率指导图序列化时的边遍历顺序。

- 输入输出：

输入：

- 训练图数据集 $D$；
- 带节点标签和边标签的图 $G = ((V, E), L, \Sigma)$。

输出：

- 局部结构频率表 $F(p)$；
- 每个图对应的结构引导符号序列 $S$。

- 为什么这样设计：

普通图遍历方法存在两个核心问题：

1. BFS / DFS / Random Walk 等方法通常只给出节点序列，缺少完整边信息，因此不可逆；
2. Eulerian circuit / CPP 等方法虽然覆盖边、可逆，但在多个候选边之间可能任意选择，因此不确定。

作者希望同时满足：

- 可逆性：序列能恢复原图结构；
- 确定性：同一个图或同构图应产生稳定序列；
- 结构友好性：高频子结构尽量在序列中相邻，方便 BPE 合并。

因此，论文提出使用全局子结构频率来指导可逆遍历。

- 关键公式/机制：

论文定义基本局部模式为一个带标签边三元组：

$p = (l_u, l_e, l_v)$

其中：

- $l_u$ 表示源节点标签；
- $l_e$ 表示边标签；
- $l_v$ 表示目标节点标签。

在训练集上统计该模式出现次数：

$Count(G, p) = |\{e=(u,v) \in E \mid (L(u), L(e), L(v)) = p\}|$

聚合整个训练集得到：

$C(p) = \sum_{G \in D} Count(G, p)$

并归一化为相对频率：

$F(p) = \frac{C(p)}{\sum_{p'} C(p')}$

在 Frequency-Guided Eulerian circuit 中，当遍历到节点 $u$ 且存在多个未访问候选边时，选择优先级最高的边：

$e^* = \arg\max_{e_i \in E_u} \pi(e_i, F)$

其中 $\pi(e_i, F)$ 可以由该边对应局部模式的频率 $F(p_i)$ 决定。

- 我的理解：

这个模块的关键不是简单地把图拍平成序列，而是让“图中常见结构”在序列中更稳定、更频繁地相邻出现。这样后续 BPE 合并到的 token 就更可能是真正有意义的图子结构，而不是随机符号组合。

### 2. 模块 B：BPE 图词表学习

- 做什么：

在图序列语料上训练 Byte Pair Encoding，将高频相邻符号对逐步合并成新的 graph token，得到图结构词表。

- 输入输出：

输入：

- 经过频率引导序列化得到的图序列集合 $D_S$；
- 初始符号表 $\Sigma$；
- BPE 合并次数 $K$。

输出：

- 图 token 词表 $V_T$；
- BPE 合并规则 $R$；
- 压缩后的图 token 序列 $S_T$。

- 为什么这样设计：

BPE 在 NLP 中用于从字符或子词中学习常见组合。本文将图序列化后，常见局部图结构会形成高频相邻符号对，因此 BPE 可以自动把它们合并成更大的结构 token。

这使 graph token 同时具备：

1. 离散性：可直接输入 Transformer embedding 层；
2. 可逆性：BPE 合并规则可以反向展开；
3. 可解释性：token 往往对应常见子结构；
4. 压缩性：显著缩短图序列长度。

- 关键公式/机制：

BPE 每一轮执行：

1. 统计所有序列中的相邻符号对频率；
2. 找到出现最多的符号对；
3. 将其合并成新 token；
4. 将合并规则加入 codebook；
5. 重复 $K$ 次。

核心步骤可以写成：

$(s_a^*, s_b^*) = \arg\max_{(s_a, s_b)} \sum_{S \in D_S} Count(S, (s_a, s_b))$

$s_{new} = s_a^* \cdot s_b^*$

$R = R \cup \{(s_a^*, s_b^*) \rightarrow s_{new}\}$

- 我的理解：

BPE 在这里不仅仅是压缩工具，更像是在学习一个“图结构词典”。例如在分子图中，它可以从原子和键逐步合并出官能团、环结构、苯环相关结构等更高层次的 token。

### 3. 训练与推理

- 训练：

1. 输入训练图集合 $D$；
2. 统计局部模式频率 $F(p)$；
3. 使用频率引导的可逆序列化函数 $f_g(G, F)$ 将每个图转换成符号序列；
4. 在序列集合 $D_S$ 上执行 BPE；
5. 得到 BPE codebook $C=(V_T, R)$；
6. 将图编码成 token 序列；
7. 使用标准 Transformer backbone，例如 BERT-small 或 GTE-base，完成下游图级任务训练。

- 推理：

1. 对新图 $G$ 使用训练阶段得到的频率表 $F$ 做结构引导序列化；
2. 按照 BPE 合并规则 $R$ 将符号序列编码为 token 序列 $S_T$；
3. 输入标准 Transformer；
4. 对图级任务，可以使用 `[CLS]` token 或 hidden state pooling 得到整图表示；
5. 对生成任务，可以用 decoder-only Transformer 自回归预测下一个 token；
6. 如果需要解码，则反向展开 BPE token，再通过逆序列化函数 $f^{-1}$ 重构图。

- 与已有方法的关键区别：

| 方法类型 | 代表方法 | 主要问题 | 本文区别 |
|---|---|---|---|
| 传统 GNN | GCN、GIN、GAT | 依赖局部消息传递，长程依赖较弱 | 将图转为序列后交给 Transformer 建模 |
| Graph Transformer | GraphGPS、Exphormer 等 | 需要图专用结构设计 | 不修改标准 Transformer 架构 |
| 随机游走序列化 | DeepWalk 等 | 局部片段不可逆，且随机性强 | 使用可逆序列化保留完整图结构 |
| BFS / DFS | 节点序列遍历 | 丢失边连接信息，不可逆 | 使用边覆盖遍历保留拓扑 |
| 普通 Eulerian / CPP | 边覆盖遍历 | 可逆但不确定 | 引入频率引导规则实现确定性 |
| SMILES | 分子图序列化 | 化学领域依赖强 | 面向一般带标签图 |
| 连续 embedding 方法 | GraphGPT、LLAGA 类映射方法 | 可能信息损失，结构表达不稳定 | 使用离散、可逆 graph token |

## 创新点

1. 提出通用 graph tokenizer 框架，将图结构数据转换为离散 token 序列，从而把图学习接入标准 Transformer 生态。
2. 将可逆图序列化与 BPE 结合，使图既能被压缩为 token 序列，又能在需要时恢复原始结构。
3. 提出结构频率引导的确定性序列化策略，让高频图子结构更容易形成相邻符号模式，适合 BPE 学习。
4. 不修改 Transformer 架构，即可在图分类和图回归任务上获得强性能。
5. 学到的 token 具有一定可解释性，例如可以对应分子图中的官能团、环结构等。
6. 将图表示学习重新表述为序列建模问题，为图生成、图基础模型、跨领域图预训练提供了新的接口思路。

## 实验与结论

### 实验设置

- 数据集：

论文在 14 个公开图 benchmark 上实验，覆盖多个领域：

1. 分子图：Mutagenicity、Proteins、OGBG-molhiv、ZINC、AQSOL、QM9；
2. 计算机视觉图：COIL-DEL；
3. 图理论数据：Colors-3、Synthetic；
4. 生物医学图：DD、Peptides；
5. 社交网络：Twitter；
6. 学术网络：DBLP。

- 任务：

1. 图分类；
2. 图回归；
3. 多任务分类；
4. 多任务回归；
5. 附录中还展示了自回归图生成 proof-of-concept。

- baseline：

主要比较对象包括：

1. GCN；
2. GIN；
3. GAT；
4. GatedGCN；
5. GraphGPS；
6. Exphormer；
7. GraphMamba；
8. GCN+；
9. GraphGT；
10. Graphormer；
11. FragNet；
12. Graph-ViT-MLPMixer；
13. GraphGPT；
14. LLAGA；
15. HAN；
16. ChebNet。

- 指标：

分类任务：

- ROC-AUC；
- Average Precision；
- Accuracy。

回归任务：

- MAE；
- Average MAE。

效率指标：

- token 序列长度；
- BPE 压缩率；
- 每个 epoch 的训练时间。

### 主要结果

- GT+GTE 在多个数据集上取得最好或接近最好的结果。
- 在 molhiv 上，GT+GTE 达到 87.4 AUC，强于 GCN、GIN、GAT、GatedGCN、GraphGPS、Exphormer、GraphMamba 和 GCN+。
- 在 p-func 上，GT+GTE 达到 73.1 AP，是主表中最优结果。
- 在 mutag 上，GT+GTE 达到 90.1 accuracy，是主表中最优结果。
- 在 dblp 上，GT+GTE 达到 93.6 accuracy，是主表中最优结果。
- 在 qm9 上，GT+GTE 达到 0.071 MAE，是主表中最优结果。
- 在 aqsol 上，GT+GTE 达到 0.609 MAE，是主表中最优结果。
- 在 ZINC 上，GT+GTE 的 MAE 为 0.131，虽不如 GCN+ 的 0.116，但仍强于大多数传统 GNN 和部分 Graph Transformer。
- BPE 显著压缩序列长度。在 ZINC 上，Frequency-Guided Eulerian circuit + BPE 的压缩比达到约 10.84×。
- 使用 BPE 后训练速度明显提升，说明该方法不仅提升性能，也改善了效率。
- 标准 Transformer 经过 GraphTokenizer 处理后，无需图专用结构即可达到强图学习性能。

### 分析实验

- 消融：

论文主要比较了以下序列化方式：

1. BFS；
2. DFS；
3. Topological Sort；
4. Eulerian circuit；
5. Frequency-Guided Eulerian circuit；
6. CPP；
7. Frequency-Guided CPP；
8. SMILES。

主要结论：

- 可逆的边覆盖方法通常优于 BFS / DFS / Topological Sort 等节点序列方法；
- Frequency-Guided Eulerian circuit 通常优于普通 Eulerian circuit；
- Feuler 不仅效果更好，而且方差更小，训练更稳定；
- CPP 和 FCPP 效果接近，但 CPP 的复杂度为 $O(|V|^3)$，不适合大图；
- BPE 几乎在所有配置下都带来性能提升和序列压缩；
- BPE vocabulary size 的选择存在饱和点，论文中 $K=2000$ 是较好的折中。

- 可视化：

论文在 ZINC 数据集上展示了 BPE 学到的 token：

1. BPE 可以从简单的原子和键合并出 sulfonyl group；
2. 可以从苯环结构继续合并出更复杂的芳香族子结构；
3. 学到的 token 不是纯粹的压缩片段，而是具有化学意义的子结构；
4. 统计上，4–6 个节点的中等规模 token 占比最高，约 41.5%，对应典型官能团大小；
5. 超过 60% 的词表 token 表示多节点子结构。

- 额外发现：

1. 论文将 MNIST 图像转换为 grid graph，用 decoder-only Transformer 做自回归图生成，生成结果可以还原出可识别的数字图像。
2. 与 GraphGPT、LLAGA 等 LLM-based graph model 相比，本文方法在纯结构任务上明显更强。
3. 简单将图 textualize 后交给 LLM，并不能可靠捕捉复杂图拓扑。
4. 对节点计数任务，节点序列化方法可能更适合；而边覆盖序列化更适合一般图结构建模。
5. 这说明序列化策略需要和任务类型匹配。

## 不足与问题

1. 对连续特征支持不足  
   当前框架主要适合节点和边标签为离散符号的图。对于连续节点特征或边特征，需要先离散化或量化，但量化会带来信息损失，这与论文强调的可逆性存在冲突。

2. 主要验证图级任务  
   论文主要实验集中在 graph-level classification 和 graph-level regression，对 node classification、edge prediction、link prediction 等细粒度任务支持还不充分。

3. BPE 可能掩盖目标节点或目标边  
   在节点级或边级任务中，BPE 可能把目标节点/边合并进更大的 composite token，导致原始实体难以直接定位。

4. 大图仍受 Transformer 上下文长度限制  
   虽然 BPE 能显著压缩序列长度，但对于特别大的图，序列仍可能超过 Transformer 的最大上下文长度。

5. CPP 扩展性较差  
   CPP 和 FCPP 的复杂度为 $O(|V|^3)$，在大图上不实用。实际更可行的是线性复杂度的 Frequency-Guided Eulerian circuit。

6. 图 token 的跨领域泛化仍需更多验证  
   不同领域图的结构分布差异很大，例如分子图、社交网络、学术网络、生物网络的局部模式不同。一个领域学到的 vocabulary 能否迁移到另一个领域，仍需要更多实验。

7. 序列化策略与任务之间存在匹配问题  
   对一般结构建模，边覆盖序列化更合理；但对节点计数类任务，边覆盖方法会重复访问节点，可能引入歧义。

## 和我当前研究的关系

- 可借鉴点：

1. 如果我的研究涉及图表示学习，可以借鉴“先设计 tokenizer，再复用标准 Transformer”的范式。
2. 如果我的研究涉及图基础模型，可以将 graph tokenizer 作为不同图数据集之间的统一输入接口。
3. 如果我的研究涉及分子图，可以借鉴其 BPE 学习化学子结构 token 的方式。
4. 如果我的研究涉及图生成，可以参考其将图 token 序列输入 decoder-only Transformer 的思路。
5. 如果我的研究涉及结构化数据与 LLM 融合，可以把 graph tokenization 看作一种结构数据对齐语言模型的接口。
6. 如果我的研究涉及图压缩或图模式挖掘，可以参考 BPE 自动学习高频子结构的思想。

- 可复用模块：

1. Frequency-Guided Eulerian circuit；
2. 图局部模式频率统计 $F(p)$；
3. BPE-based graph vocabulary；
4. 可逆图编码/解码框架；
5. 图 token 序列输入 BERT/GTE 的建模方式；
6. 使用 `[CLS]` token 或 pooling 做图级表示；
7. 将图 token 序列用于 decoder-only 生成模型。

- 能否作为 baseline：

可以作为 graph-level classification / regression 的强 baseline。

适合用作 baseline 的场景：

1. 分子性质预测；
2. 图分类；
3. 图回归；
4. 图级表示学习；
5. 结构模式明显的图任务；
6. 想比较“图专用模型”与“标准 Transformer + tokenizer”性能差异的任务。

不太适合直接作为 baseline 的场景：

1. 节点分类；
2. 链接预测；
3. 边分类；
4. 需要精确定位原始节点/边的任务；
5. 超大规模图任务。

如果用于节点/边级任务，需要进一步改造：

1. 加入 pointer token；
2. 禁止目标节点/边被 BPE 合并；
3. 保留 token 到原始节点/边的映射；
4. 针对任务重新设计序列化策略。

- 对我最有启发的一点：

这篇论文最有启发的一点是：它没有继续设计更复杂的图神经网络或 Graph Transformer，而是从 tokenizer 层解决图和 Transformer 的接口问题。

这说明在图基础模型方向中，“如何把图表示成 token”可能和“使用什么模型架构”一样重要，甚至更重要。只要 tokenizer 足够好，标准 Transformer 本身就能成为强大的图学习模型。

## 最终总结

- 值不值得重点跟进：

值得重点跟进。

尤其适合关注以下方向时深入阅读：

1. 图表示学习；
2. Graph Transformer；
3. 图基础模型；
4. 分子图建模；
5. 图生成；
6. 非文本模态 tokenization；
7. LLM 与结构化数据融合；
8. 图序列化与图压缩；
9. 图数据预训练。

- 一句话评价：

这是一篇把 NLP 中 tokenizer 的成功经验系统迁移到图学习中的论文，核心价值不在于提出复杂的新模型，而在于提出了一种将图结构数据转化为标准 Transformer 可处理 token 序列的通用接口；方法简洁，但范式意义很强。