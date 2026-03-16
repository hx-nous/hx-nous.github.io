# Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs

- **会议/期刊**：NeurIPS 2025 Poster
- **年份**：2025
- **论文链接**：https://openreview.net/forum?id=1nSynwHvu2
- **PDF**：https://openreview.net/pdf?id=1nSynwHvu2
- **代码链接**：论文页面提供了 GitHub 仓库 `bundle-neurips25`

## 一句话总结

这篇论文研究 **文本属性图（Text-Attributed Graph, TAG）上的 zero-shot 节点分类**，提出 **DENSE**：先把一组相近节点组成一个 bundle，让 LLM 预测这组文本的主类别，再用 bundle 级标签监督 GNN，并在训练中动态剔除不合群节点；最终在 10 个数据集上都优于 15 个基线。

## 研究问题

### 背景

文本属性图广泛出现在引文网络、网页网络、电商网络、知识图谱等场景中。每个节点带有自然语言文本，但整图标注通常昂贵，因此 zero-shot inference 是一个很自然且有价值的问题设定。

### 核心问题

如何在 **没有人工节点标签** 的前提下，借助 LLM 做 TAG 上的节点分类，同时尽量利用图中的局部结构信息，并减少 LLM 伪标签噪声对下游模型的伤害。

### 现有方法不足

- **LLM 很难直接吃图结构**：图是非欧式结构，不像文本那样天然是 token 序列。
- **单节点文本信息不足**：只看一个节点文本时，LLM 输出容易不稳定、含糊甚至错误。
- **伪标签噪声会伤害后续训练**：如果把 noisy node label 直接拿去监督 GNN 或传播到图上，下游误差会被放大。

## 核心思路

1. 不让 LLM 直接判断单个节点，而是让它判断一个 **bundle** 中“大多数节点属于什么类”。
2. 不做 node-wise 伪标签监督，而是设计 **bundle-wise supervision**，允许 bundle 中有少量离群点。
3. 在训练过程中，根据 GNN 的预测动态清理 bundle，把最不像 bundle 标签的节点逐步剔除。

## 方法要点

### 1. 模块 A：Bundle Sampling + Bundle Query

- **做什么**  
  先围绕一个 core node 采样出一个 bundle，再把 bundle 中所有文本拼接成 prompt，交给 LLM 预测这组文本的主类别。

- **输入输出**  
  输入是图结构、节点文本和节点文本向量；输出是一个 bundle 以及对应的 LLM 生成 bundle label。

- **为什么这样设计**  
  单个节点文本往往太短、太弱；多个相近节点放在一起，LLM 更容易抓住“局部主题”或“多数类”。

- **关键公式 / 机制**

  对于同配图，作者定义核心节点 $v_c$ 的自适应 $k$-hop 邻域为：

  $$
  N_G^k(v_c)=\{\, i \mid 1\le d_G(v_i,v_c)\le k \,\}, \qquad
  k=\inf\{\,x \mid |N_G^x(v_c)|\ge n_B-1\,\}.
  $$

  这表示：找到最小的 hop 数，使得邻域中至少有 $n_B-1$ 个节点，再从中采样，与核心节点一起构成 bundle。

  对于异配图，作者按语义近邻构造 bundle：

  $$
  B=\{\, i \mid x_i \in N_X^{n_B}(x_c)\,\},
  $$

  其中 $N_X^{n_B}(x_c)$ 表示 embedding 空间里与核心节点向量 $x_c$ 最近的前 $n_B$ 个节点。

  bundle 的 prompt 形式为：

  $$
  P(B)=\langle dataset\_description\rangle\ \mathrm{Concat}(\{t_i\mid i\in B\})\ \langle task\_description\rangle.
  $$

  然后让 LLM 判断：这组文本中大多数属于哪个类别。

- **我的理解**  
  这里真正的 insight 不是“让 LLM 学会图推理”，而是把监督粒度从 **单个节点** 提升到 **局部群体**，让 LLM 去做它更擅长的“归纳主主题”。

### 2. 模块 B：Bundle Supervision + Bundle Refinement

- **做什么**  
  用 LLM 产生的 bundle 标签去监督 GNN；训练中再根据 GNN 的预测，把 bundle 中最不合群的节点删掉。

- **输入输出**  
  输入是 bundle、bundle label、GNN 输出的节点 logits；输出是训练好的节点分类器，以及更纯净的 bundle。

- **为什么这样设计**  
  bundle 标签只代表“多数类”，并不保证 bundle 内每个节点都属于这个类，所以不能粗暴地把同一个标签强贴给 bundle 内所有节点。

- **关键公式 / 机制**

  先用 GNN 输出每个节点的 logits：

  $$
  \{z_i\}_{i=1}^n=g_\theta(\{x_i\}_{i=1}^n,E), \qquad p_i=\mathrm{softmax}(z_i).
  $$

  然后把 bundle 内所有节点的 logits 做平均，再 softmax 得到 bundle 分布：

  $$
  p(B)=\mathrm{softmax}\left(\frac{1}{|B|}\sum_{i\in B} z_i\right).
  $$

  用 bundle 标签 $\hat y_B$ 定义 entropy-based bundle loss：

  $$
  L_{BE}=\mathrm{CE}(p(B),\hat y_B).
  $$

  这不是说“每个节点都必须是 $\hat y_B$”，而是说“整个 bundle 的平均预测应像 $\hat y_B$”。

  作者还加了一个 ranking loss，要求 bundle 标签在 bundle 分布里排第一：

  $$
  L_R=-\min\Big(\log p(B)_{\hat y_B}-\log \max_i p(B)_i,\;0\Big).
  $$

  最终训练目标是：

  $$
  L=L_{BE}+L_R.
  $$

  训练过程中再做 refinement，删除对 bundle 标签最不自信的节点：

  $$
  B \leftarrow \{\, i \mid i\in B \land p_{i,\hat y_B} > \min_{j\in B} p_{j,\hat y_B}\,\}.
  $$

  这表示每次把 bundle 中“最不像 $\hat y_B$”的那个节点踢掉。

- **我的理解**  
  这篇论文最核心的设计其实在这里：**bundle label 不是 node label**，所以作者发明了 bundle-level loss，让模型“整体上像这个类”，而不是强迫每个点都像这个类。

### 3. 训练与推理

- **训练流程**
  1. 采样 $n_S$ 个 bundle；
  2. 用 LLM 给每个 bundle 打标签；
  3. 用 $L_{BE}+L_R$ 训练 GNN；
  4. 在训练中后期做 bundle refinement，再继续训练。  

  默认设置下，bundle size $n_B=5$、bundle 数 $n_S=100$、训练 500 epoch，并在 300 和 400 epoch 做 refinement。

- **推理阶段**  
  最终做节点分类的不是 LLM，而是**训练好的 GNN**。也就是说，这篇方法不是 training-free，而是“无人工标签的目标图训练”。

- **与已有方法的关键区别**  
  它不是 `individual query + individual supervision`，而是  
  **`bundle query + bundle supervision + dynamic refinement`**。

## 创新点

1. 把 LLM 在 TAG 上的监督粒度从 **node-level** 改成 **bundle-level**，使 query 更稳。
2. 设计了对 outlier 更鲁棒的 **bundle supervision**，并从理论上分析其对离群点的容忍性与优化收敛性质。
3. 提出 **dynamic bundle refinement**，让 GNN 反过来帮助净化 bundle 中的噪声节点。

## 实验与结论

### 实验设置

- **数据集**  
  共 10 个 TAG 数据集：Cora、CiteSeer、WikiCS、History、Children、Sportsfit、Cornell、Texas、Wisconsin、Washington。前 6 个偏高同配，后 4 个偏低同配 / 异配。

- **任务**  
  zero-shot inference / 节点分类。

- **baseline**  
  共 15 个，包括 SBERT、RoBERTa、Text-Embedding-3-Large、LLM2Vec、GPT-3.5-turbo、GPT-4o、DGI、GraphMAE、OFA、GOFA、UniGLM、ZeroG、GraphGPT、LLaGA、LLM-BP。

- **指标**  
  Accuracy。

### 主要结果

- DENSE 在 **10/10 个数据集上都是最优**。例如 Cora 75.09、CiteSeer 72.37、WikiCS 71.03、Texas 92.51。
- 相比强基线 LLM-BP，DENSE 在多个数据集上有明显提升，尤其在 Texas、Wisconsin、Washington 这类网页图上优势更大。
- 换不同 LLM 也能工作；GPT-4o 和 Gemini-2.5-flash 整体更强，但较小或较旧模型也能保持不错表现。

### 分析实验

- **消融实验**  
  去掉 proximity sampling、bundle query、$L_{BE}$、$L_R$、bundle refinement，性能都会下降；用 individual supervision 替换 bundle supervision 也会掉点。

- **可视化 / 超参分析**  
  Figure 3 展示了不同 bundle size、bundle 数量，以及 individual query 与 bundle query 的对比结果。

- **额外发现**
  - $n_B=5$ 效果较好；
  - bundle 数越多通常越好，但会带来更多 LLM query 成本；
  - 奇数 bundle size 往往优于偶数，因为不容易出现平票。

## 不足与问题

1. **LLM 并没有真正理解图结构。**  
   它看到的仍然是 bundle 文本，图结构主要体现在前面的采样阶段。

2. **不是 training-free，也不是一次训练后可直接迁移。**  
   每到一个新数据集，仍需重新构造 bundle、重新 query LLM、重新训练 GNN。

3. **依赖文本质量。**  
   如果节点文本很短、很脏、很弱，bundle query 的优势可能不明显；当节点属性难以被 LLM 理解时，该方法不直接适用。

4. **缺少更强的非 LLM bundle baseline。**  
   论文证明了 DENSE 很强，但没有彻底回答“这一步是否一定要用强 LLM”。

## 和我当前研究的关系

- **可借鉴点**  
  不要默认监督粒度必须是 node-level；当单点文本太噪时，可以考虑 group-level / subgraph-level / bundle-level supervision。

- **可复用模块**  
  proximity-based grouping、group-level pseudo-labeling、dynamic filtering / refinement。

- **能否作为 baseline**  
  如果我的任务也是“文本属性图 + 零/弱监督 + LLM 辅助”，它非常适合作为 baseline；如果我的任务更强调真正的结构推理或跨图迁移，它更适合作为启发而非直接对比对象。

- **对我最有启发的一点**  
  **好 idea 不一定来自更大的模型，而可能来自“把问题改写到更稳定的监督粒度上”。**

## 最终总结

- **值不值得重点跟进**  
  值得。因为它不是简单堆模型，而是给出了一个很清楚的方法论：当 individual pseudo-label 太噪时，可以通过 **局部分组 + 更粗粒度监督** 来稳定 zero-shot 图学习。

- **一句话评价**  
  这篇论文真正有启发的地方，不是“LLM 学会了图推理”，而是 **把 LLM 从逐点分类器改造成了局部弱监督老师**。