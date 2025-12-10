条件消息传递代表了GNN从无条件聚合向条件驱动推理的演进

本文提出一种新的图基础模型思路：只在少量知识图谱上做“归纳推理”预训练，就能在完全不微调的情况下，直接迁移到各种普通图上的节点/图/链路任务。

一、方法

1. SCR 框架：用“任务专用 KG”统一所有图任务
•	核心骨干：CMP（Conditional Message Passing）式 KG 归纳推理模型，只依赖 query (eq,rq)(e_q, r_q)(eq,rq) 条件化地计算实体表示。
•	节点分类：
        为每个类别建 label 节点，引入关系 node is_attributed_with label；
        预测节点类别 = 做 (node, is_attributed_with, ?) 的 tail prediction。
•	图分类：
      为每个图建 super-graph 节点，节点连 node belongs_to super_graph；
      图标签仍用 label 节点 + super_graph is_attributed_with label；
      预测图类别 = 对 super-graph 做 tail prediction。
•	KG 链路预测：
直接是 (head, relation, tail) 的标准归纳推理。
→ 三类任务统一成“给定 q=(eq,rq)q=(e_q, r_q)q=(eq,rq)，预测 tail 实体”的 KG 归纳推理。

2. SCMP：在不破坏归纳性的前提下注入语义
•	语义统一（SVD Feature Unifier）：
对任意来源的节点特征（BERT/MPNet/ontology/all-ones 等），用截断 SVD + LayerNorm 映射到统一维度的语义空间 X~\tilde XX~，解决“不同图语义空间不一致”的问题。
•	语义感知初始化 INIT2：
对每个 query 实体 eqe_qeq，在 X~\tilde XX~ 中选取一小撮语义最近邻 SqS_qSq（排除结构直接邻居）；
初始化：
	query 节点：向量 rqr_qrq；
	语义邻居：共享语义向量 vav_ava；
	其他节点：0；
保持 query 节点仍然“可区分”，不破坏 CMP 的理论假设，同时让语义近邻提前参与消息传递。
•	全局语义分支 Hg：
用同一骨干在图上对 X~\tilde XX~ 做一次无 query 的消息传递，得到全局语义表示 HgH_gHg；
最终实体表示 Hfinal=Hstructural+MLP(Hg)H_{\text{final}} = H_{\text{structural}} + \text{MLP}(H_g)Hfinal=Hstructural+MLP(Hg)，融合结构与语义。

3. 训练与推理流程
•	预训练：
仅在 3 个常用 KG（WN18RR / FB15k-237 / CodexM）上做 inductive KG reasoning 训练；
在不同 batch 中轮换使用不同语义特征（文本 / ontology / all-ones），增强跨语义模态鲁棒性。
•	下游 zero-shot 推理：
对任意新图任务，先构造任务专用 KG（加 label / super-graph 节点和关系）；
用 SVD 统一其节点特征后，直接喂给预训练好的 SCR；
全程不微调，实现在多种一般图任务上的 zero-shot 推理。

二、创新点
1.	把 KG 归纳推理正式当成 Graph Foundation Model 的预训练目标
只在少量 KG 上预训练，就能 zero-shot 迁移到大量一般图的 link/node/graph 任务上。
2.	“任务专用 KG”统一节点/图/链路任务
通过 label 节点、super-graph 节点和专用关系，把不同下游任务都转成同一种 tail prediction 格式；
模型结构统一、推理接口统一，天然适配 unseen entity / relation / label。
3.	SCMP 解决语义孤岛问题
SVD 统一语义空间 + INIT2 的局部语义注入 + 全局语义分支 Hg；
在不破坏 CMP 理论可分辨性的前提下，引入语义特征并显著提升性能；
对不同文本 encoder / 甚至非文本特征都具有较强鲁棒性。
4.	系统性 zero-shot 实验验证
在 24 个 inductive KG 数据集上，对比 ULTRA、ProLINK 及监督 SOTA；
在多种节点/图分类数据集上，对比 MLP / GNN / 各类 graph prompt / 其他 GFM；
多数情况下 zero-shot SCR 超过现有 zero-shot GFM，某些任务接近甚至优于 fully-supervised GNN/GIN。
________________________________________
三、不足与潜在局限
1.	预训练数据有限
目前只用少量静态常识 KG，规模和多样性都有限；
关于“扩规模后的 scaling law”和多源 KG 组合效果还缺乏系统研究。
2.	任务专用 KG 构造与语义预处理开销
需要为每个新任务构建专用 KG，并做 SVD + 语义邻居搜索等预处理；
在超大图、动态图或在线场景中，工程成本和效率需要进一步优化。
3.	任务类型仍相对集中
主要验证在静态图上的 link/node/graph 分类任务；
对时序图、推荐/交互图、更复杂的推理及真实工业场景，泛化能力仍待进一步验证。
________________________________________
总结
这篇工作把“在 KG 上做归纳推理”直接升级成一种 Graph Foundation Model 预训练范式，通过任务专用 KG + SCMP，把结构泛化能力和平滑的语义表示结合起来，做到：只在少量 KG 预训练，就能 zero-shot 做各种一般图任务。优势明显，但在预训练数据规模、任务广度和大规模应用上的工程实践，还有不小的拓展空间。
