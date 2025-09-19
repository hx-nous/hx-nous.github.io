# 9.15组会汇报本篇论文
本论文主要是讲了一个带有多种自约束的多通道解耦图神经网络，目的是为了解决在使用图神经网络过程中的标签不足问题，相较于以往的多通道图神经网络，它有三个通道：特征通道，拓扑通道，潜在通道。它将节点属性分割为共享部分和互补部分，并且在这两个部分上添加了三种自约束：一致性约束，互补约束，对齐约束。还加入了注意力机制对三个通道的输出进行融合，然后将这三个输出与融合进行对齐约束，这样让多通道gnn不会偏向一致性或者互补性。

### 汇报ppt
[MD_GraphNet.pptx](https://github.com/user-attachments/files/22422808/MD_GraphNet.pptx)

### 原文
[Multi-Channel Disentangled Graph Neural Networks with Different Types of Self-constraints.pdf](https://github.com/user-attachments/files/22422809/Multi-Channel.Disentangled.Graph.Neural.Networks.with.Different.Types.of.Self-constraints.pdf)

