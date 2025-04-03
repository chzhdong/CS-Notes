## 论文概述

论文标题：**Efficient Memory Management for Large Language  Model Serving with PagedAttention**

核心工作：对于大模型推理，一般通过批次化请求提升吞吐量。由于每个请求的 KV Cache 的空间是动态变化的，导致其内存管理由于片段和重复无法充分利用。为了解决该问题，参照操作系统的虚拟内存和页技术，提出了 PagedAttention实现**KV Cache近乎于无的浪费**以及**请求中和请求间 KV Cache 的灵活共享**，减少内存的使用。

## 存在的问题

对于已有的工作，如 Orca 和 Transformer 来说，其对于 KV Cache 的管理存在以下挑战：

- 系统对每个请求分配最大限度的连续内存空间，导致内部片段。
- 即使输出长度已知，其他请求无法使用当前请求空闲的空间。
- 由于对每个请求预分配的空间不同，导致存在大量的外部片段。
- 现有的系统无法充分利用内存共享，如 Parallel Sampling 和 Beam Search。

为了解决上述挑战，PagedAttention 通过参考操作系统，将 KV Cache划分为 Block 进行统一管理，即一个 Block 管理固定数量的 Tokens，这就防止了存在外部片段，并且使得请求中和请求间能够实现内存的共享。

