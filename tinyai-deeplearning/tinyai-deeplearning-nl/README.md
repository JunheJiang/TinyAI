# TinyAI Nested Learning 嵌入学习模块 (tinyai-deeplearning-nl)

## 模块概述

`tinyai-deeplearning-nl` 是 TinyAI 深度学习框架的嵌入学习模块，实现了 Google 在 NeurIPS 2025 发表的论文《Nested Learning: The Illusion of Deep Learning Architectures》中提出的革命性学习范式。

嵌入学习将传统深度学习模型重新定义为一组相互嵌套、多层级的优化问题系统，每个优化问题都有自己的上下文流和更新频率。这种范式能够有效缓解甚至完全避免持续学习中的"灾难性遗忘"问题。

## 嵌入学习核心原理

### 理论基础

嵌入学习突破了传统的架构-优化分离模式，将模型架构和优化算法统一为同一概念的不同层级表现：

- **多层级优化系统**：模型由多个嵌套的优化问题组成，每个层级有独立的上下文流
- **关联记忆模型**：反向传播过程被建模为关联记忆，学习将数据点映射到局部误差值
- **连续记忆系统（CMS）**：将记忆视为频谱，由多个不同更新频率的模块组成

### 与传统深度学习的区别

| 维度 | 传统深度学习 | 嵌入学习 |
|------|------------|---------|
| 架构视角 | 静态层堆叠 | 动态嵌套优化问题 |
| 优化视角 | 单一优化过程 | 多层级协同优化 |
| 记忆管理 | 二元（短期/长期） | 连续频谱 |
| 更新策略 | 统一更新频率 | 多时间尺度更新 |
| 持续学习 | 易产生灾难性遗忘 | 内在支持持续学习 |

## 核心组件

### 核心概念层（core）

#### NestedOptimizationLevel（嵌套优化层级）
表示嵌入学习中的单个优化层级：
- 层级索引和更新频率管理
- 上下文流传播
- 参数更新和梯度管理
- 父子层级关联

```java
// 创建优化层级
NestedOptimizationLevel level = new NestedOptimizationLevel(
    0,      // 层级索引
    1.0f,   // 更新频率（每步更新）
    0.001f  // 学习率
);

// 判断是否应该更新
if (level.shouldUpdate(currentStep)) {
    level.updateParameters(gradients);
}
```

#### ContextFlow（上下文流）
管理嵌套优化层级之间的信息流动：
- 上下文数据传播
- 上下文压缩
- 多流合并

```java
// 创建上下文流
ContextFlow contextFlow = new ContextFlow(
    contextData,                    // 上下文数据
    FlowDirection.BIDIRECTIONAL,    // 双向流动
    0.8f                            // 压缩率
);

// 流动上下文
Variable processedContext = contextFlow.flow(inputContext);
```

#### AssociativeMemory（关联记忆）
实现关联记忆模型，将输入映射到输出：
- 基于键值对的存储
- 基于惊异度的记忆优先级
- 记忆检索和修剪

```java
// 创建关联记忆
AssociativeMemory memory = new AssociativeMemory(
    100,   // 记忆容量
    0.5f   // 惊异度阈值
);

// 存储记忆
memory.store(keyVariable, valueVariable);

// 检索记忆
Variable retrieved = memory.retrieve(queryKey);

// 计算惊异度
float surprise = memory.computeSurprise(inputData);
```

## 技术依赖

本模块依赖以下 TinyAI 核心模块：

- `tinyai-deeplearning-ndarr` - 多维数组基础库，提供张量计算
- `tinyai-deeplearning-func` - 自动微分引擎，提供梯度计算支持
- `tinyai-deeplearning-nnet` - 神经网络层，提供网络构建组件
- `tinyai-deeplearning-ml` - 机器学习模块，提供训练和优化支持

外部依赖：
- `jfreechart` - 图表可视化库（可选）
- `junit` - 单元测试框架

## 快速开始

### 构建模块

```bash
cd /Users/yefei.yf/Qoder/TinyAI
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
mvn compile -pl tinyai-deeplearning-nl -am
```

### 运行测试

```bash
mvn test -pl tinyai-deeplearning-nl
```

## 版本信息

- **当前版本**: 1.0-SNAPSHOT
- **Java 版本**: 17+
- **构建工具**: Maven 3.6+
- **理论基础**: Google NeurIPS 2025 论文

## 参考资料

1. **原始论文**：Ali Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures", NeurIPS 2025
2. **Google Research博客**：[Introducing Nested Learning: A new ML paradigm for continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
3. **TinyAI项目**：[https://github.com/leavesfly/TinyAI](https://github.com/leavesfly/TinyAI)

## 相关模块

- [`tinyai-deeplearning-ml`](../tinyai-deeplearning-ml/README.md) - 机器学习核心系统
- [`tinyai-deeplearning-nnet`](../tinyai-deeplearning-nnet/README.md) - 神经网络层模块
- [`tinyai-deeplearning-func`](../tinyai-deeplearning-func/README.md) - 自动微分引擎
- [`tinyai-deeplearning-ndarr`](../tinyai-deeplearning-ndarr/README.md) - 多维数组基础库

---

**TinyAI Nested Learning 模块** - 探索深度学习的嵌套本质，实现持续学习的新范式 🚀
