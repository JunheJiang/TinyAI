# TinyAI Neural Network V2 模块

## 概述

V2 版本是 TinyAI 神经网络模块的全新实现，采用类似 PyTorch 的设计理念，提供更强大的参数管理、延迟初始化、模式切换等高级特性。

## 主要特性

### 1. 统一的参数注册机制
- `registerParameter()` 统一注册可训练参数
- `registerBuffer()` 注册非可训练张量（如 BatchNorm 的统计量）
- `namedParameters()` 自动生成分层命名路径

### 2. 延迟初始化支持
- `LazyModule` 基类支持根据输入动态推断参数形状
- `LazyLinear`、`LazyConv2d` 等层无需预先指定输入维度

### 3. 训练/推理模式切换
- `train()` 和 `eval()` 方法控制全局模式
- Dropout、BatchNorm 等层自动适配不同模式

### 4. 灵活的初始化策略
- `Initializer` 接口和丰富的内置初始化器
- `resetParameters()` 统一的参数初始化接口
- 支持外部自定义初始化策略

### 5. 完整的状态管理
- `stateDict()` 导出完整模型状态
- `loadStateDict()` 加载预训练权重
- 支持部分状态加载和模型迁移

## 目录结构

```
v2/
├── core/              # 核心抽象
│   ├── Module.java    # 模块基类（继承Function）
│   ├── Parameter.java # 增强的参数类
│   └── LazyModule.java# 延迟初始化基类
│
├── init/              # 初始化器
│   ├── Initializer.java
│   ├── Initializers.java
│   ├── ZerosInitializer.java
│   ├── KaimingInitializer.java
│   └── XavierInitializer.java
│
├── layer/             # 层实现
│   ├── dnn/          # 全连接层
│   │   ├── Linear.java
│   │   ├── LazyLinear.java
│   │   └── Dropout.java
│   ├── activation/   # 激活函数
│   │   ├── ReLU.java
│   │   ├── Sigmoid.java
│   │   ├── Tanh.java
│   │   └── SoftMax.java
│   ├── norm/         # 归一化层
│   │   ├── LayerNorm.java
│   │   └── BatchNorm1d.java  # ✨ 新增
│   ├── conv/         # 卷积层（待实现）
│   ├── rnn/          # 循环层（待实现）
│   └── transformer/  # Transformer组件（待实现）
│
├── container/         # 容器模块
│   ├── Sequential.java
│   ├── ModuleList.java
│   └── ModuleDict.java
│
└── utils/             # 工具类
    └── StateDict.java
```

## 快速开始

### 标准线性层

```java
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

// 创建线性层
Module linear = new Linear("fc", 128, 64, true);

// 前向传播
Variable output = linear.forward(input);

// 访问参数
Parameter weight = linear.getParameter("weight");
Parameter bias = linear.getParameter("bias");
```

### 延迟初始化

```java
import io.leavesfly.tinyai.nnet.v2.layer.dnn.LazyLinear;

// 无需指定输入维度
Module lazy = new LazyLinear("fc", 64, true);

// 首次前向传播时自动推断并初始化
Variable output = lazy.forward(input);  // 根据input.shape推断
```

### 模式切换

```java
// 训练模式
model.train();
output = model.forward(input);  // Dropout启用

// 推理模式
model.eval();
output = model.forward(input);  // Dropout禁用
```

### 自定义初始化

```java
import io.leavesfly.tinyai.nnet.v2.init.Initializers;

// 方式一：在resetParameters中使用
@Override
public void resetParameters() {
    Initializers.kaimingUniform(weight.data(), 0, "fan_in", "relu");
    Initializers.zeros(bias.data());
}

// 方式二：外部统一初始化
model.apply(module -> {
    if (module instanceof Linear) {
        Initializers.xavierNormal(module.getParameter("weight").data());
    }
});
```

### BatchNorm1d 使用

```java
import io.leavesfly.tinyai.nnet.v2.layer.norm.BatchNorm1d;

// 创建BatchNorm层
BatchNorm1d bn = new BatchNorm1d("bn1", 64);

// 训练模式：使用批次统计量
bn.train();
Variable output = bn.forward(input);  // 自动更新running stats

// 推理模式：使用固定统计量
bn.eval();
Variable output = bn.forward(input);  // 使用running stats

// 访问统计量
NdArray runningMean = bn.getRunningMean();
NdArray runningVar = bn.getRunningVar();
```

## V1 vs V2 对比

| 特性 | V1 (LayerAble) | V2 (Module) |
|------|---------------|-------------|
| 继承关系 | LayerAble → Function | Module → Function |
| 参数管理 | 手动Map管理 | registerParameter/Buffer |
| 命名路径 | 手动拼接 | 自动分层路径 |
| 模式切换 | ❌ 不支持 | ✅ train()/eval() |
| 延迟初始化 | ❌ 不支持 | ✅ LazyModule |
| 状态序列化 | 部分支持 | ✅ stateDict/loadStateDict |
| 自动微分 | ✅ 支持 | ✅ 支持（继承Function）|

## 兼容性

V2 与 V1 完全隔离，互不影响：
- V1 代码保持稳定，不做任何修改
- V2 使用独立的包命名空间 `io.leavesfly.tinyai.nnet.v2`
- 两者可在同一项目中共存

## 文档目录

- [API参考](api-reference.md) - 详细的API文档
- [迁移指南](migration-guide.md) - V1到V2迁移步骤
- [设计原则](design-principles.md) - V2设计理念说明

## 开发状态

### 核心功能（✅ 已完成）
- [x] 阶段一：V2基础架构搭建
- [x] 阶段二：V2高级特性实现（延迟初始化、容器模块）
- [x] BatchNorm1d 归一化层实现
- [x] 测试工具类（AssertHelper、GradientChecker、TestDataGenerator）

### 测试覆盖（🚧 进行中）
- [x] BatchNorm1d 完整单元测试（11个测试用例）
- [ ] Module核心组件测试
- [ ] Linear层功能测试
- [ ] 激活函数测试
- [ ] 初始化器测试
- [ ] 集成测试

### 高级层（📅 计划中）
- [ ] 阶段三：RNN层（LSTM、GRU、SimpleRNN）
- [ ] Transformer组件（MultiHeadAttention、EncoderLayer）
- [ ] 卷积层（Conv2d、LazyConv2d、Pooling）

## 许可证

与 TinyAI 主项目保持一致
