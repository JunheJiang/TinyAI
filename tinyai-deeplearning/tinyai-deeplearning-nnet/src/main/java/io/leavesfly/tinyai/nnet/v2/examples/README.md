# TinyAI V2 模块示例代码

本目录包含了TinyAI深度学习框架V2版本的完整示例代码，展示了如何使用各种神经网络层和模块。

## 📚 示例列表

### 1. 基础使用 (01_BasicUsage.java)
**展示内容:**
- 创建简单的全连接网络
- 使用train()和eval()模式切换
- 访问和管理模型参数
- 查看子模块结构

**适合人群:** 初学者，刚开始使用V2模块

**关键概念:**
- Module基类的使用
- 参数注册和访问
- 训练/推理模式切换

### 2. 延迟初始化 (02_LazyInitialization.java)
**展示内容:**
- 使用LazyLinear自动推断输入维度
- 使用LazyConv2d自动推断输入通道数
- 延迟初始化的优势和注意事项

**适合人群:** 需要灵活模型定义的开发者

**关键概念:**
- LazyModule模式
- 参数的延迟创建
- 首次forward时的初始化

### 3. CNN分类器 (03_CNNClassifier.java)
**展示内容:**
- 构建LeNet-5风格的卷积神经网络
- 使用Conv2d、MaxPool2d等卷积层
- 处理图像数据的形状变换
- 实现完整的分类器

**适合人群:** 计算机视觉任务开发者

**关键概念:**
- 卷积层和池化层
- 特征提取和分类
- 4D张量的展平操作

**模型结构:**
```
输入 (28x28) 
  ↓
Conv2d (6通道, 5x5) → ReLU → MaxPool (2x2)
  ↓
Conv2d (16通道, 5x5) → ReLU → MaxPool (2x2)
  ↓
展平 → Linear (120) → ReLU → Dropout
  ↓
Linear (84) → ReLU
  ↓
Linear (10输出)
```

### 4. RNN序列建模 (04_RNNSequenceModeling.java)
**展示内容:**
- 使用LSTM、GRU、SimpleRNN处理序列数据
- 管理RNN的隐藏状态
- 构建序列分类模型
- 比较不同RNN变体的参数量

**适合人群:** 自然语言处理和时序任务开发者

**关键概念:**
- 循环神经网络
- 隐藏状态管理
- 序列处理

**模型对比:**
- **LSTM**: 3个门（输入门、遗忘门、输出门）+ 细胞状态，参数最多
- **GRU**: 2个门（重置门、更新门），参数适中
- **SimpleRNN**: 无门控机制，参数最少

### 5. 模型序列化 (05_ModelSerialization.java)
**展示内容:**
- 使用stateDict保存模型参数
- 从stateDict加载模型参数
- 验证保存和加载的正确性
- 模型迁移和复用

**适合人群:** 需要保存和加载模型的开发者

**关键概念:**
- 参数序列化
- 模型检查点
- 迁移学习

**使用场景:**
- 训练后保存最佳模型
- 加载预训练模型推理
- 断点续训
- 模型共享

### 6. Transformer模型 (06_TransformerModel.java)
**展示内容:**
- 使用多头注意力机制
- 使用位置编码
- 构建Transformer编码器和解码器
- 理解自注意力机制

**适合人群:** 高级NLP任务开发者

**关键概念:**
- Self-Attention机制
- Multi-Head Attention
- 位置编码
- 编码器-解码器架构

**架构优势:**
- 并行计算能力
- 长距离依赖捕获
- 可解释的注意力权重
- 良好的可扩展性

## 🚀 快速开始

### 运行示例

每个示例都是独立的Java类，包含main方法，可以直接运行：

```bash
# 进入项目目录
cd /Users/yefei.yf/Qoder/TinyAI/tinyai-deeplearning-nnet

# 编译项目（如果需要）
mvn compile

# 运行示例（以BasicUsage为例）
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.nnet.v2.examples.BasicUsage"
```

或者在IDE中直接运行对应的main方法。

### 学习路径

建议按以下顺序学习示例：

```
01_BasicUsage.java
    ↓
02_LazyInitialization.java
    ↓
03_CNNClassifier.java (图像任务方向)
    ↓
04_RNNSequenceModeling.java (序列任务方向)
    ↓
05_ModelSerialization.java
    ↓
06_TransformerModel.java (高级)
```

## 📖 代码说明

### 通用模式

所有示例都遵循以下模式：

```java
// 1. 定义模型类（继承Module）
static class MyModel extends Module {
    private final Layer1 layer1;
    private final Layer2 layer2;
    
    public MyModel(String name) {
        super(name);
        
        // 创建子模块
        layer1 = new Layer1(...);
        layer2 = new Layer2(...);
        
        // 注册子模块
        registerModule("layer1", layer1);
        registerModule("layer2", layer2);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        x = layer1.forward(x);
        x = layer2.forward(x);
        return x;
    }
}

// 2. 在main中使用
public static void main(String[] args) {
    // 创建模型
    MyModel model = new MyModel("my_model");
    
    // 设置模式
    model.train(); // 或 model.eval()
    
    // 前向传播
    Variable output = model.forward(input);
}
```

### 核心API

#### Module基类
- `registerModule(name, module)` - 注册子模块
- `registerParameter(name, param)` - 注册可训练参数
- `registerBuffer(name, buffer)` - 注册非可训练状态
- `train()` / `eval()` - 切换训练/推理模式
- `parameters()` - 获取所有参数
- `modules()` - 获取所有子模块
- `stateDict()` - 导出参数字典
- `loadStateDict(dict)` - 加载参数字典

#### 层的使用
```java
// 全连接层
Linear fc = new Linear("fc", inputSize, outputSize, useBias);

// 延迟初始化全连接层
LazyLinear lazyFc = new LazyLinear("lazy_fc", outputSize, useBias);

// 卷积层
Conv2d conv = new Conv2d("conv", inChannels, outChannels, 
                         kernelH, kernelW, stride, padding, useBias);

// RNN层
LSTM lstm = new LSTM("lstm", inputSize, hiddenSize, useBias);
GRU gru = new GRU("gru", inputSize, hiddenSize, useBias);
SimpleRNN rnn = new SimpleRNN("rnn", inputSize, hiddenSize, useBias, "tanh");

// Transformer组件
MultiHeadAttention mha = new MultiHeadAttention("mha", dModel, nHead, dropout);
PositionalEncoding posEnc = new PositionalEncoding("pos", dModel, maxLen, dropout);
```

## 🔍 常见问题

### Q1: 为什么需要调用registerModule？
A: registerModule会自动收集子模块的参数，使得parameters()能返回所有可训练参数。这对于优化器和参数保存至关重要。

### Q2: train()和eval()有什么区别？
A: train()模式下，Dropout会随机丢弃神经元，BatchNorm会更新统计量。eval()模式下，这些行为会被禁用，确保推理的确定性。

### Q3: LazyModule什么时候初始化？
A: LazyModule在首次调用forward()时根据输入形状初始化参数。初始化后，参数形状不应改变。

### Q4: 如何保存模型到文件？
A: 示例中的stateDict返回内存中的参数字典。实际应用中，可以将其序列化为JSON、二进制等格式保存到文件。

### Q5: 能否混用V1和V2的层？
A: 不建议。V2层基于新的Module系统，与V1层的接口不兼容。建议完全使用V2层。

## 📊 性能提示

1. **批处理**: 尽可能使用批处理（batch_size > 1）以提高效率
2. **延迟初始化**: 对于不确定输入维度的场景，使用LazyModule可简化代码
3. **推理模式**: 推理时务必调用eval()以禁用Dropout和固定BatchNorm
4. **参数共享**: 可以在不同模块间共享同一个Parameter对象

## 🛠️ 调试技巧

### 检查形状
```java
Variable x = ...;
System.out.println("Shape: " + Arrays.toString(x.getValue().getShape().getShape()));
```

### 检查参数
```java
for (Map.Entry<String, Parameter> entry : model.parameters().entrySet()) {
    System.out.println(entry.getKey() + ": " + 
                      Arrays.toString(entry.getValue().data().getShape().getShape()));
}
```

### 检查梯度
```java
Variable output = model.forward(input);
output.backward(); // 反向传播

for (Map.Entry<String, Parameter> entry : model.parameters().entrySet()) {
    if (entry.getValue().grad() != null) {
        System.out.println(entry.getKey() + " has gradient");
    }
}
```

## 📝 进一步学习

- 查看 `/doc/v2/implementation-summary.md` 了解V2模块的完整设计
- 查看 `/doc/v2/final-completion-report.md` 了解实现细节
- 阅读源代码中的JavaDoc注释
- 参考单元测试了解更多用法

## 🤝 贡献

如果您发现示例中的问题或有改进建议，欢迎提交Issue或Pull Request。

## 📄 许可

本示例代码遵循TinyAI项目的许可协议。

---

**TinyAI团队**  
最后更新: 2025-10-19
