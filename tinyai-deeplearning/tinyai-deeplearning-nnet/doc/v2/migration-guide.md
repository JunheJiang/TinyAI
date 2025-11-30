# V1 到 V2 迁移指南

## 概述

本指南帮助您将现有的TinyAI V1代码迁移到V2版本。V2提供了更强大的功能，同时保持与V1的隔离，您可以选择渐进式迁移。

## 快速对比

### 包导入变化

```java
// V1导入
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.block.SequentialBlock;

// V2导入
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;
```

### 基本用法对比

#### 创建线性层

```java
// V1方式
LinearLayer fc1 = new LinearLayer("fc1", 128, 64, true);

// V2方式
Linear fc1 = new Linear("fc1", 128, 64, true);
```

#### 构建Sequential模型

```java
// V1方式
Block model = new SequentialBlock("model");
model.addLayer(new LinearLayer("fc1", 128, 64, true));
model.addLayer(new Relu("relu1"));
model.addLayer(new LinearLayer("fc2", 64, 10, true));

// V2方式 - 链式调用
Module model = new Sequential("model")
    .add(new Linear("fc1", 128, 64))
    .add(new ReLU())
    .add(new Linear("fc2", 64, 10));
```

#### 前向传播

```java
// V1方式
Variable output = model.layerForward(input);

// V2方式
Variable output = model.forward(input);
```

## 详细迁移步骤

### 步骤1：更新导入语句

将所有V1的导入替换为V2：

```java
// 查找替换规则
io.leavesfly.tinyai.nnet.Layer → io.leavesfly.tinyai.nnet.v2.core.Module
io.leavesfly.tinyai.nnet.Block → io.leavesfly.tinyai.nnet.v2.core.Module
io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer → io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear
io.leavesfly.tinyai.nnet.block.SequentialBlock → io.leavesfly.tinyai.nnet.v2.container.Sequential
```

### 步骤2：修改类继承

**V1自定义层：**
```java
public class MyLayer extends Layer {
    private Parameter weight;
    
    public MyLayer(String name, int inputSize, int outputSize) {
        super(name);
        this.inputShape = Shape.of(inputSize);
        this.outputShape = Shape.of(outputSize);
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            NdArray initWeight = NdArray.likeRandomN(Shape.of(outputShape.getColumn(), inputShape.getColumn()));
            weight = new Parameter(initWeight);
            addParam(weight.getName(), weight);
            alreadyInit = true;
        }
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        // ...
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // ...
    }
}
```

**V2自定义层：**
```java
public class MyLayer extends Module {
    private Parameter weight;
    private final int inputSize;
    private final int outputSize;
    
    public MyLayer(String name, int inputSize, int outputSize) {
        super(name);
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        
        // 创建参数
        NdArray weightData = NdArray.of(Shape.of(outputSize, inputSize));
        this.weight = registerParameter("weight", new Parameter(weightData));
        
        // 初始化参数
        init();
    }
    
    @Override
    public void resetParameters() {
        // 使用Initializers工具类
        Initializers.kaimingUniform(weight.data());
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        // 使用Variable操作保持计算图
        return x.matMul(weight.transpose());
    }
    
    // 不需要实现backward - 依赖自动微分
}
```

### 步骤3：参数管理迁移

**V1参数注册：**
```java
addParam(weight.getName(), weight);
Parameter w = getParamBy("weight");
```

**V2参数注册：**
```java
this.weight = registerParameter("weight", new Parameter(weightData));
Parameter w = getParameter("weight");
```

**V2参数遍历：**
```java
// V1
Map<String, Parameter> params = model.getParams();

// V2 - 更强大的分层路径
Map<String, Parameter> params = model.namedParameters();
// 返回：{"encoder.layer1.weight": param1, "encoder.layer1.bias": param2, ...}
```

### 步骤4：初始化策略迁移

**V1硬编码初始化：**
```java
@Override
public void init() {
    if (!alreadyInit) {
        NdArray initWeight = NdArray.likeRandomN(Shape.of(...))
            .mulNum(Math.sqrt((double) 1 / inputShape.getColumn()));
        wParam = new Parameter(initWeight);
        addParam(wParam.getName(), wParam);
        alreadyInit = true;
    }
}
```

**V2使用Initializers：**
```java
@Override
public void resetParameters() {
    // 方式1：使用静态方法
    Initializers.xavierUniform(weight.data());
    Initializers.zeros(bias.data());
    
    // 方式2：使用初始化器对象
    new KaimingUniformInitializer().initialize(weight.data());
}
```

### 步骤5：模式切换支持

V2新增train()/eval()模式切换：

```java
// 训练阶段
model.train();
for (Batch batch : trainingData) {
    Variable output = model.forward(batch.input);
    // Dropout启用
}

// 验证阶段
model.eval();
for (Batch batch : validationData) {
    Variable output = model.forward(batch.input);
    // Dropout禁用
}
```

如果您的V1层需要模式感知：

```java
// V2实现
@Override
public Variable forward(Variable... inputs) {
    if (_training) {
        // 训练模式逻辑
    } else {
        // 推理模式逻辑
    }
}
```

### 步骤6：状态序列化迁移

**V1（如果实现）：**
```java
// 手动保存参数
Map<String, Parameter> params = model.getParams();
// 保存逻辑...
```

**V2标准化方法：**
```java
// 保存模型
Map<String, NdArray> state = model.stateDict();
// 使用您的序列化工具保存state

// 加载模型
model.loadStateDict(state, true);
```

## 高级迁移场景

### 场景1：延迟初始化

V1无法实现，V2提供LazyModule：

```java
// V2新特性
LazyLinear layer = new LazyLinear("fc", 64, true);
// 无需指定输入维度

Variable output = layer.forward(input);
// 首次调用时自动推断并初始化
```

### 场景2：灵活的初始化策略

```java
// V2外部统一初始化
model.apply(module -> {
    if (module instanceof Linear) {
        Linear linear = (Linear) module;
        Initializers.xavierNormal(linear.getWeight().data(), 1.0f);
        if (linear.getBias() != null) {
            Initializers.constant(linear.getBias().data(), 0.01f);
        }
    }
});
```

### 场景3：模块化组合

**V1需要自定义Block：**
```java
public class Encoder extends Block {
    public Encoder(...) {
        addLayer(new LinearLayer(...));
        addLayer(new Relu());
        // ...
    }
}
```

**V2使用Sequential：**
```java
public class Encoder extends Module {
    private Sequential layers;
    
    public Encoder(String name, int inputSize, int hiddenSize) {
        super(name);
        this.layers = new Sequential("layers")
            .add(new Linear("fc1", inputSize, hiddenSize))
            .add(new ReLU())
            .add(new Linear("fc2", hiddenSize, hiddenSize));
        registerModule("layers", this.layers);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        return layers.forward(inputs);
    }
}
```

## 渐进式迁移策略

### 策略A：新旧共存

V1和V2完全隔离，可以在同一项目中共存：

```java
// 旧模块继续使用V1
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
Block oldModule = new SequentialBlock("old");
oldModule.addLayer(new LinearLayer("fc", 128, 64, true));

// 新模块使用V2
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
Module newModule = new Sequential("new")
    .add(new Linear("fc", 64, 10));

// 在前向传播中组合
Variable h = oldModule.layerForward(input);
Variable output = newModule.forward(h);
```

### 策略B：逐步替换

1. **第一阶段**：新功能使用V2
2. **第二阶段**：迁移核心模型
3. **第三阶段**：迁移辅助模块
4. **第四阶段**：删除V1依赖

### 策略C：一次性迁移

适用于小型项目：
1. 更新所有导入语句
2. 修改类继承和方法调用
3. 利用V2新特性优化代码
4. 测试验证

## 常见问题

### Q1: V2性能如何？

A: V2与V1使用相同的底层NdArray和自动微分引擎，性能基本一致。额外的抽象层开销可以忽略不计。

### Q2: 可以混用V1和V2吗？

A: 可以。V1和V2完全隔离，可以在同一项目中共存。但建议逐步统一到V2以获得更好的维护性。

### Q3: V2是否向后兼容V1的模型文件？

A: V2提供了`loadStateDict()`方法，可以加载V1保存的参数（需要适配参数命名）。

### Q4: 必须迁移到V2吗？

A: 不是必须的。V1将继续维护。但V2提供了更多现代特性，推荐新项目使用V2。

### Q5: 如何验证迁移正确性？

A: 
1. 对比V1和V2模型的前向传播输出
2. 对比梯度计算结果
3. 对比训练曲线
4. 编写单元测试

## 迁移检查清单

- [ ] 更新导入语句
- [ ] 修改类继承（Layer/Block → Module）
- [ ] 更新参数注册方式（addParam → registerParameter）
- [ ] 更新参数访问方式（getParamBy → getParameter）
- [ ] 修改初始化逻辑（init → resetParameters + Initializers）
- [ ] 更新前向传播方法签名（layerForward → forward）
- [ ] 移除backward实现（依赖自动微分）
- [ ] 添加模式切换支持（可选）
- [ ] 更新模型保存/加载代码（stateDict/loadStateDict）
- [ ] 运行测试验证功能正确性
- [ ] 性能基准测试

## 示例：完整迁移案例

### V1代码

```java
public class SimpleClassifier extends Block {
    public SimpleClassifier(String name, int inputSize, int hiddenSize, int outputSize) {
        super(name);
        addLayer(new LinearLayer("fc1", inputSize, hiddenSize, true));
        addLayer(new Relu("relu1"));
        addLayer(new LinearLayer("fc2", hiddenSize, outputSize, true));
    }
}

// 使用
SimpleClassifier model = new SimpleClassifier("classifier", 784, 128, 10);
Variable output = model.layerForward(input);
```

### V2代码

```java
public class SimpleClassifier extends Module {
    private Sequential layers;
    
    public SimpleClassifier(String name, int inputSize, int hiddenSize, int outputSize) {
        super(name);
        this.layers = new Sequential("layers")
            .add(new Linear("fc1", inputSize, hiddenSize))
            .add(new ReLU())
            .add(new Linear("fc2", hiddenSize, outputSize));
        registerModule("layers", this.layers);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        return layers.forward(inputs);
    }
    
    @Override
    public void resetParameters() {
        // 可选：自定义初始化
        apply(module -> {
            if (module instanceof Linear) {
                Linear linear = (Linear) module;
                Initializers.kaimingUniform(linear.getWeight().data());
            }
        });
    }
}

// 使用
SimpleClassifier model = new SimpleClassifier("classifier", 784, 128, 10);
model.train();  // 设置训练模式
Variable output = model.forward(input);
```

## 获得帮助

- 查阅 [API参考文档](api-reference.md)
- 查看 [实施总结](implementation-summary.md)
- 参考 [示例代码](../examples/)

如有问题，欢迎提交Issue或Pull Request。
