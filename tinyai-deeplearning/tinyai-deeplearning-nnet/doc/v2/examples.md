# TinyAI Neural Network V2 示例代码

## 目录

- [基础示例](#基础示例)
  - [示例1：创建简单的线性层](#示例1创建简单的线性层)
  - [示例2：使用Sequential构建多层网络](#示例2使用sequential构建多层网络)
  - [示例3：延迟初始化](#示例3延迟初始化)
- [高级示例](#高级示例)
  - [示例4：自定义初始化策略](#示例4自定义初始化策略)
  - [示例5：参数访问和遍历](#示例5参数访问和遍历)
  - [示例6：模型保存和加载](#示例6模型保存和加载)
  - [示例7：BatchNorm1d使用](#示例7batchnorm1d使用) ✨

## 基础示例

### 示例1：创建简单的线性层

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

public class LinearExample {
    public static void main(String[] args) {
        // 创建线性层：输入128维，输出64维
        Linear layer = new Linear("fc", 128, 64, true);
        
        // 创建输入数据 (batch_size=32, features=128)
        NdArray inputData = NdArray.randn(Shape.of(32, 128));
        Variable input = new Variable(inputData);
        
        // 前向传播
        Variable output = layer.forward(input);
        System.out.println("输出形状: " + output.getShape());
        // 输出: 输出形状: (32, 64)
        
        // 访问参数
        System.out.println("权重形状: " + layer.getWeight().data().getShape());
        System.out.println("偏置形状: " + layer.getBias().data().getShape());
    }
}
```

### 示例2：使用Sequential构建多层网络

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;

public class SequentialExample {
    public static void main(String[] args) {
        // 构建多层感知机
        Sequential model = new Sequential("mlp")
            .add(new Linear("fc1", 784, 256))
            .add(new ReLU())
            .add(new Dropout("drop1", 0.5f))
            .add(new Linear("fc2", 256, 128))
            .add(new ReLU())
            .add(new Dropout("drop2", 0.5f))
            .add(new Linear("fc3", 128, 10));
        
        // 创建输入 (batch_size=64, features=784)
        NdArray inputData = NdArray.randn(Shape.of(64, 784));
        Variable input = new Variable(inputData);
        
        // 训练模式
        model.train();
        Variable trainOutput = model.forward(input);
        System.out.println("训练输出形状: " + trainOutput.getShape());
        
        // 推理模式（Dropout禁用）
        model.eval();
        Variable evalOutput = model.forward(input);
        System.out.println("推理输出形状: " + evalOutput.getShape());
    }
}
```

### 示例3：延迟初始化

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.LazyLinear;

public class LazyInitExample {
    public static void main(String[] args) {
        // 创建延迟线性层，无需指定输入维度
        LazyLinear layer = new LazyLinear("lazy_fc", 64, true);
        
        System.out.println("初始化前: " + layer);
        // 输出: LazyLinear{..., inFeatures=uninitialized, ...}
        
        // 首次前向传播时自动推断输入维度
        NdArray input1 = NdArray.randn(Shape.of(32, 128));
        Variable output1 = layer.forward(new Variable(input1));
        
        System.out.println("初始化后: " + layer);
        // 输出: LazyLinear{..., inFeatures=128, ...}
        
        // 后续调用复用已初始化的参数
        NdArray input2 = NdArray.randn(Shape.of(16, 128));
        Variable output2 = layer.forward(new Variable(input2));
    }
}
```

## 高级示例

### 示例4：自定义初始化策略

```java
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;

public class CustomInitExample {
    public static void main(String[] args) {
        Sequential model = new Sequential("model")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU())
            .add(new Linear("fc2", 64, 10));
        
        // 方式1：使用apply统一初始化
        model.apply(module -> {
            if (module instanceof Linear) {
                Linear linear = (Linear) module;
                // 使用Xavier正态初始化
                Initializers.xavierNormal(linear.getWeight().data(), 1.0f);
                if (linear.getBias() != null) {
                    // 偏置初始化为0.01
                    Initializers.constant(linear.getBias().data(), 0.01f);
                }
            }
        });
        
        System.out.println("模型初始化完成");
    }
}
```

### 示例5：参数访问和遍历

```java
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import java.util.Map;

public class ParameterAccessExample {
    public static void main(String[] args) {
        Sequential model = new Sequential("model")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU())
            .add(new Linear("fc2", 64, 10));
        
        // 遍历所有参数
        Map<String, Parameter> params = model.namedParameters();
        System.out.println("模型参数列表:");
        for (Map.Entry<String, Parameter> entry : params.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue().data().getShape());
        }
        
        // 输出:
        // 0.weight: (64, 128)
        // 0.bias: (64)
        // 2.weight: (10, 64)
        // 2.bias: (10)
        
        // 访问特定参数
        Parameter fc1Weight = model.getModule("0").getParameter("weight");
        System.out.println("fc1权重形状: " + fc1Weight.data().getShape());
    }
}
```

### 示例6：模型保存和加载

```java
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;
import java.util.Map;

public class SaveLoadExample {
    public static void main(String[] args) {
        // 创建并训练模型
        Sequential model = new Sequential("model")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU())
            .add(new Linear("fc2", 64, 10));
        
        // ... 训练过程 ...
        
        // 保存模型状态
        Map<String, NdArray> state = model.stateDict();
        System.out.println("保存的参数数量: " + state.size());
        
        // 使用您的序列化工具保存state
        // ModelSerializer.save(state, "model.pth");
        
        // 加载模型状态
        // Map<String, NdArray> loadedState = ModelSerializer.load("model.pth");
        
        // 创建新模型并加载参数
        Sequential newModel = new Sequential("model")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU())
            .add(new Linear("fc2", 64, 10));
        
        newModel.loadStateDict(state, true);
        System.out.println("模型加载完成");
    }
}
```

### 示例7：BatchNorm1d使用 ✨

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.BatchNorm1d;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;

public class BatchNormExample {
    public static void main(String[] args) {
        // 构建带BatchNorm的网络
        Sequential model = new Sequential("model")
            .add(new Linear("fc1", 784, 256))
            .add(new BatchNorm1d("bn1", 256))  // BatchNorm层
            .add(new ReLU())
            .add(new Linear("fc2", 256, 128))
            .add(new BatchNorm1d("bn2", 128))  // BatchNorm层
            .add(new ReLU())
            .add(new Linear("fc3", 128, 10));
        
        NdArray inputData = NdArray.randn(Shape.of(32, 784));
        Variable input = new Variable(inputData);
        
        // 训练模式：使用批次统计量
        model.train();
        Variable trainOutput = model.forward(input);
        System.out.println("训练输出形状: " + trainOutput.getShape());
        
        // 推理模式：使用running stats
        model.eval();
        Variable evalOutput = model.forward(input);
        System.out.println("推理输出形状: " + evalOutput.getShape());
        
        // 访问移动平均统计量
        BatchNorm1d bn1 = (BatchNorm1d) model.getModule("1");
        System.out.println("Running mean: " + bn1.getRunningMean());
        System.out.println("Running var: " + bn1.getRunningVar());
        System.out.println("已处理批次数: " + bn1.getNumBatchesTracked());
    }
}
```

#### BatchNorm1d 高级使用

```java
import io.leavesfly.tinyai.nnet.v2.layer.norm.BatchNorm1d;

public class BatchNormAdvancedExample {
    public static void main(String[] args) {
        // 自定义参数的BatchNorm
        BatchNorm1d bn = new BatchNorm1d(
            "custom_bn",
            128,           // num_features
            1e-5f,         // eps
            0.1f,          // momentum
            true,          // affine（使用gamma/beta）
            true           // trackRunningStats
        );
        
        // 训练多个批次
        bn.train();
        for (int i = 0; i < 100; i++) {
            NdArray batch = NdArray.randn(Shape.of(32, 128));
            Variable input = new Variable(batch);
            Variable output = bn.forward(input);
        }
        
        System.out.println("训练后的running stats:");
        System.out.println("  Mean: " + bn.getRunningMean());
        System.out.println("  Var: " + bn.getRunningVar());
        
        // 重置统计量
        bn.resetRunningStats();
        System.out.println("重置后: " + bn.getNumBatchesTracked());
        
        // 不带affine的BatchNorm（不学习gamma/beta）
        BatchNorm1d bnNoAffine = new BatchNorm1d(
            "bn_no_affine", 128, 1e-5f, 0.1f, false, true
        );
        
        System.out.println("不带affine: gamma=" + bnNoAffine.getGamma());
    }
}
```

### 示例8：训练和推理模式切换

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;

public class TrainEvalExample {
    public static void main(String[] args) {
        Sequential model = new Sequential("model")
            .add(new Linear("fc1", 128, 64))
            .add(new ReLU())
            .add(new Dropout("drop", 0.5f))
            .add(new Linear("fc2", 64, 10));
        
        NdArray inputData = NdArray.randn(Shape.of(32, 128));
        Variable input = new Variable(inputData);
        
        // 训练阶段
        model.train();
        System.out.println("训练模式: " + model.isTraining());
        Variable trainOutput = model.forward(input);
        // Dropout被应用
        
        // 验证阶段
        model.eval();
        System.out.println("推理模式: " + model.isTraining());
        Variable evalOutput = model.forward(input);
        // Dropout被禁用
    }
}
```

### 示例9：自定义模块

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;

public class CustomModuleExample {
    
    /**
     * 自定义残差块
     */
    public static class ResidualBlock extends Module {
        private Linear fc1;
        private Linear fc2;
        private ReLU relu;
        
        public ResidualBlock(String name, int features) {
            super(name);
            
            // 注册子模块
            this.fc1 = (Linear) registerModule("fc1", new Linear("fc1", features, features));
            this.fc2 = (Linear) registerModule("fc2", new Linear("fc2", features, features));
            this.relu = (ReLU) registerModule("relu", new ReLU());
        }
        
        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];
            
            // 残差连接: y = relu(fc2(relu(fc1(x)))) + x
            Variable out = fc1.forward(x);
            out = relu.forward(out);
            out = fc2.forward(out);
            out = out.add(x);  // 残差连接
            out = relu.forward(out);
            
            return out;
        }
    }
    
    public static void main(String[] args) {
        ResidualBlock block = new ResidualBlock("res_block", 128);
        
        NdArray inputData = NdArray.randn(Shape.of(32, 128));
        Variable input = new Variable(inputData);
        
        Variable output = block.forward(input);
        System.out.println("输出形状: " + output.getShape());
        
        // 查看参数
        System.out.println("参数列表:");
        block.namedParameters().forEach((name, param) -> {
            System.out.println("  " + name + ": " + param.data().getShape());
        });
    }
}
```

### 示例10：使用ModuleList

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.container.ModuleList;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;

public class ModuleListExample {
    
    public static class DeepNetwork extends Module {
        private ModuleList layers;
        private int numLayers;
        
        public DeepNetwork(String name, int inputSize, int hiddenSize, int numLayers) {
            super(name);
            this.numLayers = numLayers;
            this.layers = new ModuleList("layers");
            
            // 添加输入层
            layers.add(new Linear("input", inputSize, hiddenSize));
            
            // 添加隐藏层
            for (int i = 0; i < numLayers - 2; i++) {
                layers.add(new Linear("hidden_" + i, hiddenSize, hiddenSize));
            }
            
            // 添加输出层
            layers.add(new Linear("output", hiddenSize, 10));
            
            registerModule("layers", layers);
        }
        
        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];
            
            // 遍历ModuleList
            for (int i = 0; i < layers.size(); i++) {
                x = layers.get(i).forward(x);
                // 最后一层不加激活
                if (i < layers.size() - 1) {
                    x = x.relu();
                }
            }
            
            return x;
        }
    }
    
    public static void main(String[] args) {
        DeepNetwork model = new DeepNetwork("deep_net", 784, 256, 5);
        
        NdArray inputData = NdArray.randn(Shape.of(32, 784));
        Variable input = new Variable(inputData);
        
        Variable output = model.forward(input);
        System.out.println("输出形状: " + output.getShape());
        System.out.println("总层数: " + model.layers.size());
    }
}
```

## 完整训练示例

### 示例10：MNIST分类器

```java
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.activation.ReLU;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;

public class MNISTClassifier {
    
    public static Sequential buildModel() {
        Sequential model = new Sequential("mnist_classifier")
            .add(new Linear("fc1", 784, 512))
            .add(new ReLU())
            .add(new Dropout("drop1", 0.2f))
            .add(new Linear("fc2", 512, 256))
            .add(new ReLU())
            .add(new Dropout("drop2", 0.2f))
            .add(new Linear("fc3", 256, 10));
        
        // 自定义初始化
        model.apply(module -> {
            if (module instanceof Linear) {
                Linear linear = (Linear) module;
                Initializers.kaimingNormal(linear.getWeight().data());
                if (linear.getBias() != null) {
                    Initializers.zeros(linear.getBias().data());
                }
            }
        });
        
        return model;
    }
    
    public static void main(String[] args) {
        // 构建模型
        Sequential model = buildModel();
        
        System.out.println("模型结构:");
        System.out.println(model);
        
        System.out.println("\n参数统计:");
        int totalParams = 0;
        for (var entry : model.namedParameters().entrySet()) {
            int paramCount = entry.getValue().data().getShape().size();
            totalParams += paramCount;
            System.out.println(entry.getKey() + ": " + paramCount);
        }
        System.out.println("总参数量: " + totalParams);
        
        // 训练循环（伪代码）
        model.train();
        // for (epoch in epochs) {
        //     for (batch in trainData) {
        //         Variable output = model.forward(batch.input);
        //         Variable loss = crossEntropyLoss(output, batch.target);
        //         loss.backward();
        //         optimizer.step();
        //         model.clearGrads();
        //     }
        // }
        
        // 推理
        model.eval();
        // Variable testOutput = model.forward(testInput);
    }
}
```

## 总结

这些示例展示了V2版本的主要特性：
- ✅ 简洁的层定义和组合
- ✅ 延迟初始化支持
- ✅ 灵活的初始化策略
- ✅ 训练/推理模式切换
- ✅ 参数访问和管理
- ✅ 模型保存/加载
- ✅ 自定义模块开发
- ✅ 容器模块使用

更多示例请参考项目的测试代码和文档。
