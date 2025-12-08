package io.leavesfly.tinyai.example.regress.v2;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ml.Plot;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.dataset.simple.SinDataSet;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.MeanSquaredLoss;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.optimize.SGD;
import io.leavesfly.tinyai.nnet.v2.container.Sequential;
import io.leavesfly.tinyai.nnet.v2.layer.activation.Sigmoid;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;

import java.util.List;

/**
 * MLP拟合正弦曲线示例 - V2 API版本
 * 
 * @author leavesfly
 * @version 0.02
 * 
 * 使用V2 API重新实现的MLP拟合正弦曲线示例。
 * 该示例演示如何使用多层感知机(MLP)神经网络拟合带有噪声的正弦曲线数据。
 * MLP是一种前馈神经网络，能够学习非线性函数映射，适用于回归和分类任务。
 * <p>
 * V2版本特性：
 * 1. 使用Sequential容器构建网络
 * 2. 使用Sigmoid激活函数（与原版保持一致）
 * 3. 统一的参数初始化策略
 * 4. 训练/推理模式切换
 */
public class MlpSinExamV2 {

    /**
     * 主函数，执行MLP训练和可视化
     * 
     * @param args 命令行参数
     */
    public static void main(String[] args) {

        //====== 1,生成数据====
        int batchSize = 100;
        SinDataSet dataSet = new SinDataSet(batchSize);
        dataSet.prepare();
        // SinDataSet 将数据存储在 splitDatasetMap 中，需要获取训练数据集
        DataSet trainDataSet = dataSet.getTrainDataSet();
        if (trainDataSet == null) {
            throw new IllegalStateException("训练数据集未准备，请确保已调用 prepare() 方法");
        }
        List<Batch> batches = trainDataSet.getBatches();

        Variable variableX = batches.get(0).toVariableX().setName("x").setRequireGrad(false);
        Variable variableY = batches.get(0).toVariableY().setName("y").setRequireGrad(false);

        //====== 2,定义模型====
        // 使用V2 Sequential构建MLP网络：1 -> 10 -> 1，使用Sigmoid激活
        int inputSize = 1;
        int hiddenSize = 10;
        int outputSize = 1;

        Sequential sequential = new Sequential("MlpSinV2")
                .add(new Linear("fc1", inputSize, hiddenSize))
                .add(new Sigmoid())
                .add(new Linear("fc2", hiddenSize, outputSize));

        // V2参数初始化：使用Xavier初始化权重，零初始化偏置
        sequential.apply(module -> {
            if (module instanceof Linear) {
                Linear linear = (Linear) module;
                Initializers.xavierUniform(linear.getWeight().data());
                if (linear.getBias() != null) {
                    Initializers.zeros(linear.getBias().data());
                }
            }
        });

        // 将V2 Sequential包装为V1 Model，以便与现有组件兼容
        Model model = new Model("MlpSinExamV2", sequential);
        Optimizer optimizer = new SGD(model, 0.2f);
        Loss lossFunc = new MeanSquaredLoss();

        //====== 3,训练模型====
        int maxEpoch = 10000;
        sequential.train(); // 设置为训练模式

        for (int i = 0; i < maxEpoch; i++) {
            Variable predictY = sequential.forward(variableX);
            Variable loss = lossFunc.loss(variableY, predictY);

            sequential.clearGrads();
            loss.backward();
            optimizer.update();

            if (i % (maxEpoch / 10) == 0 || (i == maxEpoch - 1)) {
                System.out.println("i=" + i + " loss:" + loss.getValue().getNumber());
            }
        }

        //====== 4,可视化结果====
        sequential.eval(); // 设置为推理模式
        Variable predictY = sequential.forward(variableX);
        float[] p_y = predictY.transpose().getValue().getMatrix()[0];
        float[] x = variableX.transpose().getValue().getMatrix()[0];
        float[] y = variableY.transpose().getValue().getMatrix()[0];
        
        Plot plot = new Plot();
        plot.scatter(x, y);
        plot.line(x, p_y, "line");
        plot.show();
    }
}

