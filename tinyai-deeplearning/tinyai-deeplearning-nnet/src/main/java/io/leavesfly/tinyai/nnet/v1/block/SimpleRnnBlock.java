package io.leavesfly.tinyai.nnet.v1.block;

import io.leavesfly.tinyai.nnet.v1.Block;
import io.leavesfly.tinyai.nnet.v1.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.v1.layer.rnn.SimpleRnnLayer;

/**
 * 简单的递归神经网络的实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * SimpleRnnBlock是一个简单的RNN块实现，包含一个SimpleRnnLayer层和一个线性输出层，
 * 用于构建基本的递归神经网络模型。
 */
public class SimpleRnnBlock extends Block {
    /**
     * SimpleRNN层，用于处理序列数据
     */
    private SimpleRnnLayer rnnLayer;

    /**
     * 线性输出层，用于将RNN的输出映射到目标维度
     */
    private LinearLayer linearLayer;


    public SimpleRnnBlock(String name) {
        super(name);
    }

    /**
     * 构造函数，创建一个简单的RNN块
     *
     * @param name       块的名称
     * @param inputSize  输入特征维度
     * @param hiddenSize 隐藏状态维度
     * @param outputSize 输出维度
     */
    public SimpleRnnBlock(String name, int inputSize, int hiddenSize, int outputSize) {
        super(name);

        rnnLayer = new SimpleRnnLayer("rnn");

        addLayer(rnnLayer);

        linearLayer = new LinearLayer("line", hiddenSize, outputSize, true);
        addLayer(linearLayer);

    }

    @Override
    public void init() {

    }

}