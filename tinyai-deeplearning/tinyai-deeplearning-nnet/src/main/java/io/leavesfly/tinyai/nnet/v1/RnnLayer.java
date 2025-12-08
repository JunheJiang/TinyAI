package io.leavesfly.tinyai.nnet.v1;


import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 递归神经网络层，区别于普通的前馈网络，有状态
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * RnnLayer是递归神经网络层的抽象基类，继承自Layer类。
 * 与普通前馈网络层不同，RNN层具有内部状态，能够处理序列数据。
 */
public abstract class RnnLayer extends Layer {


    public RnnLayer(String _name) {
        super(_name);
    }


    /**
     * 构造函数，初始化Layer的基本属性
     *
     * @param _name       Layer的名称
     * @param _inputShape 输入数据的形状
     */
    public RnnLayer(String _name, Shape _inputShape) {
        super(_name, _inputShape);
    }

    /**
     * 构造函数，初始化Layer的基本属性（包含输出形状）
     *
     * @param _name        Layer的名称
     * @param _inputShape  输入数据的形状
     * @param _outputShape 输出数据的形状
     */
    public RnnLayer(String _name, Shape _inputShape, Shape _outputShape) {
        super(_name, _inputShape, _outputShape);
    }

    /**
     * 重置RNN层的内部状态
     */
    public abstract void resetState();

}