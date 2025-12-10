package io.leavesfly.tinyai.nnet.v1.layer.embedd;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v1.Layer;
import io.leavesfly.tinyai.nnet.v1.ParameterV1;

import java.util.Arrays;
import java.util.List;

/**
 * 词嵌入层实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * Embedding层用于将离散的词汇索引转换为连续的向量表示。
 * 它维护一个词汇表大小×嵌入维度的权重矩阵，通过查找表的方式获取对应词向量。
 * <p>
 * 前向传播过程：
 * 1. 输入为词汇索引序列
 * 2. 根据索引从权重矩阵中查找对应的词向量
 * <p>
 * 反向传播过程：
 * 1. 梯度通过索引位置累加到权重矩阵对应位置
 */
public class Embedding extends Layer {
    /**
     * 嵌入权重矩阵
     * 形状: (vocabSize, embedSize)
     */
    private ParameterV1 wIn;

    /**
     * 词汇表大小
     */
    private int vocabSize;

    /**
     * 嵌入维度
     */
    private int embedSize;

    public Embedding(String _name) {
        super(_name);
    }

    /**
     * 构造函数，创建Embedding层实例
     *
     * @param _name     层名称
     * @param vocabSize 词汇表大小
     * @param embedSize 嵌入维度
     */
    public Embedding(String _name, int vocabSize, int embedSize) {
        super(_name, Shape.of(vocabSize), Shape.of(vocabSize, embedSize));
        this.vocabSize = vocabSize;
        this.embedSize = embedSize;
        NdArray initWeight = NdArray.likeRandomN(Shape.of(vocabSize, embedSize)).mulNum(0.01f);
        wIn = new ParameterV1(initWeight);
        wIn.setName("wIn");
        addParam(wIn.getName(), wIn);
    }

    @Override
    public void init() {
        // Embedding层不需要额外的初始化
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        //todo
        return null;
    }

    /**
     * 层的前向传播计算
     * <p>
     * 根据输入的词汇索引从权重矩阵中查找对应的词向量
     *
     * @param inputs 输入变量数组，包含词汇索引
     * @return 前向传播结果变量
     */


    /**
     * 获取词汇表大小
     *
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * 获取嵌入维度
     *
     * @return 嵌入维度
     */
    public int getEmbedSize() {
        return embedSize;
    }

    /**
     * 获取嵌入权重矩阵
     *
     * @return 嵌入权重矩阵参数
     */
    public ParameterV1 getWeight() {
        return wIn;
    }
}