package io.leavesfly.tinyai.nnet.v1.layer.transf;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v1.Layer;
import io.leavesfly.tinyai.nnet.v1.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 多头注意力机制层实现
 * <p>
 * 多头注意力是Transformer的核心组件，通过并行计算多个注意力头来捕获不同子空间的信息。
 * <p>
 * Attention(Q,K,V) = softmax(QK^T/√d_k)V
 * MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
 * <p>
 * 其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
 */
public class MultiHeadAttention extends Layer {

    private int numHeads;
    private int dModel;
    private int dK;
    private int dV;

    // 线性变换层
    private LinearLayer queryLayer;
    private LinearLayer keyLayer;
    private LinearLayer valueLayer;
    private LinearLayer outputLayer;

    private boolean useMask;

    /**
     * 构造多头注意力层
     *
     * @param name     层名称
     * @param dModel   模型维度
     * @param numHeads 注意力头数
     * @param useMask  是否使用掩码（解码器中需要）
     */
    public MultiHeadAttention(String name, int dModel, int numHeads, boolean useMask) {
        super(name);

        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by numHeads");
        }

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = dModel / numHeads;
        this.dV = dModel / numHeads;
        this.useMask = useMask;

        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化线性变换层
            queryLayer = new LinearLayer(name + "_query", dModel, dModel, false);
            keyLayer = new LinearLayer(name + "_key", dModel, dModel, false);
            valueLayer = new LinearLayer(name + "_value", dModel, dModel, false);
            outputLayer = new LinearLayer(name + "_output", dModel, dModel, false);

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        //todo
        return null;
    }

}