package io.leavesfly.tinyai.nnet.v1.layer.transf;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
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
            
            // 注册子层参数
            params.putAll(queryLayer.getParams());
            params.putAll(keyLayer.getParams());
            params.putAll(valueLayer.getParams());
            params.putAll(outputLayer.getParams());

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("MultiHeadAttention requires input");
        }
        
        // 支持自注意力：只提供1个输入
        Variable query = inputs[0];
        Variable key = inputs.length > 1 ? inputs[1] : query;
        Variable value = inputs.length > 2 ? inputs[2] : query;
        
        NdArray queryData = query.getValue();
        Shape queryShape = queryData.getShape();
        
        // 输入形状: (batch_size, seq_len, d_model)
        if (queryShape.getDimNum() != 3) {
            throw new IllegalArgumentException(
                String.format("MultiHeadAttention expects 3D input, got %dD", queryShape.getDimNum())
            );
        }
        
        int batchSize = queryShape.getDimension(0);
        int seqLen = queryShape.getDimension(1);
        
        // 1. QKV投影: (batch, seq, d_model) -> (batch, seq, d_model)
        Variable Q = queryLayer.layerForward(query);
        Variable K = keyLayer.layerForward(key);
        Variable V = valueLayer.layerForward(value);
        
        // 2. 分割成多头: (batch, seq, d_model) -> (batch, seq, num_heads, d_k)
        Q = Q.reshape(Shape.of(batchSize, seqLen, numHeads, dK));
        K = K.reshape(Shape.of(batchSize, seqLen, numHeads, dK));
        V = V.reshape(Shape.of(batchSize, seqLen, numHeads, dV));
        
        // 3. 维度置换: (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
        Q = new Permute(0, 2, 1, 3).call(Q);
        K = new Permute(0, 2, 1, 3).call(K);
        V = new Permute(0, 2, 1, 3).call(V);
        
        // 4. 计算注意力分数: scores = Q @ K^T / sqrt(d_k)
        // K^T: (batch, num_heads, seq, d_k) -> (batch, num_heads, d_k, seq)
        Variable K_T = new Permute(0, 1, 3, 2).call(K);
        
        // QK^T: (batch, num_heads, seq, seq)
        Variable scores = Q.bmm(K_T);
        
        // 缩放
        double scale = Math.sqrt(dK);
        Variable scaleVar = new Variable(NdArray.of((float)scale));
        scores = scores.div(scaleVar);
        
        // 5. 应用掩码（如果需要）
        if (useMask) {
            // 创建因果掩码: 上三角为-inf
            NdArray maskData = NdArray.zeros(Shape.of(seqLen, seqLen));
            for (int i = 0; i < seqLen; i++) {
                for (int j = i + 1; j < seqLen; j++) {
                    maskData.set(Float.NEGATIVE_INFINITY, i, j);
                }
            }
            Variable mask = new Variable(maskData);
            scores = scores.add(mask);
        }
        
        // 6. Softmax归一化
        Variable attnWeights = scores.softMax();
        
        // 7. 应用注意力权重: (batch, num_heads, seq, seq) @ (batch, num_heads, seq, d_v)
        //    -> (batch, num_heads, seq, d_v)
        Variable attnOutput = attnWeights.bmm(V);
        
        // 8. 维度置换回来: (batch, num_heads, seq, d_v) -> (batch, seq, num_heads, d_v)
        attnOutput = new Permute(0, 2, 1, 3).call(attnOutput);
        
        // 9. 合并多头: (batch, seq, num_heads, d_v) -> (batch, seq, d_model)
        attnOutput = attnOutput.reshape(Shape.of(batchSize, seqLen, dModel));
        
        // 10. 输出投影
        Variable output = outputLayer.layerForward(attnOutput);
        
        return output;
    }

}