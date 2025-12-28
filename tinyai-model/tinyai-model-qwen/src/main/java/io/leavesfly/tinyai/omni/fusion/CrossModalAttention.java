package io.leavesfly.tinyai.omni.fusion;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Permute;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

/**
 * 跨模态注意力层
 * 
 * 实现不同模态特征之间的跨模态交互,支持文本、图像、音频的融合。
 * 
 * 核心机制:
 * - Query来自一个模态
 * - Key和Value来自另一个模态
 * - 通过缩放点积注意力实现信息融合
 * 
 * @author leavesfly
 * @version 1.0
 */
public class CrossModalAttention extends Module {
    
    private final int hiddenSize;
    private final int numHeads;
    private final int headDim;
    private final float dropout;
    
    private final Linear queryProj;
    private final Linear keyProj;
    private final Linear valueProj;
    private final Linear outputProj;
    private final Dropout attnDropout;
    
    public CrossModalAttention(String name, int hiddenSize, int numHeads, float dropout) {
        super(name);
        
        if (hiddenSize % numHeads != 0) {
            throw new IllegalArgumentException(
                "hiddenSize必须能被numHeads整除: " + hiddenSize + " % " + numHeads + " != 0"
            );
        }
        
        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.headDim = hiddenSize / numHeads;
        this.dropout = dropout;
        
        this.queryProj = new Linear(name + "_q_proj", hiddenSize, hiddenSize, true);
        this.keyProj = new Linear(name + "_k_proj", hiddenSize, hiddenSize, true);
        this.valueProj = new Linear(name + "_v_proj", hiddenSize, hiddenSize, true);
        this.outputProj = new Linear(name + "_o_proj", hiddenSize, hiddenSize, true);
        
        registerModule("q_proj", queryProj);
        registerModule("k_proj", keyProj);
        registerModule("v_proj", valueProj);
        registerModule("o_proj", outputProj);
        
        this.attnDropout = new Dropout(name + "_attn_dropout", dropout);
        registerModule("attn_dropout", attnDropout);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块自行初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]: query特征 [batch, query_len, hidden_size]
     *               inputs[1]: key/value特征 [batch, kv_len, hidden_size]
     * @return 融合后的特征 [batch, query_len, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length < 2) {
            throw new IllegalArgumentException("CrossModalAttention需要2个输入");
        }
        
        Variable queryFeatures = inputs[0];
        Variable kvFeatures = inputs[1];
        
        int[] queryShape = queryFeatures.getValue().getShape().getShapeDims();
        int[] kvShape = kvFeatures.getValue().getShape().getShapeDims();
        
        int batchSize = queryShape[0];
        int queryLen = queryShape[1];
        int kvLen = kvShape[1];
        
        // 投影Q, K, V
        Variable Q = queryProj.forward(queryFeatures);
        Variable K = keyProj.forward(kvFeatures);
        Variable V = valueProj.forward(kvFeatures);
        
        // 分割成多头
        Q = splitHeads(Q, batchSize, queryLen);
        K = splitHeads(K, batchSize, kvLen);
        V = splitHeads(V, batchSize, kvLen);
        
        // 计算注意力
        Variable attnOutput = scaledDotProductAttention(Q, K, V);
        
        // 合并多头
        Variable merged = mergeHeads(attnOutput, batchSize, queryLen);
        
        // 输出投影
        return outputProj.forward(merged);
    }
    
    private Variable splitHeads(Variable x, int batchSize, int seqLen) {
        Variable reshaped = x.reshape(Shape.of(batchSize, seqLen, numHeads, headDim));
        return new Permute(0, 2, 1, 3).call(reshaped);
    }
    
    private Variable mergeHeads(Variable x, int batchSize, int seqLen) {
        Variable permuted = new Permute(0, 2, 1, 3).call(x);
        return permuted.reshape(Shape.of(batchSize, seqLen, hiddenSize));
    }
    
    private Variable scaledDotProductAttention(Variable Q, Variable K, Variable V) {
        Variable KT = new Permute(0, 1, 3, 2).call(K);
        Variable scores = Q.matMul(KT);
        
        double scale = Math.sqrt(headDim);
        Variable scaledScores = scores.div(new Variable((float) scale));
        Variable attnWeights = scaledScores.softMax();
        
        if (io.leavesfly.tinyai.util.Config.train && dropout > 0) {
            attnWeights = attnDropout.forward(attnWeights);
        }
        
        return attnWeights.matMul(V);
    }
}
