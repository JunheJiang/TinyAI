package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen3混合专家块(Mixture of Experts Block)
 * 
 * 基于DeepSeek V3的MoE架构,通过门控网络动态选择Top-K个专家处理输入:
 * 1. 门控网络(Gating Network) - 计算每个专家的选择概率
 * 2. 专家网络(Expert Networks) - 多个独立的SwiGLU FFN
 * 3. Top-K选择 - 选择概率最高的K个专家
 * 4. 加权组合 - 根据门控权重组合专家输出
 * 
 * 架构流程:
 * Input → Gating Network → Top-K Selection → Expert Processing → Weighted Combination → Output
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3MoEBlock extends Module {
    
    private final Qwen3Config config;
    
    // 门控网络
    private final Linear gatingNetwork;
    
    // 专家网络列表
    private final List<ExpertNetwork> experts;
    
    // 统计信息
    private long[] expertUsageCount;
    private long totalCalls;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Qwen3配置
     */
    public Qwen3MoEBlock(String name, Qwen3Config config) {
        super(name);
        this.config = config;
        
        int hiddenSize = config.getHiddenSize();
        int numExperts = config.getNumExperts();
        int expertHiddenSize = config.getExpertHiddenSize();
        
        // 初始化门控网络: hiddenSize -> numExperts
        this.gatingNetwork = new Linear(
            name + "_gating",
            hiddenSize,
            numExperts,
            false  // 不使用偏置
        );
        registerModule("gating", gatingNetwork);
        
        // 初始化专家网络
        this.experts = new ArrayList<>();
        for (int i = 0; i < numExperts; i++) {
            ExpertNetwork expert = new ExpertNetwork(
                name + "_expert_" + i,
                hiddenSize,
                expertHiddenSize
            );
            experts.add(expert);
            registerModule("expert_" + i, expert);
        }
        
        // 初始化统计信息
        this.expertUsageCount = new long[numExperts];
        this.totalCalls = 0;
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为输入隐藏状态 [batch_size, seq_len, hidden_size]
     * @return 输出隐藏状态 [batch_size, seq_len, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("MoE输入不能为空");
        }
        
        Variable input = inputs[0];
        
        // 1. 计算门控logits: [batch, seq, numExperts]
        Variable gatingLogits = gatingNetwork.forward(input);
        
        // 2. Softmax归一化得到概率
        Variable gatingProbs = softmax(gatingLogits);
        
        // 3. Top-K选择
        TopKResult topKResult = selectTopK(gatingProbs, config.getExpertTopK());
        
        // 4. 专家计算并加权组合
        Variable output = computeExpertOutputs(input, topKResult);
        
        return output;
    }
    
    /**
     * 对最后一个维度进行Softmax
     */
    private Variable softmax(Variable logits) {
        // 使用Variable的softmax算子
        return logits.softMax();
    }
    
    /**
     * Top-K选择
     */
    private TopKResult selectTopK(Variable probs, int topK) {
        NdArray probsArray = probs.getValue();
        Shape shape = probsArray.getShape();
        int batchSize = shape.getDimension(0);
        int seqLen = shape.getDimension(1);
        int numExperts = shape.getDimension(2);
        
        // 存储Top-K索引和权重
        int[][][] topKIndices = new int[batchSize][seqLen][topK];
        float[][][] topKWeights = new float[batchSize][seqLen][topK];
        
        // 对每个batch和每个token进行Top-K选择
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                // 获取当前token的专家概率
                float[] expertProbs = new float[numExperts];
                for (int e = 0; e < numExperts; e++) {
                    expertProbs[e] = probsArray.get(b, t, e);
                }
                
                // 找到Top-K专家
                int[] indices = new int[numExperts];
                for (int i = 0; i < numExperts; i++) {
                    indices[i] = i;
                }
                
                // 简单选择排序找Top-K
                for (int k = 0; k < topK; k++) {
                    int maxIdx = k;
                    for (int i = k + 1; i < numExperts; i++) {
                        if (expertProbs[indices[i]] > expertProbs[indices[maxIdx]]) {
                            maxIdx = i;
                        }
                    }
                    // 交换
                    int temp = indices[k];
                    indices[k] = indices[maxIdx];
                    indices[maxIdx] = temp;
                    
                    topKIndices[b][t][k] = indices[k];
                    topKWeights[b][t][k] = expertProbs[indices[k]];
                }
                
                // 归一化Top-K权重
                float sum = 0.0f;
                for (int k = 0; k < topK; k++) {
                    sum += topKWeights[b][t][k];
                }
                if (sum > 0) {
                    for (int k = 0; k < topK; k++) {
                        topKWeights[b][t][k] /= sum;
                    }
                }
            }
        }
        
        return new TopKResult(topKIndices, topKWeights);
    }
    
    /**
     * 计算专家输出并加权组合
     * 
     * ✅ 优化方案：批量计算，完全在Variable层面
     * 策略：
     * 1. 让所有专家并行处理整个batch的输入
     * 2. 根据TopK结果构建权重矩阵
     * 3. 使用Variable算子进行加权组合
     * 
     * 这样既保证了梯度回传，又避免了逐位置的循环
     */
    private Variable computeExpertOutputs(Variable input, TopKResult topKResult) {
        NdArray inputArray = input.getValue();
        Shape shape = inputArray.getShape();
        int batchSize = shape.getDimension(0);
        int seqLen = shape.getDimension(1);
        int hiddenSize = shape.getDimension(2);
        int numExperts = experts.size();
        
        // ✅ 步骤1：所有专家并行计算
        List<Variable> expertOutputs = new ArrayList<>();
        for (int i = 0; i < numExperts; i++) {
            // 每个专家处理整个batch
            Variable expertOut = experts.get(i).forward(input);
            expertOutputs.add(expertOut);
        }
        
        // ✅ 步骤2：在Variable层面进行加权组合
        Variable output = createWeightedExpertCombination(
            expertOutputs, topKResult, batchSize, seqLen, hiddenSize
        );
        
        return output;
    }
    
    /**
     * ✅ 根据TopK结果加权组合专家输出（完全在Variable层面）
     * 
     * 关键：使用Variable算子(mul, add)保持计算图完整性
     */
    private Variable createWeightedExpertCombination(
            List<Variable> expertOutputs,
            TopKResult topKResult,
            int batchSize,
            int seqLen,
            int hiddenSize) {
        
        // 初始化输出为零
        NdArray outputArray = NdArray.zeros(Shape.of(batchSize, seqLen, hiddenSize));
        Variable output = new Variable(outputArray);
        
        // 对每个专家，构建其权重mask并累加
        for (int expertIdx = 0; expertIdx < expertOutputs.size(); expertIdx++) {
            // 构建该专家的权重矩阵 [batch_size, seq_len, 1]
            Variable weightMask = createExpertWeightMask(
                expertIdx, topKResult, batchSize, seqLen
            );
            
            // 如果该专家没有被任何位置选中，跳过
            if (isZeroMask(weightMask)) {
                continue;
            }
            
            // 更新统计信息
            updateExpertUsageStats(expertIdx, topKResult, batchSize, seqLen);
            
            // 获取该专家的输出
            Variable expertOut = expertOutputs.get(expertIdx);
            
            // ✅ 使用Variable算子进行加权：[batch, seq, 1] × [batch, seq, hidden]
            // 广播机制自动扩展维度
            Variable weightedOut = expertOut.mul(weightMask);
            
            // ✅ 使用Variable算子累加（保持计算图）
            output = output.add(weightedOut);
        }
        
        totalCalls += batchSize * seqLen;
        
        return output;
    }
    
    /**
     * 为指定专家创建权重mask
     * 返回 [batch_size, seq_len, 1] 的权重矩阵
     */
    private Variable createExpertWeightMask(
            int expertIdx,
            TopKResult topKResult,
            int batchSize,
            int seqLen) {
        
        float[][][] weights = new float[batchSize][seqLen][1];
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                // 检查该位置的TopK中是否包含当前专家
                for (int k = 0; k < config.getExpertTopK(); k++) {
                    if (topKResult.indices[b][t][k] == expertIdx) {
                        weights[b][t][0] = topKResult.weights[b][t][k];
                        break;
                    }
                }
            }
        }
        
        return new Variable(NdArray.of(weights));
    }
    
    /**
     * 检查权重mask是否全为0
     */
    private boolean isZeroMask(Variable mask) {
        NdArray arr = mask.getValue();
        float sum = arr.sum().getNumber().floatValue();
        return Math.abs(sum) < 1e-9f;
    }
    
    /**
     * 更新专家使用统计
     */
    private void updateExpertUsageStats(
            int expertIdx,
            TopKResult topKResult,
            int batchSize,
            int seqLen) {
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int k = 0; k < config.getExpertTopK(); k++) {
                    if (topKResult.indices[b][t][k] == expertIdx) {
                        expertUsageCount[expertIdx]++;
                        break;
                    }
                }
            }
        }
    }
    
    /**
     * 获取专家使用统计
     */
    public ExpertUsageStats getUsageStats() {
        return new ExpertUsageStats(expertUsageCount, totalCalls);
    }
    
    /**
     * 重置统计信息
     */
    public void resetStats() {
        for (int i = 0; i < expertUsageCount.length; i++) {
            expertUsageCount[i] = 0;
        }
        totalCalls = 0;
    }
    
    /**
     * 专家网络（使用SwiGLU激活的FFN）
     */
    private static class ExpertNetwork extends Module {
        private final Linear gateProj;
        private final Linear upProj;
        private final Linear downProj;
        
        public ExpertNetwork(String name, int hiddenSize, int expertHiddenSize) {
            super(name);
            
            // Gate投影
            this.gateProj = new Linear(
                name + "_gate",
                hiddenSize,
                expertHiddenSize,
                false
            );
            registerModule("gate", gateProj);
            
            // Up投影
            this.upProj = new Linear(
                name + "_up",
                hiddenSize,
                expertHiddenSize,
                false
            );
            registerModule("up", upProj);
            
            // Down投影
            this.downProj = new Linear(
                name + "_down",
                expertHiddenSize,
                hiddenSize,
                false
            );
            registerModule("down", downProj);
        }
        
        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];
            
            // SwiGLU: down(swish(gate(x)) * up(x))
            Variable gateOut = gateProj.forward(x);
            Variable gateActivated = applySwish(gateOut);
            Variable upOut = upProj.forward(x);
            Variable combined = gateActivated.mul(upOut);
            
            return downProj.forward(combined);
        }
        
        /**
         * 应用Swish激活函数: swish(x) = x * sigmoid(x)
         */
        private Variable applySwish(Variable x) {
            Variable negX = x.mul(new Variable(-1.0f));
            Variable expNegX = negX.exp();
            Variable onePlusExp = expNegX.add(new Variable(1.0f));
            Variable sigmoid = new Variable(1.0f).div(onePlusExp);
            return x.mul(sigmoid);
        }
    }
    
    /**
     * Top-K选择结果
     */
    private static class TopKResult {
        final int[][][] indices;   // [batch, seq, topK]
        final float[][][] weights; // [batch, seq, topK]
        
        TopKResult(int[][][] indices, float[][][] weights) {
            this.indices = indices;
            this.weights = weights;
        }
    }
    
    /**
     * 专家使用统计
     */
    public static class ExpertUsageStats {
        private final long[] counts;
        private final long total;
        
        public ExpertUsageStats(long[] counts, long total) {
            this.counts = counts.clone();
            this.total = total;
        }
        
        public long[] getCounts() {
            return counts.clone();
        }
        
        public long getTotal() {
            return total;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("ExpertUsageStats{\n");
            for (int i = 0; i < counts.length; i++) {
                double rate = total > 0 ? (counts[i] * 100.0 / total) : 0.0;
                sb.append(String.format("  Expert%d: count=%d, rate=%.2f%%\n", 
                    i, counts[i], rate));
            }
            sb.append(String.format("  Total calls: %d\n", total));
            sb.append("}");
            return sb.toString();
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "Qwen3MoEBlock{name='%s', numExperts=%d, topK=%d, hiddenSize=%d}",
            name, experts.size(), config.getExpertTopK(), config.getHiddenSize()
        );
    }
}
