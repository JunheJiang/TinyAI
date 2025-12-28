package io.leavesfly.tinyai.omni.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.SiLU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.omni.config.Qwen3OmniConfig;
import io.leavesfly.tinyai.omni.config.ModalityType;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen3-Omni混合专家层(MoE Layer)
 * 
 * 基于DeepSeek V3的MoE架构,为多模态场景优化:
 * 1. 支持文本/图像/音频三种模态的专家路由
 * 2. 使用SwiGLU激活函数(与Qwen3一致)
 * 3. 模态感知的专家分配
 * 4. 负载均衡和效率优化
 * 
 * 核心流程:
 * Input → Gating Network → Top-K Selection → Expert Processing → Weighted Combination → Output
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3OmniMoELayer extends Module {
    
    private final Qwen3OmniConfig config;
    
    // 门控网络
    private final Linear gatingNetwork;
    
    // 专家网络列表
    private final List<ExpertNetwork> experts;
    
    // Dropout层
    private final Dropout expertDropout;
    
    // 统计信息
    private long[] expertUsageCount;
    private long totalCalls;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Qwen3-Omni配置对象
     */
    public Qwen3OmniMoELayer(String name, Qwen3OmniConfig config) {
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
            true  // 使用偏置
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
        
        // 初始化Dropout层
        this.expertDropout = new Dropout(
            name + "_expert_dropout",
            (float) config.getDropoutRate()
        );
        registerModule("expert_dropout", expertDropout);
        
        // 初始化统计信息
        this.expertUsageCount = new long[numExperts];
        this.totalCalls = 0;
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块自行初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]: 输入张量 [batch_size, seq_len, hidden_size]
     *               inputs[1](可选): 模态类型 ModalityType
     * @return MoE输出结果
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable input = inputs[0];
        ModalityType modalityType = null;
        if (inputs.length > 1 && inputs[1] != null) {
            // 模态类型可用于专家路由偏置
        }
        
        // 执行MoE计算
        MoEOutput moeOutput = computeMoE(input, modalityType);
        
        // 应用dropout
        return expertDropout.forward(moeOutput.output);
    }
    
    /**
     * 执行MoE计算（核心方法）
     * 
     * @param input 输入张量 [batch_size, seq_len, hidden_size]
     * @param modalityType 模态类型（可选）
     * @return MoE输出结果
     */
    public MoEOutput computeMoE(Variable input, ModalityType modalityType) {
        
        // 1. 计算门控logits: [batch_size, seq_len, numExperts]
        Variable gatingLogits = gatingNetwork.forward(input);
        
        // 2. 应用模态感知偏置（如果提供了模态类型）
        if (modalityType != null) {
            gatingLogits = applyModalityBias(gatingLogits, modalityType);
        }
        
        // 3. 计算门控概率（softmax）
        Variable gatingProbs = gatingLogits.softMax();
        
        // 4. Top-K选择
        TopKResult topKResult = selectTopK(gatingProbs, config.getExpertTopK());
        
        // 5. 专家计算和加权组合
        Variable expertOutputs = computeExpertOutputs(input, topKResult);
        
        // 6. 计算负载均衡损失
        double loadBalanceLoss = 0.0;
        if (config.isExpertLoadBalance()) {
            loadBalanceLoss = computeLoadBalanceLoss(gatingProbs);
        }
        
        return new MoEOutput(expertOutputs, gatingProbs, topKResult, loadBalanceLoss);
    }
    
    /**
     * 应用模态感知偏置
     * 不同模态倾向于使用不同的专家组
     */
    private Variable applyModalityBias(Variable gatingLogits, ModalityType modalityType) {
        float[] bias = getModalityBias(modalityType);
        
        // 将偏置转换为 [1, 1, numExperts] 形状
        NdArray biasArray = NdArray.of(bias);
        Variable biasVar = new Variable(biasArray);
        Variable bias3D = biasVar.reshape(Shape.of(1, 1, config.getNumExperts()));
        
        // 使用Variable的add算子（自动广播）
        return gatingLogits.add(bias3D);
    }
    
    /**
     * 获取模态类型的专家偏置
     */
    private float[] getModalityBias(ModalityType modalityType) {
        int numExperts = config.getNumExperts();
        float[] bias = new float[numExperts];
        
        // 为不同模态分配专家组
        int expertsPerModality = Math.max(1, numExperts / 3);  // 3种模态
        int startIdx = 0;
        
        switch (modalityType) {
            case TEXT:
                startIdx = 0;
                break;
            case IMAGE:
                startIdx = expertsPerModality;
                break;
            case AUDIO:
                startIdx = expertsPerModality * 2;
                break;
        }
        
        // 为对应模态的专家组添加正偏置
        for (int i = 0; i < expertsPerModality && (startIdx + i) < numExperts; i++) {
            bias[startIdx + i] = 0.5f;
        }
        
        return bias;
    }
    
    /**
     * 选择Top-K专家
     */
    private TopKResult selectTopK(Variable probs, int k) {
        NdArray probsArray = probs.getValue();
        int batchSize = probsArray.getShape().getDimension(0);
        int seqLen = probsArray.getShape().getDimension(1);
        int numExperts = probsArray.getShape().getDimension(2);
        
        int[][][] topKIndices = new int[batchSize][seqLen][k];
        float[][][] topKWeights = new float[batchSize][seqLen][k];
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                // 获取当前位置的所有专家概率
                float[] expertProbs = new float[numExperts];
                for (int e = 0; e < numExperts; e++) {
                    expertProbs[e] = probsArray.get(b, t, e);
                }
                
                // 选择Top-K
                int[] topK = getTopKIndices(expertProbs, k);
                float[] topKProbs = new float[k];
                float sumTopKProbs = 0.0f;
                
                for (int i = 0; i < k; i++) {
                    topKProbs[i] = expertProbs[topK[i]];
                    sumTopKProbs += topKProbs[i];
                }
                
                // 归一化权重（使Top-K概率和为1）
                for (int i = 0; i < k; i++) {
                    topKIndices[b][t][i] = topK[i];
                    topKWeights[b][t][i] = topKProbs[i] / sumTopKProbs;
                }
            }
        }
        
        return new TopKResult(topKIndices, topKWeights);
    }
    
    /**
     * 获取Top-K索引
     */
    private int[] getTopKIndices(float[] values, int k) {
        int[] indices = new int[k];
        boolean[] used = new boolean[values.length];
        
        for (int i = 0; i < k; i++) {
            int maxIdx = -1;
            float maxVal = Float.NEGATIVE_INFINITY;
            
            for (int j = 0; j < values.length; j++) {
                if (!used[j] && values[j] > maxVal) {
                    maxVal = values[j];
                    maxIdx = j;
                }
            }
            
            indices[i] = maxIdx;
            used[maxIdx] = true;
        }
        
        return indices;
    }
    
    /**
     * 计算所有专家的输出并加权组合
     * 
     * 优化策略：批量计算，完全在Variable层面
     */
    private Variable computeExpertOutputs(Variable input, TopKResult topKResult) {
        int batchSize = input.getValue().getShape().getDimension(0);
        int seqLen = input.getValue().getShape().getDimension(1);
        int hiddenSize = input.getValue().getShape().getDimension(2);
        int numExperts = config.getNumExperts();
        
        // 所有专家并行计算
        List<Variable> expertOutputs = new ArrayList<>();
        for (int i = 0; i < numExperts; i++) {
            Variable expertOut = experts.get(i).forward(input);
            expertOutputs.add(expertOut);
        }
        
        // 根据TopK结果加权组合
        Variable result = createWeightedExpertCombination(
            expertOutputs, topKResult, batchSize, seqLen, hiddenSize
        );
        
        return result;
    }
    
    /**
     * 根据TopK结果加权组合专家输出
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
            
            // 获取该专家的输出并加权
            Variable expertOut = expertOutputs.get(expertIdx);
            
            // 使用广播乘法: [batch, seq, 1] × [batch, seq, hidden]
            Variable weightedOut = expertOut.mul(weightMask);
            
            // 累加到输出
            output = output.add(weightedOut);
            
            // 更新统计
            expertUsageCount[expertIdx]++;
        }
        
        totalCalls += batchSize * seqLen;
        
        return output;
    }
    
    /**
     * 为指定专家创建权重mask
     */
    private Variable createExpertWeightMask(
            int expertIdx,
            TopKResult topKResult,
            int batchSize,
            int seqLen) {
        
        float[][][] weights = new float[batchSize][seqLen][1];
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
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
     * 计算负载均衡损失
     */
    private double computeLoadBalanceLoss(Variable gatingProbs) {
        NdArray probsArray = gatingProbs.getValue();
        int batchSize = probsArray.getShape().getDimension(0);
        int seqLen = probsArray.getShape().getDimension(1);
        int numExperts = probsArray.getShape().getDimension(2);
        
        // 计算每个专家的平均使用频率
        float[] expertFreq = new float[numExperts];
        int totalTokens = batchSize * seqLen;
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int e = 0; e < numExperts; e++) {
                    expertFreq[e] += probsArray.get(b, t, e);
                }
            }
        }
        
        for (int e = 0; e < numExperts; e++) {
            expertFreq[e] /= totalTokens;
        }
        
        // 计算方差
        float idealFreq = 1.0f / numExperts;
        float variance = 0.0f;
        
        for (int e = 0; e < numExperts; e++) {
            float diff = expertFreq[e] - idealFreq;
            variance += diff * diff;
        }
        
        return variance * config.getExpertLoadCoef();
    }
    
    /**
     * 获取专家使用统计
     */
    public ExpertUsageStats getUsageStats() {
        float[] usageRate = new float[config.getNumExperts()];
        
        if (totalCalls > 0) {
            for (int e = 0; e < config.getNumExperts(); e++) {
                usageRate[e] = (float) expertUsageCount[e] / totalCalls;
            }
        }
        
        return new ExpertUsageStats(expertUsageCount.clone(), usageRate, totalCalls);
    }
    
    /**
     * 重置统计信息
     */
    public void resetStats() {
        expertUsageCount = new long[config.getNumExperts()];
        totalCalls = 0;
    }
    
    /**
     * 专家网络内部类
     * 使用SwiGLU激活函数（与Qwen3一致）
     */
    private static class ExpertNetwork extends Module {
        private final Linear gate;
        private final Linear up;
        private final Linear down;
        private final SiLU silu;
        
        public ExpertNetwork(String name, int inputDim, int hiddenDim) {
            super(name);
            
            // SwiGLU需要两个上投影
            this.gate = new Linear(name + "_gate", inputDim, hiddenDim, false);
            this.up = new Linear(name + "_up", inputDim, hiddenDim, false);
            this.down = new Linear(name + "_down", hiddenDim, inputDim, false);
            this.silu = new SiLU(name + "_silu");
            
            registerModule("gate", gate);
            registerModule("up", up);
            registerModule("down", down);
            registerModule("silu", silu);
        }
        
        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];
            
            // SwiGLU: down(silu(gate(x)) * up(x))
            Variable gateOut = gate.forward(x);
            Variable gateActivated = silu.forward(gateOut);
            Variable upOut = up.forward(x);
            Variable combined = gateActivated.mul(upOut);
            
            return down.forward(combined);
        }
    }
    
    /**
     * Top-K选择结果类
     */
    public static class TopKResult {
        public final int[][][] indices;   // [batch_size, seq_len, k]
        public final float[][][] weights; // [batch_size, seq_len, k]
        
        public TopKResult(int[][][] indices, float[][][] weights) {
            this.indices = indices;
            this.weights = weights;
        }
    }
    
    /**
     * MoE输出结果类
     */
    public static class MoEOutput {
        public final Variable output;
        public final Variable gatingProbs;
        public final TopKResult topKResult;
        public final double loadBalanceLoss;
        
        public MoEOutput(Variable output, Variable gatingProbs, 
                        TopKResult topKResult, double loadBalanceLoss) {
            this.output = output;
            this.gatingProbs = gatingProbs;
            this.topKResult = topKResult;
            this.loadBalanceLoss = loadBalanceLoss;
        }
        
        @Override
        public String toString() {
            return String.format(
                "MoEOutput{loadBalanceLoss=%.6f, outputShape=%s}",
                loadBalanceLoss,
                output.getValue().getShape()
            );
        }
    }
    
    /**
     * 专家使用统计类
     */
    public static class ExpertUsageStats {
        private final long[] usageCount;
        private final float[] usageRate;
        private final long totalCalls;
        
        public ExpertUsageStats(long[] usageCount, float[] usageRate, long totalCalls) {
            this.usageCount = usageCount;
            this.usageRate = usageRate;
            this.totalCalls = totalCalls;
        }
        
        public long[] getUsageCount() {
            return usageCount;
        }
        
        public float[] getUsageRate() {
            return usageRate;
        }
        
        public long getTotalCalls() {
            return totalCalls;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("ExpertUsageStats{\n");
            for (int i = 0; i < usageCount.length; i++) {
                sb.append(String.format("  Expert%d: count=%d, rate=%.2f%%\n",
                    i, usageCount[i], usageRate[i] * 100));
            }
            sb.append(String.format("  Total calls: %d\n}", totalCalls));
            return sb.toString();
        }
    }
}
