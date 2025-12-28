package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * GPT-1模型类
 */
public class GPT1Model extends Model {
    
    private final GPT1Config config;
    private final GPT1MainBlock gpt1Block;
    
    public GPT1Model(String name, GPT1Config config) {
        super(name, new GPT1MainBlock(name + "_main", config));
        this.config = config;
        this.gpt1Block = (GPT1MainBlock) getModule();
        setDescription(buildDescription());
    }
    
    private String buildDescription() {
        return String.format(
            "GPT-1语言模型 | 参数量: %s | 层数: %d | 维度: %d | 注意力头: %d | 架构: Post-LayerNorm",
            formatParamCount(config.estimateParameterCount()),
            config.getNLayer(),
            config.getNEmbd(),
            config.getNHead());
    }
    
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    // 工厂方法
    public static GPT1Model createStandardModel(String name) {
        return new GPT1Model(name, GPT1Config.createStandardConfig());
    }
    
    public static GPT1Model createTinyModel(String name) {
        return new GPT1Model(name, GPT1Config.createTinyConfig());
    }
    
    public static GPT1Model createSmallModel(String name) {
        return new GPT1Model(name, GPT1Config.createSmallConfig());
    }
    
    // 推理方法
    public Variable predict(Variable tokenIds) {
        return forward(tokenIds);
    }
    
    public NdArray generateSequence(NdArray promptIds, int maxNewTokens) {
        int batchSize = promptIds.getShape().getDimension(0);
        int promptLen = promptIds.getShape().getDimension(1);
        
        float[][] generatedSeq = new float[batchSize][promptLen + maxNewTokens];
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < promptLen; t++) {
                generatedSeq[b][t] = promptIds.get(b, t);
            }
        }
        
        for (int i = 0; i < maxNewTokens; i++) {
            int currentLen = promptLen + i;
            float[][] currentInput = new float[batchSize][currentLen];
            for (int b = 0; b < batchSize; b++) {
                System.arraycopy(generatedSeq[b], 0, currentInput[b], 0, currentLen);
            }
            
            Variable logits = predict(new Variable(NdArray.of(currentInput)));
            NdArray logitsArray = logits.getValue();
            
            for (int b = 0; b < batchSize; b++) {
                int nextToken = argmax(logitsArray, b, currentLen - 1);
                generatedSeq[b][currentLen] = nextToken;
            }
        }
        
        return NdArray.of(generatedSeq);
    }
    
    private int argmax(NdArray logits, int batchIdx, int seqIdx) {
        int vocabSize = logits.getShape().getDimension(2);
        int maxIdx = 0;
        float maxVal = logits.get(batchIdx, seqIdx, 0);
        
        for (int i = 1; i < vocabSize; i++) {
            float val = logits.get(batchIdx, seqIdx, i);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    @Override
    public void printModelInfo() {
        System.out.println("=".repeat(70));
        System.out.println("GPT-1 模型详细信息");
        System.out.println("=".repeat(70));
        System.out.println("模型名称: " + getName());
        System.out.println("模型描述: " + buildDescription());
        System.out.println("-".repeat(70));
        System.out.println(config);
        System.out.println("-".repeat(70));
        if (gpt1Block != null) {
            gpt1Block.printArchitecture();
        }
        System.out.println("=".repeat(70));
    }
    
    public String getConfigSummary() {
        return String.format(
            "GPT-1配置摘要:\n" +
            "  - 词汇表大小: %,d\n" +
            "  - 嵌入维度: %d\n" +
            "  - Transformer层数: %d\n" +
            "  - 注意力头数: %d\n" +
            "  - 前馈网络维度: %d\n" +
            "  - 最大序列长度: %d\n" +
            "  - 架构: Post-LayerNorm\n" +
            "  - 估算参数量: %s",
            config.getVocabSize(),
            config.getNEmbd(),
            config.getNLayer(),
            config.getNHead(),
            config.getNInner(),
            config.getNPositions(),
            formatParamCount(config.estimateParameterCount()));
    }
    
    public GPT1Config getConfig() { return config; }
    public GPT1MainBlock getGPT1Block() { return gpt1Block; }
    
    @Override
    public String toString() {
        return String.format("GPT1Model{name='%s', params=%s, nLayer=%d, nEmbd=%d}",
            getName(), formatParamCount(config.estimateParameterCount()), 
            config.getNLayer(), config.getNEmbd());
    }
}
