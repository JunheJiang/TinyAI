package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * Qwen3模型类
 * 
 * Qwen3是基于现代Transformer架构的大语言模型，集成了：
 * 1. RMSNorm归一化
 * 2. 旋转位置编码(RoPE)
 * 3. 分组查询注意力(GQA)
 * 4. SwiGLU激活函数
 * 
 * 本实现完全基于TinyAI框架的V2 API，遵循Model-Module-Variable设计模式。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3Model extends Model {
    
    private final Qwen3Config config;
    
    /**
     * 构造函数
     * 
     * @param name 模型名称
     * @param config Qwen3配置
     */
    public Qwen3Model(String name, Qwen3Config config) {
        super(name, new Qwen3Block(name + "_main", config, true));
        this.config = config;
        setDescription(buildDescription());
    }
    
    /**
     * 构建模型描述
     */
    private String buildDescription() {
        return String.format(
            "Qwen3语言模型 | 参数量: %s | 层数: %d | 维度: %d | 头数: %d | 架构: Pre-RMSNorm+GQA+SwiGLU",
            formatParamCount(config.estimateParameterCount()),
            config.getNumHiddenLayers(),
            config.getHiddenSize(),
            config.getNumAttentionHeads()
        );
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    // ==================== 工厂方法 ====================
    
    /**
     * 创建小型Qwen3模型（用于测试）
     */
    public static Qwen3Model createSmallModel(String name) {
        return new Qwen3Model(name, Qwen3Config.createSmallConfig());
    }
    
    /**
     * 创建演示Qwen3模型
     */
    public static Qwen3Model createDemoModel(String name) {
        return new Qwen3Model(name, Qwen3Config.createDemoConfig());
    }
    
    /**
     * 创建标准Qwen3模型
     */
    public static Qwen3Model createStandardModel(String name) {
        return new Qwen3Model(name, Qwen3Config.createStandardConfig());
    }
    
    // ==================== 推理方法 ====================
    
    /**
     * 标准预测方法
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    public Variable predict(Variable tokenIds) {
        return forward(tokenIds);
    }
    
    /**
     * 生成序列（简化版贪婪解码）
     * 
     * @param promptIds 提示词token ID序列 [batch_size, prompt_len]
     * @param maxNewTokens 最大生成token数量
     * @return 生成的完整序列 [batch_size, prompt_len + maxNewTokens]
     */
    public NdArray generateSequence(NdArray promptIds, int maxNewTokens) {
        int batchSize = promptIds.getShape().getDimension(0);
        int promptLen = promptIds.getShape().getDimension(1);
        
        float[][] generatedSeq = new float[batchSize][promptLen + maxNewTokens];
        
        // 复制提示词
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < promptLen; t++) {
                generatedSeq[b][t] = promptIds.get(b, t);
            }
        }
        
        // 自回归生成
        for (int i = 0; i < maxNewTokens; i++) {
            int currentLen = promptLen + i;
            float[][] currentInput = new float[batchSize][currentLen];
            for (int b = 0; b < batchSize; b++) {
                System.arraycopy(generatedSeq[b], 0, currentInput[b], 0, currentLen);
            }
            
            // 预测下一个token
            Variable logits = predict(new Variable(NdArray.of(currentInput)));
            NdArray logitsArray = logits.getValue();
            
            // 贪婪选择
            for (int b = 0; b < batchSize; b++) {
                int nextToken = argmax(logitsArray, b, currentLen - 1);
                generatedSeq[b][currentLen] = nextToken;
            }
        }
        
        return NdArray.of(generatedSeq);
    }
    
    /**
     * 查找最大值的索引(argmax)
     */
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
    
    // ==================== 模型信息 ====================
    
    /**
     * 打印模型详细信息
     */
    @Override
    public void printModelInfo() {
        System.out.println("=".repeat(80));
        System.out.println("Qwen3 模型详细信息");
        System.out.println("=".repeat(80));
        System.out.println("模型名称: " + getName());
        System.out.println("模型描述: " + buildDescription());
        System.out.println("-".repeat(80));
        System.out.println(config);
        System.out.println("=".repeat(80));
    }
    
    /**
     * 获取配置摘要
     */
    public String getConfigSummary() {
        return String.format(
            "Qwen3配置摘要:\n" +
            "  - 词汇表大小: %,d\n" +
            "  - 隐藏维度: %d\n" +
            "  - 中间层维度: %d\n" +
            "  - Transformer层数: %d\n" +
            "  - 注意力头数: %d\n" +
            "  - 键值头数: %d\n" +
            "  - 最大序列长度: %d\n" +
            "  - 架构: Pre-RMSNorm + GQA + SwiGLU\n" +
            "  - 估算总参数: %s",
            config.getVocabSize(),
            config.getHiddenSize(),
            config.getIntermediateSize(),
            config.getNumHiddenLayers(),
            config.getNumAttentionHeads(),
            config.getNumKeyValueHeads(),
            config.getMaxPositionEmbeddings(),
            formatParamCount(config.estimateParameterCount())
        );
    }
    
    // ==================== Getter方法 ====================
    
    public Qwen3Config getConfig() {
        return config;
    }
    
    @Override
    public String toString() {
        return String.format(
            "Qwen3Model{name='%s', params=%s, nLayer=%d, hiddenSize=%d}",
            getName(), 
            formatParamCount(config.estimateParameterCount()),
            config.getNumHiddenLayers(), 
            config.getHiddenSize()
        );
    }
}
