package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-3主体块实现（完全基于V2 Module）
 * 
 * 组成部分：
 * 1. Token嵌入层（GPT3TokenEmbedding - 独立实现）
 * 2. N个GPT3TransformerBlock（并行架构）
 * 3. 最终LayerNorm层
 * 4. 输出投影层（词汇表映射）
 * 
 * 架构流程：
 * <pre>
 * TokenIDs -> TokenEmbedding -> [TransformerBlock x N] -> LayerNorm -> OutputProjection -> Logits
 * </pre>
 * 
 * @author leavesfly
 * @version 2.0 - 完全基于V2 API，完全独立实现
 */
public class GPT3MainBlock extends Module {
    
    private final GPT3Config config;
    
    // 核心组件
    private GPT3TokenEmbedding tokenEmbedding;                // Token嵌入（独立实现）
    private List<GPT3TransformerBlock> transformerBlocks;         // Transformer块列表
    private LayerNorm finalLayerNorm;                             // 最终归一化
    private Linear outputProjection;                              // 输出投影（V2 API）
    
    /**
     * 构造GPT-3主体块
     * 
     * @param name 块名称
     * @param config GPT-3配置
     */
    public GPT3MainBlock(String name, GPT3Config config) {
        super(name);
        this.config = config;
        
        // 初始化所有组件
        initializeComponents();
    }
    
    /**
     * 初始化所有组件（V2 Module方式）
     */
    private void initializeComponents() {
        // 1. 初始化Token嵌入层（GPT-3独立实现，基于V2 Module）
        tokenEmbedding = new GPT3TokenEmbedding(name + "_token_embedding", config);
        registerModule("token_embedding", tokenEmbedding);
        
        // 2. 初始化Transformer块列表
        transformerBlocks = new ArrayList<>();
        for (int i = 0; i < config.getNLayer(); i++) {
            GPT3TransformerBlock block = new GPT3TransformerBlock(
                name + "_transformer_" + i, 
                config
            );
            transformerBlocks.add(block);
            // 注册为子模块
            registerModule("transformer_" + i, block);
        }
        
        // 3. 初始化最终LayerNorm
        finalLayerNorm = new LayerNorm(
            name + "_final_ln", 
            config.getNEmbd(),
            (float) config.getLayerNormEpsilon()
        );
        registerModule("final_ln", finalLayerNorm);
        
        // 4. 初始化输出投影层（将隐藏状态映射到词汇表维度）
        outputProjection = new Linear(
            name + "_output_proj",
            config.getNEmbd(),
            config.getVocabSize(),
            false  // 通常不使用bias
        );
        registerModule("output_proj", outputProjection);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable tokenIds = inputs[0];  // (batch_size, seq_len)
        
        // 验证输入形状
        validateInput(tokenIds);
        
        // 1. Token嵌入
        Variable x = tokenEmbedding.forward(tokenIds);  // (batch_size, seq_len, n_embd)
        
        // 2. 通过所有Transformer块
        for (int i = 0; i < transformerBlocks.size(); i++) {
            GPT3TransformerBlock block = transformerBlocks.get(i);
            x = block.forward(x);
            
            // 可选：梯度检查点（暂不实现）
            if (config.isGradientCheckpointing()) {
                // TODO: 实现梯度检查点逻辑
            }
        }
        
        // 3. 最终LayerNorm
        x = finalLayerNorm.forward(x);  // (batch_size, seq_len, n_embd)
        
        // 4. 输出投影到词汇表维度
        Variable logits = outputProjection.forward(x);  // (batch_size, seq_len, vocab_size)
        
        return logits;
    }
    
    /**
     * 验证输入的有效性
     * 
     * @param tokenIds 输入的token IDs
     */
    private void validateInput(Variable tokenIds) {
        NdArray data = tokenIds.getValue();
        
        if (data.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s", 
                    data.getShape())
            );
        }
        
        int seqLen = data.getShape().getDimension(1);
        if (seqLen > config.getNPositions()) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大位置数(%d)", seqLen, config.getNPositions())
            );
        }
    }
    
    // ==================== 公共方法 ====================
    
    /**
     * 获取模型参数数量（估算）
     * 
     * @return 参数数量
     */
    public long getParameterCount() {
        return config.estimateParameterCount();
    }
    
    /**
     * 打印模型架构信息
     */
    public void printArchitecture() {
        System.out.println("=".repeat(60));
        System.out.println("GPT-3 主体块架构");
        System.out.println("=".repeat(60));
        System.out.printf("配置: %s\n", config);
        System.out.println("-".repeat(60));
        System.out.printf("Token嵌入层: %s\n", tokenEmbedding.getClass().getSimpleName());
        System.out.printf("  - 词汇表大小: %,d\n", config.getVocabSize());
        System.out.printf("  - 嵌入维度: %d\n", config.getNEmbd());
        System.out.printf("  - 最大序列长度: %d\n", config.getNPositions());
        System.out.printf("  - 基于: V2 Module (完全独立实现)\n");
        System.out.println("-".repeat(60));
        System.out.printf("Transformer块数量: %d\n", transformerBlocks.size());
        if (!transformerBlocks.isEmpty()) {
            System.out.printf("  - 每块配置: %s\n", transformerBlocks.get(0));
            System.out.printf("  - 架构模式: %s\n", 
                config.isParallelAttention() ? "并行注意力+MLP" : "串行（GPT-2风格）");
        }
        System.out.println("-".repeat(60));
        System.out.printf("最终LayerNorm: 维度=%d, epsilon=%.1e\n", 
            config.getNEmbd(), config.getLayerNormEpsilon());
        System.out.println("-".repeat(60));
        System.out.printf("输出投影层: %d -> %d\n", 
            config.getNEmbd(), config.getVocabSize());
        System.out.println("-".repeat(60));
        System.out.printf("估算参数数量: %s\n", formatParamCount(getParameterCount()));
        System.out.println("=".repeat(60));
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2f B", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2f M", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    // ==================== Getter方法 ====================
    
    public GPT3TokenEmbedding getTokenEmbedding() {
        return tokenEmbedding;
    }
    
    public List<GPT3TransformerBlock> getTransformerBlocks() {
        return transformerBlocks;
    }
    
    public GPT3TransformerBlock getTransformerBlock(int index) {
        if (index < 0 || index >= transformerBlocks.size()) {
            throw new IndexOutOfBoundsException(
                String.format("Transformer块索引越界: %d (总共%d个块)", index, transformerBlocks.size())
            );
        }
        return transformerBlocks.get(index);
    }
    
    public LayerNorm getFinalLayerNorm() {
        return finalLayerNorm;
    }
    
    public Linear getOutputProjection() {
        return outputProjection;
    }
    
    public GPT3Config getConfig() {
        return config;
    }
    
    @Override
    public String toString() {
        return String.format("GPT3MainBlock{name='%s', nLayer=%d, nEmbd=%d, nHead=%d, params=%s}",
            name, config.getNLayer(), config.getNEmbd(), config.getNHead(), 
            formatParamCount(getParameterCount()));
    }
}
