package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;

/**
 * Qwen3 Transformer解码器块
 * 
 * Pre-LayerNorm架构：
 * 1. RMSNorm -> Self-Attention -> 残差连接
 * 2. RMSNorm -> MLP/MoE -> 残差连接
 * 
 * 支持两种FFN模式:
 * - 标准模式: Qwen3MLPBlock (SwiGLU FFN)
 * - MoE模式: Qwen3MoEBlock (混合专家)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3TransformerBlock extends Module {
    
    private final Qwen3Config config;
    
    private RMSNormLayer inputLayerNorm;        // 注意力前归一化
    private Qwen3AttentionBlock selfAttention;  // 自注意力
    private RMSNormLayer postAttentionLayerNorm; // MLP前归一化
    private Qwen3MLPBlock mlp;                  // 前馈网络(标准模式)
    private Qwen3MoEBlock moe;                  // 混合专家网络(MoE模式)
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Qwen3配置
     */
    public Qwen3TransformerBlock(String name, Qwen3Config config) {
        super(name);
        this.config = config;
        initializeLayers();
    }
    
    /**
     * 初始化各层
     */
    private void initializeLayers() {
        // 输入归一化层
        inputLayerNorm = new RMSNormLayer(
            name + "_input_layernorm",
            config.getHiddenSize(),
            config.getRmsNormEps()
        );
        registerModule("input_layernorm", inputLayerNorm);
        
        // 自注意力块
        selfAttention = new Qwen3AttentionBlock(
            name + "_self_attn",
            config
        );
        registerModule("self_attn", selfAttention);
        
        // 注意力后归一化层
        postAttentionLayerNorm = new RMSNormLayer(
            name + "_post_attention_layernorm",
            config.getHiddenSize(),
            config.getRmsNormEps()
        );
        registerModule("post_attention_layernorm", postAttentionLayerNorm);
        
        // FFN层: 根据配置选择MLP或MoE
        if (config.isEnableMoE()) {
            // MoE模式
            moe = new Qwen3MoEBlock(name + "_moe", config);
            registerModule("moe", moe);
        } else {
            // 标准MLP模式
            mlp = new Qwen3MLPBlock(name + "_mlp", config);
            registerModule("mlp", mlp);
        }
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
            throw new IllegalArgumentException("Transformer块输入不能为空");
        }
        
        Variable hiddenStates = inputs[0];
        
        // 1. 自注意力子层：LayerNorm -> SelfAttention -> Residual
        Variable normed1 = inputLayerNorm.forward(hiddenStates);
        Variable attnOutput = selfAttention.forward(normed1);
        Variable residual1 = hiddenStates.add(attnOutput);
        
        // 2. MLP/MoE子层：LayerNorm -> MLP/MoE -> Residual
        Variable normed2 = postAttentionLayerNorm.forward(residual1);
        Variable ffnOutput;
        if (config.isEnableMoE()) {
            ffnOutput = moe.forward(normed2);
        } else {
            ffnOutput = mlp.forward(normed2);
        }
        Variable output = residual1.add(ffnOutput);
        
        return output;
    }
    
    @Override
    public String toString() {
        String ffnType = config.isEnableMoE() ? "MoE" : "MLP";
        return String.format("Qwen3TransformerBlock{name='%s', ffn=%s}", name, ffnType);
    }
    
    /**
     * 获取MoE使用统计(仅在MoE模式下有效)
     */
    public Qwen3MoEBlock.ExpertUsageStats getMoEStats() {
        if (moe != null) {
            return moe.getUsageStats();
        }
        return null;
    }
}
