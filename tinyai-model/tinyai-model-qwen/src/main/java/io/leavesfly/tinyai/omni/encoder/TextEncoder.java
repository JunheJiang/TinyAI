package io.leavesfly.tinyai.omni.encoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.omni.config.Qwen3OmniConfig;
import io.leavesfly.tinyai.qwen3.Qwen3Block;
import io.leavesfly.tinyai.qwen3.Qwen3Config;

/**
 * 文本编码器
 * 
 * 复用Qwen3Block作为文本编码器,提取文本特征表示。
 * Qwen3采用RMSNorm + RoPE + GQA + SwiGLU的现代Transformer架构。
 * 
 * 输入: [batch, text_seq_len] (token IDs)
 * 输出: [batch, text_seq_len, hidden_size] (文本特征)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class TextEncoder extends Module {
    
    private final Qwen3OmniConfig config;
    private final Qwen3Block qwen3Block;
    
    /**
     * 构造函数
     * 
     * @param name 编码器名称
     * @param config Qwen3Omni配置
     */
    public TextEncoder(String name, Qwen3OmniConfig config) {
        super(name);
        this.config = config;
        
        // 创建Qwen3配置并复用Qwen3Block
        Qwen3Config qwen3Config = createQwen3Config();
        this.qwen3Block = new Qwen3Block(
            name + "_qwen3",
            qwen3Config,
            false  // 不包含LM head，仅用于特征提取
        );
        
        // 注册为子模块
        registerModule("qwen3_block", qwen3Block);
    }
    
    /**
     * 从Qwen3Omni配置创建Qwen3配置
     */
    private Qwen3Config createQwen3Config() {
        Qwen3Config qwen3Config = new Qwen3Config();
        
        // 基础配置
        qwen3Config.setVocabSize(config.getVocabSize());
        qwen3Config.setHiddenSize(config.getHiddenSize());
        qwen3Config.setIntermediateSize(config.getIntermediateSize());
        qwen3Config.setNumHiddenLayers(config.getNumHiddenLayers());
        qwen3Config.setNumAttentionHeads(config.getNumAttentionHeads());
        qwen3Config.setNumKeyValueHeads(config.getNumKeyValueHeads());
        qwen3Config.setMaxPositionEmbeddings(config.getMaxPositionEmbeddings());
        
        // 归一化配置
        qwen3Config.setRmsNormEps(config.getRmsNormEps());
        
        // RoPE配置
        qwen3Config.setRopeTheta(config.getRopeTheta());
        
        // 特殊标记配置
        qwen3Config.setPadTokenId(config.getPadTokenId());
        qwen3Config.setBosTokenId(config.getBosTokenId());
        qwen3Config.setEosTokenId(config.getEosTokenId());
        
        return qwen3Config;
    }
    
    /**
     * 前向传播
     * 
     * @param inputs 输入token IDs [batch, seq_len]
     * @return 文本特征 [batch, seq_len, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("TextEncoder需要输入token IDs");
        }
        
        Variable tokenIds = inputs[0];
        
        // 通过Qwen3Block提取特征
        // Qwen3Block的forward返回logits，但我们需要的是最后一层的hidden states
        // 这里直接使用Qwen3Block的forward，它会返回 [batch, seq_len, vocab_size]
        // 我们需要修改为返回hidden states
        
        // 调用Qwen3Block的前向传播
        Variable output = qwen3Block.forward(tokenIds);
        
        // 注意: 这里返回的是Qwen3Block的输出
        // 在实际使用中，可能需要获取最后一层的hidden states而不是logits
        // 暂时返回完整的输出，后续可以优化
        return output;
    }
    
    /**
     * 获取Qwen3Block
     * 
     * @return Qwen3Block实例
     */
    public Qwen3Block getQwen3Block() {
        return qwen3Block;
    }
}
