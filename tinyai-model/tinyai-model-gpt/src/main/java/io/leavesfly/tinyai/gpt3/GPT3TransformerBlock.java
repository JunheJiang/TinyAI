package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.GELU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;
import io.leavesfly.tinyai.nnet.v2.layer.transformer.MultiHeadAttention;

/**
 * GPT-3 Transformer块实现（使用V2 API）
 * 
 * GPT-3的核心创新：并行注意力和前馈网络计算
 * 与GPT-2的串行架构不同，GPT-3同时计算注意力和MLP，然后合并结果
 * 
 * 并行架构流程：
 * <pre>
 * input -> split
 *           |-> LayerNorm1 -> MultiHeadAttention -> attn_output
 *           |-> LayerNorm2 -> FeedForward -> mlp_output
 *          merge -> input + attn_output + mlp_output -> output
 * </pre>
 * 
 * 对比GPT-2串行架构：
 * <pre>
 * input -> LayerNorm1 -> Attention -> Add(input)
 *       -> LayerNorm2 -> FeedForward -> Add -> output
 * </pre>
 * 
 * 优势：
 * 1. 提升计算效率：注意力和MLP可并行计算
 * 2. 更好的梯度流动
 * 3. 充分利用现代硬件的并行能力
 * 
 * @author leavesfly
 * @version 1.0
 */
public class GPT3TransformerBlock extends Module {
    
    private final GPT3Config config;
    
    // 注意力分支
    private final LayerNorm layerNorm1;           // 注意力分支的LayerNorm
    private final MultiHeadAttention attention;    // 多头自注意力
    private final Dropout attnDropout;            // 注意力输出dropout
    
    // 前馈分支
    private final LayerNorm layerNorm2;           // MLP分支的LayerNorm
    private final Linear ffnLinear1;              // 第一个线性层
    private final GELU activation;                // GELU激活函数
    private final Linear ffnLinear2;              // 第二个线性层
    private final Dropout mlpDropout;             // MLP输出dropout
    
    /**
     * 构造GPT-3 Transformer块
     * 
     * @param name 块名称
     * @param config GPT-3配置
     */
    public GPT3TransformerBlock(String name, GPT3Config config) {
        super(name);
        this.config = config;
        
        int dModel = config.getNEmbd();
        int numHeads = config.getNHead();
        int dFF = config.getNInner();
        float dropout = (float) config.getResidPdrop();
        float attnDropoutRate = (float) config.getAttnPdrop();
        
        // 初始化注意力分支
        this.layerNorm1 = new LayerNorm("ln1", dModel, (float) config.getLayerNormEpsilon());
        this.attention = new MultiHeadAttention("attn", dModel, numHeads, attnDropoutRate);
        this.attnDropout = new Dropout("attn_dropout", dropout);
        
        // 初始化前馈分支
        this.layerNorm2 = new LayerNorm("ln2", dModel, (float) config.getLayerNormEpsilon());
        this.ffnLinear1 = new Linear("ffn_fc1", dModel, dFF, true);
        this.activation = new GELU("gelu");
        this.ffnLinear2 = new Linear("ffn_fc2", dFF, dModel, true);
        this.mlpDropout = new Dropout("mlp_dropout", dropout);
        
        // 注册所有子模块
        registerModule("ln1", layerNorm1);
        registerModule("attn", attention);
        registerModule("attn_dropout", attnDropout);
        registerModule("ln2", layerNorm2);
        registerModule("ffn_fc1", ffnLinear1);
        registerModule("gelu", activation);
        registerModule("ffn_fc2", ffnLinear2);
        registerModule("mlp_dropout", mlpDropout);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];  // (batch_size, seq_len, n_embd)
        
        // 获取序列长度并生成因果掩码
        int seqLen = x.getValue().getShape().getDimension(1);
        Variable causalMask = MultiHeadAttention.generateCausalMaskBatched(seqLen);
        
        if (config.isParallelAttention()) {
            // GPT-3并行架构
            return forwardParallel(x, causalMask);
        } else {
            // 回退到GPT-2串行架构（兼容模式）
            return forwardSequential(x, causalMask);
        }
    }
    
    /**
     * GPT-3并行前向传播
     * 同时计算注意力和MLP，然后合并
     * 
     * @param x 输入变量
     * @param causalMask 因果掩码
     * @return 输出变量
     */
    private Variable forwardParallel(Variable x, Variable causalMask) {
        // 注意力分支：LayerNorm -> Attention
        Variable attnInput = layerNorm1.forward(x);
        Variable attnOutput = attention.forward(attnInput, attnInput, attnInput, causalMask, null);
        attnOutput = attnDropout.forward(attnOutput);
        
        // MLP分支：LayerNorm -> Linear -> GELU -> Linear
        Variable mlpInput = layerNorm2.forward(x);
        Variable mlpOutput = ffnLinear1.forward(mlpInput);
        mlpOutput = activation.forward(mlpOutput);
        mlpOutput = ffnLinear2.forward(mlpOutput);
        mlpOutput = mlpDropout.forward(mlpOutput);
        
        // 合并：input + attn_output + mlp_output
        Variable output = x.add(attnOutput).add(mlpOutput);
        
        return output;
    }
    
    /**
     * GPT-2风格的串行前向传播（兼容模式）
     * 
     * @param x 输入变量
     * @param causalMask 因果掩码
     * @return 输出变量
     */
    private Variable forwardSequential(Variable x, Variable causalMask) {
        // 第一个子层：LayerNorm -> Attention -> Residual
        Variable normalized1 = layerNorm1.forward(x);
        Variable attnOutput = attention.forward(normalized1, normalized1, normalized1, causalMask, null);
        attnOutput = attnDropout.forward(attnOutput);
        Variable residual1 = x.add(attnOutput);
        
        // 第二个子层：LayerNorm -> MLP -> Residual
        Variable normalized2 = layerNorm2.forward(residual1);
        Variable mlpOutput = ffnLinear1.forward(normalized2);
        mlpOutput = activation.forward(mlpOutput);
        mlpOutput = ffnLinear2.forward(mlpOutput);
        mlpOutput = mlpDropout.forward(mlpOutput);
        Variable output = residual1.add(mlpOutput);
        
        return output;
    }
    
    // ==================== Getter方法 ====================
    
    public LayerNorm getLayerNorm1() {
        return layerNorm1;
    }
    
    public MultiHeadAttention getAttention() {
        return attention;
    }
    
    public LayerNorm getLayerNorm2() {
        return layerNorm2;
    }
    
    public Linear getFfnLinear1() {
        return ffnLinear1;
    }
    
    public GELU getActivation() {
        return activation;
    }
    
    public Linear getFfnLinear2() {
        return ffnLinear2;
    }
    
    public GPT3Config getConfig() {
        return config;
    }
    
    @Override
    public String toString() {
        String mode = config.isParallelAttention() ? "Parallel" : "Sequential";
        return String.format("GPT3TransformerBlock{name='%s', mode=%s, dModel=%d, numHeads=%d, dFF=%d}",
                name, mode, config.getNEmbd(), config.getNHead(), config.getNInner());
    }
}
