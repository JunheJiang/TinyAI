package io.leavesfly.tinyai.omni.alignment;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * 模态对齐投影基类
 * 
 * 将不同模态的特征投影到统一的隐藏空间,实现模态对齐。
 * 
 * 架构: 
 * Input [batch, seq_len, source_dim]
 *   ↓ Linear
 * [batch, seq_len, target_dim]
 *   ↓ LayerNorm
 * Output [batch, seq_len, target_dim]
 * 
 * @author leavesfly
 * @version 1.0
 */
public abstract class ModalityAlignment extends Module {
    
    protected final int sourceDim;
    protected final int targetDim;
    
    protected final Linear projection;
    protected final LayerNorm layerNorm;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param sourceDim 源维度
     * @param targetDim 目标维度
     */
    public ModalityAlignment(String name, int sourceDim, int targetDim) {
        super(name);
        this.sourceDim = sourceDim;
        this.targetDim = targetDim;
        
        // 线性投影层
        this.projection = new Linear(
            name + "_proj",
            sourceDim,
            targetDim,
            true  // 使用bias
        );
        registerModule("projection", projection);
        
        // LayerNorm归一化
        this.layerNorm = new LayerNorm(
            name + "_norm",
            targetDim,
            1e-6f
        );
        registerModule("layer_norm", layerNorm);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块自行初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为源特征 [batch, seq_len, source_dim]
     * @return 对齐后的特征 [batch, seq_len, target_dim]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("ModalityAlignment需要输入特征");
        }
        
        Variable sourceFeatures = inputs[0];
        
        // 线性投影
        Variable projected = projection.forward(sourceFeatures);
        
        // LayerNorm归一化
        Variable aligned = layerNorm.forward(projected);
        
        return aligned;
    }
    
    public int getSourceDim() {
        return sourceDim;
    }
    
    public int getTargetDim() {
        return targetDim;
    }
}
