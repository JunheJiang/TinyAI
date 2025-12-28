package io.leavesfly.tinyai.omni.config;

/**
 * 多模态融合策略枚举
 * 
 * 定义不同的模态融合方式：
 * - EARLY_FUSION: 早期融合，在特征层直接拼接
 * - LATE_FUSION: 后期融合，独立编码后通过注意力融合
 * - HYBRID_FUSION: 混合融合，结合早期和后期融合优势
 * 
 * @author leavesfly
 * @version 1.0
 */
public enum FusionStrategy {
    
    /**
     * 早期融合策略
     * 在特征提取后立即拼接所有模态特征
     */
    EARLY_FUSION("early", "早期融合"),
    
    /**
     * 后期融合策略
     * 各模态独立编码，通过跨模态注意力融合
     */
    LATE_FUSION("late", "后期融合"),
    
    /**
     * 混合融合策略（推荐）
     * 浅层使用早期融合，深层使用跨模态注意力
     */
    HYBRID_FUSION("hybrid", "混合融合");
    
    private final String code;
    private final String description;
    
    FusionStrategy(String code, String description) {
        this.code = code;
        this.description = description;
    }
    
    public String getCode() {
        return code;
    }
    
    public String getDescription() {
        return description;
    }
    
    @Override
    public String toString() {
        return code + "(" + description + ")";
    }
}
