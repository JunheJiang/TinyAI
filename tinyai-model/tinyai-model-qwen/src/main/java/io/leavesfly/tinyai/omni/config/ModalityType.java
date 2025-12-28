package io.leavesfly.tinyai.omni.config;

/**
 * 模态类型枚举
 * 
 * 定义Qwen3-Omni支持的模态类型：
 * - TEXT: 文本模态
 * - IMAGE: 图像模态
 * - AUDIO: 音频模态
 * 
 * @author leavesfly
 * @version 1.0
 */
public enum ModalityType {
    
    /**
     * 文本模态
     */
    TEXT("text", "文本"),
    
    /**
     * 图像模态
     */
    IMAGE("image", "图像"),
    
    /**
     * 音频模态
     */
    AUDIO("audio", "音频");
    
    private final String code;
    private final String description;
    
    ModalityType(String code, String description) {
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
