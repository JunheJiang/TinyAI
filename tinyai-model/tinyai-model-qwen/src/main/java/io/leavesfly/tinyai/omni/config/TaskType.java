package io.leavesfly.tinyai.omni.config;

/**
 * 任务类型枚举
 * 
 * 定义Qwen3-Omni支持的多模态任务类型
 * 
 * @author leavesfly
 * @version 1.0
 */
public enum TaskType {
    
    /**
     * 多模态理解任务
     */
    UNDERSTANDING("understanding", "多模态理解"),
    
    /**
     * 文本生成任务
     */
    TEXT_GENERATION("text_generation", "文本生成"),
    
    /**
     * 图像生成任务
     */
    IMAGE_GENERATION("image_generation", "图像生成"),
    
    /**
     * 音频生成任务
     */
    AUDIO_GENERATION("audio_generation", "音频生成"),
    
    /**
     * 图像描述任务
     */
    IMAGE_CAPTIONING("image_captioning", "图像描述"),
    
    /**
     * 音频理解任务
     */
    AUDIO_UNDERSTANDING("audio_understanding", "音频理解"),
    
    /**
     * 跨模态检索任务
     */
    CROSS_MODAL_RETRIEVAL("cross_modal_retrieval", "跨模态检索");
    
    private final String code;
    private final String description;
    
    TaskType(String code, String description) {
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
