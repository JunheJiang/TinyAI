package io.leavesfly.tinyai.nl.memory;

/**
 * 记忆类型枚举
 * 定义了嵌套学习中不同类型的记忆系统
 * 
 * <p>基于连续体记忆理论，记忆可以分为不同的时间尺度类型，
 * 从短期记忆到长期记忆。</p>
 * 
 * @author TinyAI Team
 */
public enum MemoryType {
    /**
     * 短期记忆（高频率更新）
     * 持续时间：秒到分钟
     * 更新频率：1.0（每步更新）
     */
    SHORT_TERM(1.0f, "短期记忆"),
    
    /**
     * 中期记忆（中频率更新）
     * 持续时间：分钟到小时
     * 更新频率：0.1（每10步更新）
     */
    MEDIUM_TERM(0.1f, "中期记忆"),
    
    /**
     * 长期记忆（低频率更新）
     * 持续时间：小时到天
     * 更新频率：0.01（每100步更新）
     */
    LONG_TERM(0.01f, "长期记忆"),
    
    /**
     * 超长期记忆（极低频率更新）
     * 持续时间：天到永久
     * 更新频率：0.001（每1000步更新）
     */
    ULTRA_LONG_TERM(0.001f, "超长期记忆");
    
    /**
     * 该类型记忆的更新频率
     */
    private final float updateFrequency;
    
    /**
     * 记忆类型的描述
     */
    private final String description;
    
    /**
     * 构造函数
     * 
     * @param updateFrequency 更新频率
     * @param description 描述
     */
    MemoryType(float updateFrequency, String description) {
        this.updateFrequency = updateFrequency;
        this.description = description;
    }
    
    /**
     * 获取更新频率
     * 
     * @return 更新频率
     */
    public float getUpdateFrequency() {
        return updateFrequency;
    }
    
    /**
     * 获取描述
     * 
     * @return 描述
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * 根据更新频率获取记忆类型
     * 
     * @param frequency 更新频率
     * @return 最接近的记忆类型
     */
    public static MemoryType fromFrequency(float frequency) {
        MemoryType closest = SHORT_TERM;
        float minDiff = Float.MAX_VALUE;
        
        for (MemoryType type : values()) {
            float diff = Math.abs(type.updateFrequency - frequency);
            if (diff < minDiff) {
                minDiff = diff;
                closest = type;
            }
        }
        
        return closest;
    }
}
