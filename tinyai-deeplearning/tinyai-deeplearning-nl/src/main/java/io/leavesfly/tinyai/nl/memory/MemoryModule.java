package io.leavesfly.tinyai.nl.memory;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nl.core.AssociativeMemory;

/**
 * 记忆模块（MemoryModule）
 * 封装记忆存储和检索的基本功能
 * 
 * <p>记忆模块是嵌套学习中的基础组件，负责管理特定类型的记忆，
 * 包括存储、检索、更新和遗忘机制。</p>
 * 
 * @author TinyAI Team
 */
public class MemoryModule {
    
    /**
     * 记忆类型
     */
    private MemoryType memoryType;
    
    /**
     * 关联记忆实现
     */
    private AssociativeMemory memory;
    
    /**
     * 更新频率（继承自记忆类型）
     */
    private float updateFrequency;
    
    /**
     * 遗忘率（控制记忆衰减速度）
     */
    private float forgettingRate;
    
    /**
     * 上次更新的步骤
     */
    private int lastUpdateStep;
    
    /**
     * 是否启用遗忘机制
     */
    private boolean enableForgetting;
    
    /**
     * 构造函数
     * 
     * @param memoryType 记忆类型
     * @param memorySize 记忆容量
     * @param forgettingRate 遗忘率
     */
    public MemoryModule(MemoryType memoryType, int memorySize, float forgettingRate) {
        this.memoryType = memoryType;
        this.updateFrequency = memoryType.getUpdateFrequency();
        this.forgettingRate = forgettingRate;
        this.memory = new AssociativeMemory(memorySize);
        this.lastUpdateStep = -1;
        this.enableForgetting = true;
    }
    
    /**
     * 简化构造函数
     * 
     * @param memoryType 记忆类型
     * @param memorySize 记忆容量
     */
    public MemoryModule(MemoryType memoryType, int memorySize) {
        this(memoryType, memorySize, 0.01f);
    }
    
    /**
     * 存储记忆
     * 
     * @param key 记忆键
     * @param value 记忆值
     */
    public void store(Variable key, Variable value) {
        if (key == null || value == null) {
            return;
        }
        
        memory.store(key, value);
    }
    
    /**
     * 检索记忆
     * 
     * @param queryKey 查询键
     * @return 检索到的记忆值
     */
    public Variable retrieve(Variable queryKey) {
        if (queryKey == null) {
            return null;
        }
        
        return memory.retrieve(queryKey);
    }
    
    /**
     * 更新记忆模块
     * 包括遗忘机制和记忆整合
     * 
     * @param currentStep 当前步骤
     */
    public void update(int currentStep) {
        // 检查是否需要更新
        if (!shouldUpdate(currentStep)) {
            return;
        }
        
        // 应用遗忘机制
        if (enableForgetting) {
            applyForgetting();
        }
        
        // 更新最后更新步骤
        lastUpdateStep = currentStep;
    }
    
    /**
     * 判断是否应该更新
     * 
     * @param currentStep 当前步骤
     * @return 是否应该更新
     */
    public boolean shouldUpdate(int currentStep) {
        if (currentStep <= 0 || lastUpdateStep < 0) {
            return true;
        }
        
        int updateInterval = (int)(1.0f / updateFrequency);
        return (currentStep % updateInterval == 0);
    }
    
    /**
     * 应用遗忘机制
     * 基于惊异度阈值修剪低优先级记忆
     */
    private void applyForgetting() {
        // 计算遗忘阈值
        float threshold = memory.getSurpriseThreshold() * forgettingRate;
        
        // 执行记忆修剪
        memory.prune(threshold);
    }
    
    /**
     * 计算记忆惊异度
     * 
     * @param input 输入数据
     * @return 惊异度分数
     */
    public float computeSurprise(Variable input) {
        return memory.computeSurprise(input);
    }
    
    /**
     * 清空所有记忆
     */
    public void clear() {
        memory.clear();
        lastUpdateStep = -1;
    }
    
    /**
     * 获取记忆大小
     * 
     * @return 当前存储的记忆数量
     */
    public int getSize() {
        return memory.getCurrentSize();
    }
    
    /**
     * 获取记忆容量
     * 
     * @return 最大容量
     */
    public int getCapacity() {
        return memory.getMemorySize();
    }
    
    // Getters and Setters
    
    public MemoryType getMemoryType() {
        return memoryType;
    }
    
    public void setMemoryType(MemoryType memoryType) {
        this.memoryType = memoryType;
        this.updateFrequency = memoryType.getUpdateFrequency();
    }
    
    public float getUpdateFrequency() {
        return updateFrequency;
    }
    
    public float getForgettingRate() {
        return forgettingRate;
    }
    
    public void setForgettingRate(float forgettingRate) {
        this.forgettingRate = Math.max(0.0f, Math.min(1.0f, forgettingRate));
    }
    
    public boolean isEnableForgetting() {
        return enableForgetting;
    }
    
    public void setEnableForgetting(boolean enableForgetting) {
        this.enableForgetting = enableForgetting;
    }
    
    public int getLastUpdateStep() {
        return lastUpdateStep;
    }
    
    public AssociativeMemory getMemory() {
        return memory;
    }
}
