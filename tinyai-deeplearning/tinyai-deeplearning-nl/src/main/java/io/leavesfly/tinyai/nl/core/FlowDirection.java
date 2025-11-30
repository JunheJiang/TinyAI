package io.leavesfly.tinyai.nl.core;

/**
 * 上下文流动方向枚举
 * 定义了嵌套学习中上下文信息的流动方向
 * 
 * @author TinyAI Team
 */
public enum FlowDirection {
    /**
     * 向上流动 - 从子层级传播到父层级
     */
    UPWARD,
    
    /**
     * 向下流动 - 从父层级传播到子层级
     */
    DOWNWARD,
    
    /**
     * 双向流动 - 同时支持向上和向下传播
     */
    BIDIRECTIONAL
}
