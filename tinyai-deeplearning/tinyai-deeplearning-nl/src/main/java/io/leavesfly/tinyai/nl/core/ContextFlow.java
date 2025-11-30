package io.leavesfly.tinyai.nl.core;

import io.leavesfly.tinyai.func.Variable;

/**
 * 上下文流（ContextFlow）
 * 管理嵌套优化层级之间的信息流动
 * 
 * <p>在嵌入学习范式中，不同层级的优化问题通过上下文流交换信息。
 * 上下文流负责在层级间传播、压缩和合并上下文数据。</p>
 * 
 * @author TinyAI Team
 */
public class ContextFlow {
    
    /**
     * 当前上下文数据
     */
    private Variable contextData;
    
    /**
     * 流动方向
     */
    private FlowDirection flowDirection;
    
    /**
     * 上下文压缩率（0-1之间，1表示不压缩）
     */
    private float compressionRate;
    
    /**
     * 构造函数
     * 
     * @param contextData 初始上下文数据
     * @param flowDirection 流动方向
     * @param compressionRate 压缩率
     */
    public ContextFlow(Variable contextData, FlowDirection flowDirection, float compressionRate) {
        this.contextData = contextData;
        this.flowDirection = flowDirection;
        this.compressionRate = Math.max(0.0f, Math.min(1.0f, compressionRate));
    }
    
    /**
     * 简化构造函数，默认双向流动，不压缩
     * 
     * @param contextData 初始上下文数据
     */
    public ContextFlow(Variable contextData) {
        this(contextData, FlowDirection.BIDIRECTIONAL, 1.0f);
    }
    
    /**
     * 执行上下文流动
     * 根据流动方向和压缩率处理输入上下文
     * 
     * @param inputContext 输入上下文
     * @return 处理后的上下文
     */
    public Variable flow(Variable inputContext) {
        if (inputContext == null) {
            return this.contextData;
        }
        
        // 如果需要压缩，应用压缩
        Variable processedContext = inputContext;
        if (compressionRate < 1.0f) {
            processedContext = compress(inputContext, compressionRate);
        }
        
        // 更新当前上下文
        this.contextData = processedContext;
        
        return processedContext;
    }
    
    /**
     * 压缩上下文信息
     * 使用线性投影降低维度，保留最重要的特征
     * 
     * @param context 原始上下文
     * @param rate 压缩率
     * @return 压缩后的上下文
     */
    public Variable compress(Variable context, float rate) {
        if (context == null || rate >= 1.0f) {
            return context;
        }
        
        // 简化实现：通过缩放模拟压缩
        // 在实际应用中，可以使用线性层进行维度降低
        int[] shape = context.getValue().getShape().getShapeDims();
        if (shape.length == 2) {
            // 对于2D张量，压缩第二维
            int newDim = Math.max(1, (int)(shape[1] * rate));
            
            // 这里简化处理：返回前newDim个特征
            // 实际应用中应该使用可学习的压缩层
            if (newDim < shape[1]) {
                // 创建一个选择前newDim个特征的变量
                // 简化：直接返回原context，实际需要切片操作
                return context;
            }
        }
        
        return context;
    }
    
    /**
     * 合并多个上下文流
     * 将另一个上下文流的信息合并到当前流中
     * 
     * @param otherContext 其他上下文流
     * @return 合并后的新上下文流
     */
    public ContextFlow merge(ContextFlow otherContext) {
        if (otherContext == null) {
            return this;
        }
        
        // 简化实现：取平均
        Variable mergedData = this.contextData;
        if (otherContext.contextData != null) {
            // 如果两个上下文数据都存在，进行平均
            if (this.contextData != null) {
                mergedData = this.contextData.add(otherContext.contextData).mul(new Variable(0.5f));
            } else {
                mergedData = otherContext.contextData;
            }
        }
        
        // 创建新的上下文流
        return new ContextFlow(mergedData, this.flowDirection, this.compressionRate);
    }
    
    // Getters and Setters
    
    public Variable getContextData() {
        return contextData;
    }
    
    public void setContextData(Variable contextData) {
        this.contextData = contextData;
    }
    
    public FlowDirection getFlowDirection() {
        return flowDirection;
    }
    
    public void setFlowDirection(FlowDirection flowDirection) {
        this.flowDirection = flowDirection;
    }
    
    public float getCompressionRate() {
        return compressionRate;
    }
    
    public void setCompressionRate(float compressionRate) {
        this.compressionRate = Math.max(0.0f, Math.min(1.0f, compressionRate));
    }
}
