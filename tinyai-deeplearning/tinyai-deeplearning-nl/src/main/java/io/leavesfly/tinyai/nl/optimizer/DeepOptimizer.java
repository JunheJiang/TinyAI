package io.leavesfly.tinyai.nl.optimizer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nl.core.NestedOptimizationLevel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 深度优化器（DeepOptimizer）
 * 支持多层级嵌套优化的优化器基类
 * 
 * <p>与传统优化器不同，深度优化器管理多个嵌套的优化层级，
 * 每个层级有自己的学习率和更新频率。这允许模型的不同部分
 * 以不同的速度学习。</p>
 * 
 * @author TinyAI Team
 */
public abstract class DeepOptimizer {
    
    /**
     * 优化层级列表
     */
    protected List<NestedOptimizationLevel> levels;
    
    /**
     * 全局学习率
     */
    protected float globalLearningRate;
    
    /**
     * 当前训练步骤
     */
    protected int currentStep;
    
    /**
     * 参数到层级的映射
     */
    protected Map<Variable, NestedOptimizationLevel> parameterToLevel;
    
    /**
     * 是否启用梯度裁剪
     */
    protected boolean enableGradientClipping;
    
    /**
     * 梯度裁剪阈值
     */
    protected float gradientClipThreshold;
    
    /**
     * 构造函数
     * 
     * @param globalLearningRate 全局学习率
     */
    public DeepOptimizer(float globalLearningRate) {
        this.globalLearningRate = globalLearningRate;
        this.levels = new ArrayList<>();
        this.parameterToLevel = new HashMap<>();
        this.currentStep = 0;
        this.enableGradientClipping = false;
        this.gradientClipThreshold = 5.0f;
    }
    
    /**
     * 添加优化层级
     * 
     * @param level 优化层级
     */
    public void addLevel(NestedOptimizationLevel level) {
        if (level != null && !levels.contains(level)) {
            levels.add(level);
            
            // 将层级的参数映射到该层级
            for (Variable param : level.getParameters()) {
                parameterToLevel.put(param, level);
            }
        }
    }
    
    /**
     * 执行一步优化
     * 根据各层级的更新频率决定是否更新
     * 
     * @param gradients 所有参数的梯度
     */
    public void step(Map<Variable, Variable> gradients) {
        if (gradients == null || gradients.isEmpty()) {
            return;
        }
        
        // 增加步数
        currentStep++;
        
        // 遍历所有层级
        for (NestedOptimizationLevel level : levels) {
            // 检查该层级是否应该在当前步骤更新
            if (level.shouldUpdate(currentStep)) {
                // 收集该层级的梯度
                List<Variable> levelGradients = new ArrayList<>();
                
                for (Variable param : level.getParameters()) {
                    Variable grad = gradients.get(param);
                    if (grad != null) {
                        // 应用梯度裁剪
                        if (enableGradientClipping) {
                            grad = clipGradient(grad);
                        }
                        levelGradients.add(grad);
                    }
                }
                
                // 执行层级特定的更新
                updateLevel(level, levelGradients);
                
                // 更新最后更新步骤
                level.setLastUpdateStep(currentStep);
            }
        }
    }
    
    /**
     * 更新特定层级的参数
     * 子类需要实现具体的更新逻辑
     * 
     * @param level 优化层级
     * @param gradients 梯度列表
     */
    protected abstract void updateLevel(NestedOptimizationLevel level, List<Variable> gradients);
    
    /**
     * 梯度裁剪
     * 防止梯度爆炸
     * 
     * @param gradient 原始梯度
     * @return 裁剪后的梯度
     */
    protected Variable clipGradient(Variable gradient) {
        if (gradient == null) {
            return null;
        }
        
        // 简化实现：基于L2范数裁剪
        // 计算梯度的L2范数
        Variable squared = gradient.mul(gradient);
        Variable sum = squared.sum();
        
        float norm = (float) Math.sqrt(sum.getValue().get(new int[]{0}));
        
        // 如果范数超过阈值，进行缩放
        if (norm > gradientClipThreshold) {
            float scale = gradientClipThreshold / norm;
            return gradient.mul(new Variable(scale));
        }
        
        return gradient;
    }
    
    /**
     * 计算元梯度
     * 元梯度用于优化优化器本身的超参数
     * 
     * @param level 优化层级
     * @param loss 损失值
     * @return 元梯度
     */
    protected Variable computeMetaGradient(NestedOptimizationLevel level, Variable loss) {
        // 简化实现：返回损失对学习率的导数近似
        // 实际应该使用更复杂的元学习算法
        if (loss == null) {
            return null;
        }
        
        // 元梯度 ≈ ∂L/∂η (损失对学习率的导数)
        // 这里简化为损失本身
        return loss;
    }
    
    /**
     * 重置优化器状态
     */
    public void reset() {
        currentStep = 0;
        // 子类可以重写以重置额外的状态
    }
    
    /**
     * 零化所有梯度
     */
    public void zeroGrad() {
        for (NestedOptimizationLevel level : levels) {
            for (Variable param : level.getParameters()) {
                // 清空梯度
                if (param.getGrad() != null) {
                    param.setGrad(null);
                }
            }
        }
    }
    
    /**
     * 获取优化器状态信息
     * 
     * @return 状态字符串
     */
    public String getStateInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("优化器状态 (步数: %d):\n", currentStep));
        sb.append(String.format("  全局学习率: %.6f\n", globalLearningRate));
        sb.append(String.format("  层级数量: %d\n", levels.size()));
        
        for (int i = 0; i < levels.size(); i++) {
            NestedOptimizationLevel level = levels.get(i);
            sb.append(String.format("  层级 %d: 频率=%.4f, 学习率=%.6f, 参数数=%d\n",
                i,
                level.getUpdateFrequency(),
                level.getLearningRate(),
                level.getParameters().size()));
        }
        
        return sb.toString();
    }
    
    // Getters and Setters
    
    public float getGlobalLearningRate() {
        return globalLearningRate;
    }
    
    public void setGlobalLearningRate(float globalLearningRate) {
        this.globalLearningRate = Math.max(0.0f, globalLearningRate);
    }
    
    public int getCurrentStep() {
        return currentStep;
    }
    
    public List<NestedOptimizationLevel> getLevels() {
        return new ArrayList<>(levels);
    }
    
    public boolean isEnableGradientClipping() {
        return enableGradientClipping;
    }
    
    public void setEnableGradientClipping(boolean enableGradientClipping) {
        this.enableGradientClipping = enableGradientClipping;
    }
    
    public float getGradientClipThreshold() {
        return gradientClipThreshold;
    }
    
    public void setGradientClipThreshold(float gradientClipThreshold) {
        this.gradientClipThreshold = Math.max(0.0f, gradientClipThreshold);
    }
}
