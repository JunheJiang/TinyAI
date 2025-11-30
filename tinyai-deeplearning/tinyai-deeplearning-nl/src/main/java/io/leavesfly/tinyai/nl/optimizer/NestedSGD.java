package io.leavesfly.tinyai.nl.optimizer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nl.core.NestedOptimizationLevel;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 嵌套随机梯度下降优化器（NestedSGD）
 * 实现多层级的SGD优化
 * 
 * <p>每个嵌套层级使用独立的SGD更新，支持动量和权重衰减。
 * 不同层级可以有不同的学习率和更新频率。</p>
 * 
 * @author TinyAI Team
 */
public class NestedSGD extends DeepOptimizer {
    
    /**
     * 动量系数
     */
    private float momentum;
    
    /**
     * 权重衰减系数（L2正则化）
     */
    private float weightDecay;
    
    /**
     * 是否使用Nesterov动量
     */
    private boolean nesterov;
    
    /**
     * 动量缓存：存储每个参数的动量
     */
    private Map<Variable, Variable> velocities;
    
    /**
     * 构造函数
     * 
     * @param globalLearningRate 全局学习率
     * @param momentum 动量系数
     * @param weightDecay 权重衰减系数
     * @param nesterov 是否使用Nesterov动量
     */
    public NestedSGD(float globalLearningRate, float momentum, float weightDecay, boolean nesterov) {
        super(globalLearningRate);
        this.momentum = Math.max(0.0f, Math.min(1.0f, momentum));
        this.weightDecay = Math.max(0.0f, weightDecay);
        this.nesterov = nesterov;
        this.velocities = new HashMap<>();
    }
    
    /**
     * 简化构造函数（无动量）
     * 
     * @param globalLearningRate 全局学习率
     */
    public NestedSGD(float globalLearningRate) {
        this(globalLearningRate, 0.0f, 0.0f, false);
    }
    
    /**
     * 带动量的构造函数
     * 
     * @param globalLearningRate 全局学习率
     * @param momentum 动量系数
     */
    public NestedSGD(float globalLearningRate, float momentum) {
        this(globalLearningRate, momentum, 0.0f, false);
    }
    
    @Override
    protected void updateLevel(NestedOptimizationLevel level, List<Variable> gradients) {
        if (level == null || gradients == null || gradients.isEmpty()) {
            return;
        }
        
        List<Variable> parameters = level.getParameters();
        float levelLearningRate = level.getLearningRate();
        
        // 如果层级学习率为0，使用全局学习率
        if (levelLearningRate == 0.0f) {
            levelLearningRate = globalLearningRate;
        }
        
        // 更新每个参数
        int count = Math.min(parameters.size(), gradients.size());
        for (int i = 0; i < count; i++) {
            Variable param = parameters.get(i);
            Variable grad = gradients.get(i);
            
            if (param == null || grad == null) {
                continue;
            }
            
            // 应用权重衰减（L2正则化）
            if (weightDecay > 0.0f) {
                // grad = grad + weightDecay * param
                Variable decayTerm = param.mul(new Variable(weightDecay));
                grad = grad.add(decayTerm);
            }
            
            // 应用动量
            if (momentum > 0.0f) {
                grad = applyMomentum(param, grad, levelLearningRate);
            }
            
            // 执行参数更新：param = param - learningRate * grad
            Variable update = grad.mul(new Variable(levelLearningRate));
            Variable newParam = param.sub(update);
            
            // 更新参数
            parameters.set(i, newParam);
        }
    }
    
    /**
     * 应用动量
     * 
     * @param param 参数
     * @param grad 梯度
     * @param learningRate 学习率
     * @return 应用动量后的梯度
     */
    private Variable applyMomentum(Variable param, Variable grad, float learningRate) {
        // 获取或初始化速度
        Variable velocity = velocities.get(param);
        if (velocity == null) {
            // 初始化为零（简化：使用梯度乘以0）
            velocity = grad.mul(new Variable(0.0f));
        }
        
        if (nesterov) {
            // Nesterov动量：v = momentum * v + grad
            //              grad_new = grad + momentum * v
            velocity = velocity.mul(new Variable(momentum)).add(grad);
            Variable momentumTerm = velocity.mul(new Variable(momentum));
            grad = grad.add(momentumTerm);
        } else {
            // 标准动量：v = momentum * v + grad
            //          grad_new = v
            velocity = velocity.mul(new Variable(momentum)).add(grad);
            grad = velocity;
        }
        
        // 更新速度缓存
        velocities.put(param, velocity);
        
        return grad;
    }
    
    @Override
    public void reset() {
        super.reset();
        velocities.clear();
    }
    
    /**
     * 获取优化器配置信息
     * 
     * @return 配置字符串
     */
    public String getConfig() {
        StringBuilder sb = new StringBuilder();
        sb.append("NestedSGD配置:\n");
        sb.append(String.format("  全局学习率: %.6f\n", globalLearningRate));
        sb.append(String.format("  动量: %.4f\n", momentum));
        sb.append(String.format("  权重衰减: %.6f\n", weightDecay));
        sb.append(String.format("  Nesterov: %s\n", nesterov ? "是" : "否"));
        sb.append(String.format("  梯度裁剪: %s", enableGradientClipping ? 
            String.format("是 (阈值=%.2f)", gradientClipThreshold) : "否"));
        
        return sb.toString();
    }
    
    // Getters and Setters
    
    public float getMomentum() {
        return momentum;
    }
    
    public void setMomentum(float momentum) {
        this.momentum = Math.max(0.0f, Math.min(1.0f, momentum));
    }
    
    public float getWeightDecay() {
        return weightDecay;
    }
    
    public void setWeightDecay(float weightDecay) {
        this.weightDecay = Math.max(0.0f, weightDecay);
    }
    
    public boolean isNesterov() {
        return nesterov;
    }
    
    public void setNesterov(boolean nesterov) {
        this.nesterov = nesterov;
    }
    
    public Map<Variable, Variable> getVelocities() {
        return new HashMap<>(velocities);
    }
}
