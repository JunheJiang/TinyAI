package io.leavesfly.tinyai.ml.loss;

import io.leavesfly.tinyai.func.Variable;

/**
 * 损失函数抽象类
 * <p>
 * 该类是所有损失函数实现的基类，定义了计算损失值的基本接口。
 * 子类需要实现具体的损失计算逻辑。
 * 
 * <p>V2增强功能:
 * <ul>
 *   <li>支持PyTorch风格的reduction参数(NONE, MEAN, SUM)</li>
 *   <li>支持类别权重(weight)</li>
 *   <li>支持忽略索引(ignore_index)</li>
 * </ul>
 *
 * @author TinyDL
 * @version 2.0
 */
public abstract class Loss {
    
    /**
     * Reduction枚举 - 定义损失值的归约方式
     * <p>
     * 与PyTorch的reduction参数保持一致
     */
    public enum Reduction {
        /**
         * 不进行归约,返回每个样本的损失值
         */
        NONE,
        
        /**
         * 计算所有样本损失的平均值(默认)
         */
        MEAN,
        
        /**
         * 计算所有样本损失的总和
         */
        SUM
    }
    
    /**
     * 归约方式
     */
    protected Reduction reduction;
    
    /**
     * 类别权重(可选)
     */
    protected Variable weight;
    
    /**
     * 忽略的索引值(用于分类任务,默认-100)
     */
    protected int ignoreIndex = -100;
    
    /**
     * 默认构造函数(使用MEAN归约)
     */
    public Loss() {
        this(Reduction.MEAN);
    }
    
    /**
     * 带归约方式的构造函数
     *
     * @param reduction 归约方式
     */
    public Loss(Reduction reduction) {
        this.reduction = reduction;
    }
    
    /**
     * 完整参数构造函数
     *
     * @param reduction   归约方式
     * @param weight      类别权重
     * @param ignoreIndex 忽略的索引值
     */
    public Loss(Reduction reduction, Variable weight, int ignoreIndex) {
        this.reduction = reduction;
        this.weight = weight;
        this.ignoreIndex = ignoreIndex;
    }
    
    /**
     * 计算损失值
     *
     * @param y       真实标签
     * @param predict 预测值
     * @return 损失值变量
     */
    public abstract Variable loss(Variable y, Variable predict);
    
    /**
     * 应用归约操作
     * <p>
     * 根据reduction参数对损失值进行归约
     *
     * @param loss 原始损失值
     * @return 归约后的损失值
     */
    protected Variable applyReduction(Variable loss) {
        switch (reduction) {
            case MEAN:
                return loss.mean(-1, false).mean(-1, false);  // 对所有维度求平均
            case SUM:
                return loss.sum();  // 对所有元素求和
            case NONE:
            default:
                return loss;  // 不归约,返回原始值
        }
    }
    
    /**
     * 设置归约方式
     *
     * @param reduction 归约方式
     * @return 当前Loss实例(支持链式调用)
     */
    public Loss setReduction(Reduction reduction) {
        this.reduction = reduction;
        return this;
    }
    
    /**
     * 获取归约方式
     *
     * @return 归约方式
     */
    public Reduction getReduction() {
        return reduction;
    }
    
    /**
     * 设置类别权重
     *
     * @param weight 类别权重
     * @return 当前Loss实例(支持链式调用)
     */
    public Loss setWeight(Variable weight) {
        this.weight = weight;
        return this;
    }
    
    /**
     * 获取类别权重
     *
     * @return 类别权重
     */
    public Variable getWeight() {
        return weight;
    }
    
    /**
     * 设置忽略索引
     *
     * @param ignoreIndex 忽略的索引值
     * @return 当前Loss实例(支持链式调用)
     */
    public Loss setIgnoreIndex(int ignoreIndex) {
        this.ignoreIndex = ignoreIndex;
        return this;
    }
    
    /**
     * 获取忽略索引
     *
     * @return 忽略的索引值
     */
    public int getIgnoreIndex() {
        return ignoreIndex;
    }
}