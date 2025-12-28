package io.leavesfly.tinyai.ml.loss;

import io.leavesfly.tinyai.func.Variable;

/**
 * 均方误差损失函数
 * <p>
 * 用于回归任务的损失计算，计算预测值与真实值之间的均方误差。
 * 支持PyTorch风格的reduction参数。
 *
 * @author TinyDL
 * @version 2.0
 */
public class MeanSquaredLoss extends Loss {
    
    /**
     * 默认构造函数(使用MEAN归约)
     */
    public MeanSquaredLoss() {
        super(Reduction.MEAN);
    }
    
    /**
     * 带归约方式的构造函数
     *
     * @param reduction 归约方式
     */
    public MeanSquaredLoss(Reduction reduction) {
        super(reduction);
    }

    @Override
    public Variable loss(Variable y, Variable predict) {
        // 计算平方误差: (predict - y)^2
        Variable diff = predict.sub(y);
        Variable squaredError = diff.mul(diff);
        
        // 应用归约
        return applyReduction(squaredError);
    }
}