package io.leavesfly.tinyai.nnet.v2.init;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 正交初始化器
 * <p>
 * 生成正交矩阵，常用于RNN的权重初始化
 * <p>
 * 注意：当前实现为简化版本，使用Xavier正态初始化代替真正的正交矩阵分解。
 * 完整的正交初始化需要SVD或QR分解，这在纯Java环境中实现较为复杂。
 * <p>
 * 参考论文：
 * Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
 * by Saxe et al. (2013)
 *
 * @author leavesfly
 * @version 2.0
 */
public class OrthogonalInitializer implements Initializer {

    private final float gain;

    /**
     * 构造函数
     *
     * @param gain 增益系数（默认为1.0）
     */
    public OrthogonalInitializer(float gain) {
        this.gain = gain;
    }

    /**
     * 默认构造函数（gain=1.0）
     */
    public OrthogonalInitializer() {
        this(1.0f);
    }

    @Override
    public void initialize(NdArray tensor) {
        // 简化实现：使用Xavier正态初始化代替
        // TODO: 实现真正的正交初始化（需要QR分解或SVD）
        new XavierNormalInitializer(gain).initialize(tensor);
    }
}
