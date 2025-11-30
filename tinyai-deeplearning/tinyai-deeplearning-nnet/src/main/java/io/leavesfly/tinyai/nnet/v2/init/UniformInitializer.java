package io.leavesfly.tinyai.nnet.v2.init;

import io.leavesfly.tinyai.ndarr.NdArray;
import java.util.Random;

/**
 * 均匀分布初始化器
 * <p>
 * 从均匀分布 U(a, b) 中采样初始化张量
 *
 * @author leavesfly
 * @version 2.0
 */
public class UniformInitializer implements Initializer {

    private final float a;
    private final float b;
    private final Random random;

    /**
     * 构造函数
     *
     * @param a 均匀分布下界
     * @param b 均匀分布上界
     */
    public UniformInitializer(float a, float b) {
        this.a = a;
        this.b = b;
        this.random = new Random();
    }

    @Override
    public void initialize(NdArray tensor) {
        float[] data = tensor.getArray();
        float range = b - a;
        for (int i = 0; i < data.length; i++) {
            data[i] = a + random.nextFloat() * range;
        }
    }
}
