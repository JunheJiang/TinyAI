package io.leavesfly.tinyai.nnet.v2.layer.dnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;

import java.util.Random;

/**
 * V2版本的Dropout层
 * <p>
 * Dropout是一种正则化技术，在训练时随机将部分神经元输出置为0
 * <p>
 * 特性：
 * - 训练模式：应用dropout
 * - 推理模式：直接返回输入（不应用dropout）
 * - 使用inverted dropout：训练时缩放以保持期望值不变
 *
 * @author leavesfly
 * @version 2.0
 */
public class Dropout extends Module {

    private final float p;
    private final Random random;

    /**
     * 构造函数
     *
     * @param name 层名称
     * @param p    dropout概率（0到1之间）
     */
    public Dropout(String name, float p) {
        super(name);
        if (p < 0 || p >= 1) {
            throw new IllegalArgumentException("Dropout probability must be in [0, 1), got: " + p);
        }
        this.p = p;
        this.random = new Random();
    }

    /**
     * 默认构造函数（p=0.5）
     *
     * @param name 层名称
     */
    public Dropout(String name) {
        this(name, 0.5f);
    }

    /**
     * 默认构造函数
     */
    public Dropout() {
        this("dropout", 0.5f);
    }

    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];

        // 推理模式：直接返回输入
        if (!_training) {
            return x;
        }

        // 训练模式：应用dropout
        if (p == 0) {
            return x;
        }

        // 生成dropout mask
        NdArray mask = generateMask(x.getValue());

        // 应用mask并缩放（inverted dropout）
        Variable masked = x.mul(new Variable(mask));
        return masked.mul(new Variable(NdArray.of(1.0f / (1 - p))));
    }

    /**
     * 生成dropout mask
     *
     * @param input 输入张量
     * @return mask张量（0或1）
     */
    private NdArray generateMask(NdArray input) {
        float[] maskData = new float[input.getShape().size()];
        for (int i = 0; i < maskData.length; i++) {
            maskData[i] = random.nextFloat() > p ? 1.0f : 0.0f;
        }
        return NdArray.of(maskData, input.getShape());
    }

    /**
     * 获取dropout概率
     *
     * @return dropout概率
     */
    public float getP() {
        return p;
    }

    @Override
    public String toString() {
        return "Dropout{name='" + name + "', p=" + p + ", training=" + _training + '}';
    }
}
