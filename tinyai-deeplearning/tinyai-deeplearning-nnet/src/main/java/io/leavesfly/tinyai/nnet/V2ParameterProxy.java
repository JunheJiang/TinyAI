package io.leavesfly.tinyai.nnet;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v1.ParameterV1;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

/**
 * V2 Parameter 的代理类
 * <p>
 * 用于将 V2 Parameter 包装为 V1 ParameterV1 的代理，
 * 当调用 setValue 时会同步更新原始的 V2 Parameter。
 * 这解决了 SGD 等优化器通过 V1 接口更新 V2 模型参数的问题。
 *
 * @author TinyAI
 * @version 2.0
 */
public class V2ParameterProxy extends ParameterV1 {

    /**
     * 被代理的原始 V2 Parameter
     */
    private final Parameter v2Parameter;

    /**
     * 构造函数
     *
     * @param v2Parameter 被代理的 V2 Parameter
     */
    public V2ParameterProxy(Parameter v2Parameter) {
        super(v2Parameter.data());
        this.v2Parameter = v2Parameter;
        // 同步 requireGrad 设置
        this.setRequireGrad(v2Parameter.requiresGrad());
    }

    /**
     * 设置参数值，同时同步更新原始的 V2 Parameter
     *
     * @param value 新的参数值
     */
    @Override
    public void setValue(NdArray value) {
        super.setValue(value);
        // 同步更新 V2 Parameter
        if (v2Parameter != null) {
            v2Parameter.setData(value);
        }
    }

    /**
     * 获取梯度，从原始 V2 Parameter 获取最新的梯度
     *
     * @return 参数梯度
     */
    @Override
    public NdArray getGrad() {
        // 优先从 V2 Parameter 获取梯度，确保获取最新的梯度值
        if (v2Parameter != null && v2Parameter.grad() != null) {
            return v2Parameter.grad();
        }
        return super.getGrad();
    }

    /**
     * 设置梯度，同时同步更新原始的 V2 Parameter
     *
     * @param grad 新的梯度值
     */
    @Override
    public void setGrad(NdArray grad) {
        super.setGrad(grad);
        // 同步更新 V2 Parameter
        if (v2Parameter != null) {
            v2Parameter.setGrad(grad);
        }
    }

    /**
     * 获取被代理的 V2 Parameter
     *
     * @return 原始的 V2 Parameter
     */
    public Parameter getV2Parameter() {
        return v2Parameter;
    }
}

