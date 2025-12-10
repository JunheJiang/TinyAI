package io.leavesfly.tinyai.nnet.v1.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v1.Layer;
import io.leavesfly.tinyai.nnet.v1.ParameterV1;

import java.util.ArrayList;
import java.util.List;

/**
 * 卷积层实现类
 * <p>
 * 实现了标准的卷积操作，支持步长、填充、偏置等参数。
 * 使用Im2Col技术将卷积操作转换为矩阵乘法，提高计算效率。
 * <p>
 * 卷积公式：output = input * weight + bias
 * 其中 weight形状为 (out_channels, in_channels, kernel_height, kernel_width)
 */
public class ConvLayer extends Layer {

    private ParameterV1 weight;        // 卷积核参数
    private ParameterV1 bias;          // 偏置参数(可选)

    private int inChannels;          // 输入通道数
    private int outChannels;         // 输出通道数
    private int kernelHeight;        // 卷积核高度
    private int kernelWidth;         // 卷积核宽度
    private int stride;              // 步长
    private int padding;             // 填充
    private boolean useBias;        // 是否使用偏置

    /**
     * 前向传播中缓存的中间结果，供反向传播使用
     */
    private NdArray lastInput;       // 输入缓存
    private NdArray lastIm2col;      // im2col 展开结果
    private int lastOutHeight;
    private int lastOutWidth;
    private int lastBatchSize;

    public ConvLayer(String name) {
        super(name);
    }

    /**
     * 构造卷积层
     *
     * @param name        层名称
     * @param inChannels  输入通道数
     * @param outChannels 输出通道数
     * @param kernelSize  卷积核尺寸(正方形)
     * @param stride      步长
     * @param padding     填充
     * @param useBias     是否使用偏置
     */
    public ConvLayer(String name, int inChannels, int outChannels, int kernelSize,
                     int stride, int padding, boolean useBias) {
        this(name, inChannels, outChannels, kernelSize, kernelSize, stride, padding, useBias);
    }

    /**
     * 构造卷积层(非正方形卷积核)
     */
    public ConvLayer(String name, int inChannels, int outChannels, int kernelHeight, int kernelWidth,
                     int stride, int padding, boolean useBias) {
        super(name, null, null);  // 输入输出形状将在运行时确定

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
        this.useBias = useBias;

        init();
    }

    public ConvLayer(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        // 从输入形状推断参数(默认值)
        if (_inputShape != null && _inputShape.size() == 4) {
            this.inChannels = _inputShape.getDimension(1);
            this.outChannels = 32;  // 默认输出通道数
        } else {
            this.inChannels = 1;
            this.outChannels = 32;
        }
        this.kernelHeight = 3;
        this.kernelWidth = 3;
        this.stride = 1;
        this.padding = 1;
        this.useBias = true;

        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化权重参数 (out_channels, in_channels, kernel_height, kernel_width)
            // 使用He初始化
            double fan_in = inChannels * kernelHeight * kernelWidth;
            double std = Math.sqrt(2.0 / fan_in);

            Shape weightShape = Shape.of(outChannels, inChannels, kernelHeight, kernelWidth);
            NdArray weightData = NdArray.likeRandomN(weightShape).mulNum(std);

            weight = new ParameterV1(weightData);
            weight.setName(name + "_weight");
            addParam("weight", weight);

            // 初始化偏置参数(如果使用)
            if (useBias) {
                bias = new ParameterV1(NdArray.zeros(Shape.of(outChannels)));
                bias.setName(name + "_bias");
                addParam("bias", bias);
            }

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        //todo
        return null;
    }

}