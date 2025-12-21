package io.leavesfly.tinyai.banana.decoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

/**
 * 上采样块 (Upsample Block)
 * 
 * 将低分辨率特征图上采样到高分辨率:
 * 1. 双线性插值上采样 - 2x分辨率提升
 * 2. 卷积投影 - 通道数调整和特征精炼
 * 3. LayerNorm + ReLU - 归一化和激活
 * 
 * 架构流程:
 * 输入 [batch, in_channels, in_h, in_w]
 *   ↓ Bilinear Upsample (2x)
 * [batch, in_channels, in_h*2, in_w*2]
 *   ↓ Spatial Conv (模拟卷积)
 * [batch, out_channels, out_h, out_w]
 *   ↓ LayerNorm + ReLU
 * 输出 [batch, out_channels, out_h, out_w]
 * 
 * 注意: 由于TinyAI V2暂无转置卷积,使用双线性插值+Linear投影模拟
 * 
 * @author leavesfly
 * @version 1.0
 */
public class UpsampleBlock extends Module {
    
    private final int inChannels;
    private final int outChannels;
    private final int inSize;
    private final int outSize;
    
    // 通道投影层
    private final Linear channelProjection;
    
    // 归一化层
    private final LayerNorm layerNorm;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param inChannels 输入通道数
     * @param outChannels 输出通道数
     * @param inSize 输入空间尺寸（假设正方形）
     * @param outSize 输出空间尺寸（假设正方形）
     */
    public UpsampleBlock(String name, int inChannels, int outChannels, 
                         int inSize, int outSize) {
        super(name);
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.inSize = inSize;
        this.outSize = outSize;
        
        // 通道投影：in_channels -> out_channels
        this.channelProjection = new Linear(
            name + "_ch_proj",
            inChannels,
            outChannels,
            true
        );
        registerModule("ch_proj", channelProjection);
        
        // LayerNorm（应用在通道维度）
        this.layerNorm = new LayerNorm(
            name + "_ln",
            outChannels,
            1e-5f
        );
        registerModule("ln", layerNorm);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块自行初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为输入特征图 [batch, in_channels, in_h, in_w]
     * @return 上采样后的特征图 [batch, out_channels, out_h, out_w]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("UpsampleBlock需要输入特征图");
        }
        
        Variable x = inputs[0];
        
        // 1. 双线性插值上采样
        Variable upsampled = bilinearUpsample(x, outSize, outSize);
        
        // 2. 手动重排维度：[batch, in_channels, out_h, out_w] -> [batch, out_h, out_w, in_channels]
        upsampled = permuteNCHWToNHWC(upsampled);
        
        // 3. 通道投影
        // [batch, out_h, out_w, in_channels] -> [batch, out_h, out_w, out_channels]
        Variable projected = channelProjection.forward(upsampled);
        
        // 4. LayerNorm
        projected = layerNorm.forward(projected);
        
        // 5. ReLU激活
        projected = projected.relu();
        
        // 6. 恢复维度顺序: [batch, out_h, out_w, out_channels] -> [batch, out_channels, out_h, out_w]
        Variable output = permuteNHWCToNCHW(projected);
        
        return output;
    }
    
    /**
     * 双线性插值上采样
     * 
     * 简化实现：使用最近邻插值（双线性插值需要复杂的索引计算）
     * 
     * @param x 输入 [batch, channels, in_h, in_w]
     * @param outH 目标高度
     * @param outW 目标宽度
     * @return 上采样结果 [batch, channels, out_h, out_w]
     */
    private Variable bilinearUpsample(Variable x, int outH, int outW) {
        int[] shape = x.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int channels = shape[1];
        int inH = shape[2];
        int inW = shape[3];
        
        // 计算缩放比例
        float scaleH = (float) inH / outH;
        float scaleW = (float) inW / outW;
        
        // 准备输出数组
        float[] outputData = new float[batchSize * channels * outH * outW];
        NdArray inputData = x.getValue();
        
        // 最近邻上采样（简化实现）
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        // 计算源坐标（最近邻）
                        int srcH = Math.min((int) (h * scaleH), inH - 1);
                        int srcW = Math.min((int) (w * scaleW), inW - 1);
                        
                        // 复制像素值
                        float value = inputData.get(b, c, srcH, srcW);
                        int outIdx = ((b * channels + c) * outH + h) * outW + w;
                        outputData[outIdx] = value;
                    }
                }
            }
        }
        
        NdArray outputArray = NdArray.of(outputData, Shape.of(batchSize, channels, outH, outW));
        return new Variable(outputArray);
    }
    
    /**
     * 手动重排维度: NCHW -> NHWC
     * [batch, channels, height, width] -> [batch, height, width, channels]
     */
    private Variable permuteNCHWToNHWC(Variable x) {
        int[] shape = x.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        
        NdArray inputData = x.getValue();
        float[] outputData = new float[batchSize * height * width * channels];
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int c = 0; c < channels; c++) {
                        float value = inputData.get(b, c, h, w);
                        int outIdx = ((b * height + h) * width + w) * channels + c;
                        outputData[outIdx] = value;
                    }
                }
            }
        }
        
        NdArray outputArray = NdArray.of(outputData, Shape.of(batchSize, height, width, channels));
        return new Variable(outputArray);
    }
    
    /**
     * 手动重排维度: NHWC -> NCHW
     * [batch, height, width, channels] -> [batch, channels, height, width]
     */
    private Variable permuteNHWCToNCHW(Variable x) {
        int[] shape = x.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int height = shape[1];
        int width = shape[2];
        int channels = shape[3];
        
        NdArray inputData = x.getValue();
        float[] outputData = new float[batchSize * channels * height * width];
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int c = 0; c < channels; c++) {
                        float value = inputData.get(b, h, w, c);
                        int outIdx = ((b * channels + c) * height + h) * width + w;
                        outputData[outIdx] = value;
                    }
                }
            }
        }
        
        NdArray outputArray = NdArray.of(outputData, Shape.of(batchSize, channels, height, width));
        return new Variable(outputArray);
    }
    
    // ==================== Getter方法 ====================
    
    public int getInChannels() {
        return inChannels;
    }
    
    public int getOutChannels() {
        return outChannels;
    }
    
    public int getInSize() {
        return inSize;
    }
    
    public int getOutSize() {
        return outSize;
    }
    
    @Override
    public String toString() {
        return String.format(
            "UpsampleBlock{inChannels=%d, outChannels=%d, %dx%d->%dx%d}",
            inChannels, outChannels, inSize, inSize, outSize, outSize
        );
    }
}
