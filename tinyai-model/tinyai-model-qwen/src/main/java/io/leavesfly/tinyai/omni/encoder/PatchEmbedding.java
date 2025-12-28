package io.leavesfly.tinyai.omni.encoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.conv.Conv2d;

/**
 * Patch嵌入层
 * 
 * 将图像分割成patches并嵌入到向量空间,是Vision Transformer的核心组件。
 * 使用Conv2D实现,卷积核和步长都等于patch_size,保证patches不重叠。
 * 
 * 输入: [batch, channels, height, width]
 * 输出: [batch, num_patches, hidden_size]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class PatchEmbedding extends Module {
    
    private final int imageSize;
    private final int patchSize;
    private final int imageChannels;
    private final int hiddenSize;
    private final int numPatches;
    
    private final Conv2d patchConv;
    
    public PatchEmbedding(String name, int imageSize, int patchSize, 
                         int imageChannels, int hiddenSize) {
        super(name);
        
        if (imageSize % patchSize != 0) {
            throw new IllegalArgumentException(
                "imageSize必须能被patchSize整除: " + imageSize + " % " + patchSize + " != 0"
            );
        }
        
        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.imageChannels = imageChannels;
        this.hiddenSize = hiddenSize;
        this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        
        this.patchConv = new Conv2d(
            name + "_patch_conv",
            imageChannels,
            hiddenSize,
            patchSize,
            patchSize,
            0,
            true
        );
        registerModule("patch_conv", patchConv);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // Conv2D自动初始化参数
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("PatchEmbedding需要输入图像");
        }
        
        Variable image = inputs[0];
        Variable patchFeatures = patchConv.forward(image);
        
        // 重塑为序列格式 [B, H, h', w'] -> [B, N, H]
        int batchSize = patchFeatures.size(0);
        int hidden = patchFeatures.size(1);
        int numPatchesH = patchFeatures.size(2);
        int numPatchesW = patchFeatures.size(3);
        int totalPatches = numPatchesH * numPatchesW;
        
        Variable reshaped = patchFeatures.reshape(
            Shape.of(batchSize, hidden, totalPatches)
        );
        
        NdArray data = reshaped.getValue();
        NdArray transposed = data.transpose(0, 2, 1);
        
        return new Variable(transposed);
    }
    
    public int getNumPatches() {
        return numPatches;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
}
