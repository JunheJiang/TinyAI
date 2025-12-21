package io.leavesfly.tinyai.banana.decoder;

import io.leavesfly.tinyai.banana.config.BananaConfig;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;
import io.leavesfly.tinyai.nnet.v2.layer.transformer.TransformerDecoderLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 图像解码器 (Image Decoder)
 * 
 * 将融合后的多模态特征解码为图像像素:
 * 1. Transformer解码器层 - 提取高层特征
 * 2. 特征重塑 - 将序列特征重塑为2D特征图
 * 3. 上采样模块 - 逐步恢复图像分辨率
 * 4. 像素投影 - 输出RGB图像
 * 
 * 架构流程:
 * 融合特征 [batch, 256, 512]
 *   ↓ Transformer Decoder Layers
 * 解码特征 [batch, 256, 512]
 *   ↓ Reshape
 * 特征图 [batch, 512, 16, 16]
 *   ↓ Upsample Blocks (16→32→64→128→256)
 * 高分辨率特征 [batch, 64, 256, 256]
 *   ↓ Pixel Projection
 * 图像 [batch, 3, 256, 256]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ImageDecoder extends Module {
    
    private final BananaConfig config;
    
    // Transformer解码器层列表
    private final List<TransformerDecoderLayer> decoderLayers;
    
    // Dropout层
    private final Dropout featureDropout;
    
    // 特征投影层（降维准备上采样）
    private final Linear featureProjection;
    private final LayerNorm featureNorm;
    
    // 上采样模块列表
    private final List<UpsampleBlock> upsampleBlocks;
    
    // 像素投影层（最终输出层）
    private final PixelProjection pixelProjection;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Banana配置对象
     */
    public ImageDecoder(String name, BananaConfig config) {
        super(name);
        this.config = config;
        
        // 1. 初始化Transformer解码器层
        this.decoderLayers = new ArrayList<>();
        int numDecoderLayers = config.getNumEncoderLayers(); // 与编码器层数相同
        for (int i = 0; i < numDecoderLayers; i++) {
            TransformerDecoderLayer layer = new TransformerDecoderLayer(
                name + "_decoder_" + i,
                config.getHiddenSize(),
                config.getNumHeads(),
                config.getFfnHiddenSize(),
                (float) config.getDropoutRate(),
                true  // 使用Pre-LayerNorm
            );
            decoderLayers.add(layer);
            registerModule("decoder_" + i, layer);
        }
        
        // 2. 特征Dropout
        this.featureDropout = new Dropout(
            name + "_feat_dropout",
            (float) config.getDropoutRate()
        );
        registerModule("feat_dropout", featureDropout);
        
        // 3. 特征投影层（降维以便上采样）
        // hidden_size (512) -> upsampling_base_dim (256)
        int upsamplingBaseDim = 256;
        this.featureProjection = new Linear(
            name + "_feat_proj",
            config.getHiddenSize(),
            upsamplingBaseDim,
            true
        );
        registerModule("feat_proj", featureProjection);
        
        this.featureNorm = new LayerNorm(
            name + "_feat_norm",
            upsamplingBaseDim,
            (float) config.getLayerNormEpsilon()
        );
        registerModule("feat_norm", featureNorm);
        
        // 4. 初始化上采样模块
        // 从16x16上采样到256x256：需要4个2x上采样步骤
        // 16 -> 32 -> 64 -> 128 -> 256
        this.upsampleBlocks = new ArrayList<>();
        
        // 计算初始patch网格尺寸
        int patchGridSize = config.getImageSize() / config.getPatchSize(); // 256/16=16
        
        // 第一个上采样块: [256, 16, 16] -> [128, 32, 32]
        UpsampleBlock block1 = new UpsampleBlock(
            name + "_upsample_0",
            upsamplingBaseDim,
            128,
            patchGridSize,
            patchGridSize * 2
        );
        upsampleBlocks.add(block1);
        registerModule("upsample_0", block1);
        
        // 第二个上采样块: [128, 32, 32] -> [64, 64, 64]
        UpsampleBlock block2 = new UpsampleBlock(
            name + "_upsample_1",
            128,
            64,
            patchGridSize * 2,
            patchGridSize * 4
        );
        upsampleBlocks.add(block2);
        registerModule("upsample_1", block2);
        
        // 第三个上采样块: [64, 64, 64] -> [32, 128, 128]
        UpsampleBlock block3 = new UpsampleBlock(
            name + "_upsample_2",
            64,
            32,
            patchGridSize * 4,
            patchGridSize * 8
        );
        upsampleBlocks.add(block3);
        registerModule("upsample_2", block3);
        
        // 第四个上采样块: [32, 128, 128] -> [16, 256, 256]
        UpsampleBlock block4 = new UpsampleBlock(
            name + "_upsample_3",
            32,
            16,
            patchGridSize * 8,
            config.getImageSize()
        );
        upsampleBlocks.add(block4);
        registerModule("upsample_3", block4);
        
        // 5. 像素投影层（最终输出）
        this.pixelProjection = new PixelProjection(
            name + "_pixel_proj",
            16,
            config.getImageChannels() // 3 (RGB)
        );
        registerModule("pixel_proj", pixelProjection);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块的参数由其自己初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为融合特征 [batch, num_patches, hidden_size]
     *               inputs[1]为编码器输出（用于cross-attention）[batch, num_patches, hidden_size]
     * @return 生成的图像 [batch, channels, height, width]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length < 2) {
            throw new IllegalArgumentException("ImageDecoder需要至少2个输入: fusedFeatures和encoderOutput");
        }
        
        Variable fusedFeatures = inputs[0];
        Variable encoderOutput = inputs[1];
        
        validateInput(fusedFeatures);
        
        // 1. 通过Transformer解码器层
        Variable x = fusedFeatures;
        for (TransformerDecoderLayer layer : decoderLayers) {
            x = layer.forward(x, encoderOutput);
        }
        
        // 2. 应用Dropout
        x = featureDropout.forward(x);
        
        // 3. 特征投影和归一化
        // [batch, num_patches, hidden_size] -> [batch, num_patches, upsampling_base_dim]
        x = featureProjection.forward(x);
        x = featureNorm.forward(x);
        
        // 4. 重塑为2D特征图
        // [batch, num_patches, dim] -> [batch, dim, grid_h, grid_w]
        x = reshapeTo2D(x);
        
        // 5. 通过上采样模块
        for (UpsampleBlock block : upsampleBlocks) {
            x = block.forward(x);
        }
        
        // 6. 像素投影，输出RGB图像
        Variable image = pixelProjection.forward(x);
        
        // 7. 应用Tanh激活，将像素值归一化到[-1, 1]
        image = image.tanh();
        
        return image;
    }
    
    /**
     * 将序列特征重塑为2D特征图
     * 
     * @param x 输入特征 [batch, num_patches, dim]
     * @return 2D特征图 [batch, dim, grid_h, grid_w]
     */
    private Variable reshapeTo2D(Variable x) {
        int[] shape = x.getValue().getShape().getShapeDims();
        int batchSize = shape[0];
        int numPatches = shape[1];
        int dim = shape[2];
        
        // 计算网格尺寸
        int gridSize = (int) Math.sqrt(numPatches); // 16 for 256 patches
        
        if (gridSize * gridSize != numPatches) {
            throw new IllegalStateException(
                "numPatches必须是完全平方数: " + numPatches
            );
        }
        
        // 手动重排数据: [batch, num_patches, dim] -> [batch, dim, grid_h, grid_w]
        NdArray inputData = x.getValue();
        float[] outputData = new float[batchSize * dim * gridSize * gridSize];
        
        for (int b = 0; b < batchSize; b++) {
            for (int p = 0; p < numPatches; p++) {
                int h = p / gridSize;
                int w = p % gridSize;
                
                for (int d = 0; d < dim; d++) {
                    float value = inputData.get(b, p, d);
                    int outIdx = ((b * dim + d) * gridSize + h) * gridSize + w;
                    outputData[outIdx] = value;
                }
            }
        }
        
        NdArray outputArray = NdArray.of(outputData, Shape.of(batchSize, dim, gridSize, gridSize));
        return new Variable(outputArray);
    }
    
    /**
     * 验证输入有效性
     */
    private void validateInput(Variable fusedFeatures) {
        if (fusedFeatures == null) {
            throw new IllegalArgumentException("fusedFeatures不能为null");
        }
        
        int[] shape = fusedFeatures.getValue().getShape().getShapeDims();
        if (shape.length != 3) {
            throw new IllegalArgumentException(
                "fusedFeatures必须是3维 [batch, num_patches, hidden_size], 当前shape: " + 
                java.util.Arrays.toString(shape)
            );
        }
        
        int numPatches = shape[1];
        int hiddenSize = shape[2];
        
        if (numPatches != config.getNumPatches()) {
            throw new IllegalArgumentException(
                "patch数量不匹配: 期望" + config.getNumPatches() + ", 实际" + numPatches
            );
        }
        
        if (hiddenSize != config.getHiddenSize()) {
            throw new IllegalArgumentException(
                "隐藏层维度不匹配: 期望" + config.getHiddenSize() + ", 实际" + hiddenSize
            );
        }
    }
    
    // ==================== Getter方法 ====================
    
    public BananaConfig getConfig() {
        return config;
    }
    
    public int getNumLayers() {
        return decoderLayers.size();
    }
    
    public List<TransformerDecoderLayer> getDecoderLayers() {
        return decoderLayers;
    }
    
    public List<UpsampleBlock> getUpsampleBlocks() {
        return upsampleBlocks;
    }
    
    @Override
    public String toString() {
        return String.format(
            "ImageDecoder{numLayers=%d, hiddenSize=%d, outputSize=%dx%d}",
            decoderLayers.size(),
            config.getHiddenSize(),
            config.getImageSize(),
            config.getImageSize()
        );
    }
}
