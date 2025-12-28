package io.leavesfly.tinyai.omni.encoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.transformer.TransformerEncoderLayer;
import io.leavesfly.tinyai.omni.config.Qwen3OmniConfig;

import java.util.ArrayList;
import java.util.List;

/**
 * 图像编码器 (Vision Transformer)
 * 
 * 架构流程:
 * 图像 [batch, 3, H, W]
 *   ↓ PatchEmbedding
 * Patches [batch, num_patches, image_hidden_size]
 *   ↓ + Position2D
 * Positioned Patches
 *   ↓ Transformer Layers
 * 图像特征 [batch, num_patches, image_hidden_size]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ImageEncoder extends Module {
    
    private final Qwen3OmniConfig config;
    private final PatchEmbedding patchEmbedding;
    private final Position2D position2D;
    private final Dropout embeddingDropout;
    private final List<TransformerEncoderLayer> encoderLayers;
    
    public ImageEncoder(String name, Qwen3OmniConfig config) {
        super(name);
        this.config = config;
        
        // Patch嵌入层
        this.patchEmbedding = new PatchEmbedding(
            name + "_patch_emb",
            config.getImageSize(),
            config.getPatchSize(),
            config.getImageChannels(),
            config.getImageHiddenSize()  // 注意：使用imageHiddenSize
        );
        registerModule("patch_emb", patchEmbedding);
        
        // 2D位置编码
        this.position2D = new Position2D(
            name + "_pos_2d",
            config.getNumImagePatches(),
            config.getImageHiddenSize()
        );
        registerModule("pos_2d", position2D);
        
        // 嵌入Dropout
        this.embeddingDropout = new Dropout(
            name + "_emb_dropout",
            (float) config.getEmbeddingDropout()
        );
        registerModule("emb_dropout", embeddingDropout);
        
        // Transformer编码器层
        this.encoderLayers = new ArrayList<>();
        for (int i = 0; i < config.getImageEncoderLayers(); i++) {
            TransformerEncoderLayer layer = new TransformerEncoderLayer(
                name + "_encoder_" + i,
                config.getImageHiddenSize(),
                config.getNumAttentionHeads(),
                config.getImageHiddenSize() * 4,  // FFN hidden size
                (float) config.getDropoutRate(),
                true  // Pre-LayerNorm
            );
            encoderLayers.add(layer);
            registerModule("encoder_" + i, layer);
        }
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块自行初始化
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为图像像素 [batch, channels, height, width]
     * @return 图像特征 [batch, num_patches, image_hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("ImageEncoder需要输入图像");
        }
        
        Variable imagePixels = inputs[0];
        
        // Patch嵌入
        Variable patches = patchEmbedding.forward(imagePixels);
        
        // 添加位置编码
        Variable posEncodings = position2D.forward(patches);
        Variable x = patches.add(posEncodings);
        
        // 应用Dropout
        x = embeddingDropout.forward(x);
        
        // 通过Transformer层
        for (TransformerEncoderLayer layer : encoderLayers) {
            x = layer.forward(x);
        }
        
        return x;
    }
    
    public Qwen3OmniConfig getConfig() {
        return config;
    }
    
    public int getNumLayers() {
        return encoderLayers.size();
    }
}
