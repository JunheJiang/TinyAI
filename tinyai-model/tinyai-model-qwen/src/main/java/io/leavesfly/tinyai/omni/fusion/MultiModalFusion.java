package io.leavesfly.tinyai.omni.fusion;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;
import io.leavesfly.tinyai.omni.config.Qwen3OmniConfig;

/**
 * 多模态融合模块
 * 
 * 整合文本、图像、音频特征,实现三模态交互和信息融合。
 * 
 * 融合策略:
 * 1. Text → Image 跨模态注意力
 * 2. Text → Audio 跨模态注意力
 * 3. Image → Audio 跨模态注意力(可选)
 * 4. 残差连接和LayerNorm
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MultiModalFusion extends Module {
    
    private final Qwen3OmniConfig config;
    
    // Text → Image 跨模态注意力
    private final CrossModalAttention text2ImageAttn;
    private final LayerNorm text2ImageNorm;
    private final Dropout text2ImageDropout;
    
    // Text → Audio 跨模态注意力
    private final CrossModalAttention text2AudioAttn;
    private final LayerNorm text2AudioNorm;
    private final Dropout text2AudioDropout;
    
    // Image → Audio 跨模态注意力
    private final CrossModalAttention image2AudioAttn;
    private final LayerNorm image2AudioNorm;
    private final Dropout image2AudioDropout;
    
    public MultiModalFusion(String name, Qwen3OmniConfig config) {
        super(name);
        this.config = config;
        
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getCrossModalHeads();
        float dropout = (float) config.getDropoutRate();
        
        // Text → Image
        this.text2ImageAttn = new CrossModalAttention(
            name + "_text2image", hiddenSize, numHeads, dropout
        );
        this.text2ImageNorm = new LayerNorm(
            name + "_text2image_norm", hiddenSize, (float) config.getRmsNormEps()
        );
        this.text2ImageDropout = new Dropout(
            name + "_text2image_dropout", dropout
        );
        registerModule("text2image_attn", text2ImageAttn);
        registerModule("text2image_norm", text2ImageNorm);
        registerModule("text2image_dropout", text2ImageDropout);
        
        // Text → Audio
        this.text2AudioAttn = new CrossModalAttention(
            name + "_text2audio", hiddenSize, numHeads, dropout
        );
        this.text2AudioNorm = new LayerNorm(
            name + "_text2audio_norm", hiddenSize, (float) config.getRmsNormEps()
        );
        this.text2AudioDropout = new Dropout(
            name + "_text2audio_dropout", dropout
        );
        registerModule("text2audio_attn", text2AudioAttn);
        registerModule("text2audio_norm", text2AudioNorm);
        registerModule("text2audio_dropout", text2AudioDropout);
        
        // Image → Audio
        this.image2AudioAttn = new CrossModalAttention(
            name + "_image2audio", hiddenSize, numHeads, dropout
        );
        this.image2AudioNorm = new LayerNorm(
            name + "_image2audio_norm", hiddenSize, (float) config.getRmsNormEps()
        );
        this.image2AudioDropout = new Dropout(
            name + "_image2audio_dropout", dropout
        );
        registerModule("image2audio_attn", image2AudioAttn);
        registerModule("image2audio_norm", image2AudioNorm);
        registerModule("image2audio_dropout", image2AudioDropout);
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 子模块自行初始化
    }
    
    /**
     * 三模态融合
     * 
     * @param inputs inputs[0]: textFeatures [batch, text_len, hidden_size]
     *               inputs[1]: imageFeatures [batch, num_image_patches, hidden_size]
     *               inputs[2]: audioFeatures [batch, num_audio_patches, hidden_size]
     * @return 融合后的特征数组 [fusedText, fusedImage, fusedAudio]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length < 3) {
            throw new IllegalArgumentException("MultiModalFusion需要3个输入: text, image, audio");
        }
        
        Variable textFeatures = inputs[0];
        Variable imageFeatures = inputs[1];
        Variable audioFeatures = inputs[2];
        
        // Text融合Image和Audio信息
        Variable textWithImage = fuseModality(
            textFeatures, imageFeatures, 
            text2ImageNorm, text2ImageAttn, text2ImageDropout
        );
        Variable textWithAudio = fuseModality(
            textFeatures, audioFeatures,
            text2AudioNorm, text2AudioAttn, text2AudioDropout
        );
        Variable fusedText = textFeatures.add(textWithImage).add(textWithAudio);
        
        // Image融合Audio信息
        Variable imageWithAudio = fuseModality(
            imageFeatures, audioFeatures,
            image2AudioNorm, image2AudioAttn, image2AudioDropout
        );
        Variable fusedImage = imageFeatures.add(imageWithAudio);
        
        // Audio保持原样或可选融合
        Variable fusedAudio = audioFeatures;
        
        // 返回融合后的三个模态特征
        return new Variable[]{fusedText, fusedImage, fusedAudio}[0];  // 简化返回
    }
    
    /**
     * 单个模态融合辅助方法
     */
    private Variable fuseModality(
        Variable queryFeatures,
        Variable kvFeatures,
        LayerNorm norm,
        CrossModalAttention attn,
        Dropout dropout
    ) {
        // Pre-LayerNorm
        Variable normedQuery = norm.forward(queryFeatures);
        
        // 跨模态注意力
        Variable attnOutput = attn.forward(normedQuery, kvFeatures);
        
        // Dropout
        Variable dropped = dropout.forward(attnOutput);
        
        return dropped;
    }
    
    /**
     * 双模态融合(向后兼容)
     */
    public Variable[] forwardBoth(Variable textFeatures, Variable imageFeatures) {
        Variable textWithImage = fuseModality(
            textFeatures, imageFeatures,
            text2ImageNorm, text2ImageAttn, text2ImageDropout
        );
        Variable fusedText = textFeatures.add(textWithImage);
        Variable fusedImage = imageFeatures;
        
        return new Variable[]{fusedText, fusedImage};
    }
}
