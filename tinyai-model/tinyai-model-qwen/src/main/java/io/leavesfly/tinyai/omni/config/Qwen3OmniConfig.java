package io.leavesfly.tinyai.omni.config;

/**
 * Qwen3-Omni 全模态模型配置类
 * 
 * Qwen3-Omni是基于Qwen3架构扩展的全模态大模型，支持文本、图像、音频的统一编码和生成。
 * 
 * 核心特性：
 * 1. 统一隐藏空间 - 所有模态对齐到相同维度
 * 2. 混合融合策略 - 早期+后期融合
 * 3. Qwen3主干 - RMSNorm + RoPE + GQA + SwiGLU
 * 4. 多模态生成 - 支持文本/图像/音频生成
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3OmniConfig {
    
    // ==================== 基础模型配置 (继承Qwen3) ====================
    
    /** 词汇表大小，默认32000 */
    private int vocabSize = 32000;
    
    /** 统一隐藏层维度（所有模态对齐到此维度），默认768 */
    private int hiddenSize = 768;
    
    /** 中间层维度（FFN），默认为隐藏维度的2.75倍 */
    private int intermediateSize = 2112;
    
    /** Transformer解码器层数，默认12层 */
    private int numHiddenLayers = 12;
    
    /** 注意力头数，默认12 */
    private int numAttentionHeads = 12;
    
    /** 键值头数（GQA），默认12 */
    private int numKeyValueHeads = 12;
    
    /** 最大位置编码数（序列长度），默认2048 */
    private int maxPositionEmbeddings = 2048;
    
    // ==================== 图像编码器配置 (参考Banana) ====================
    
    /** 图像尺寸（高度和宽度），默认224x224 */
    private int imageSize = 224;
    
    /** Patch尺寸（每个patch的高度和宽度），默认16x16 */
    private int patchSize = 16;
    
    /** 图像通道数，默认3（RGB） */
    private int imageChannels = 3;
    
    /** 图像Patch数量（自动计算） = (imageSize / patchSize)^2 */
    private int numImagePatches;
    
    /** 图像编码器层数，默认6层 */
    private int imageEncoderLayers = 6;
    
    /** 图像编码器隐藏维度，默认512 */
    private int imageHiddenSize = 512;
    
    // ==================== 音频编码器配置 (新增) ====================
    
    /** 音频采样率，默认16000 Hz */
    private int audioSampleRate = 16000;
    
    /** Mel频谱bin数量，默认80 */
    private int melBins = 80;
    
    /** 音频帧长度（毫秒），默认25ms */
    private int audioFrameLengthMs = 25;
    
    /** 音频帧移位（毫秒），默认10ms */
    private int audioFrameShiftMs = 10;
    
    /** 音频Patch大小（时间维度），默认4帧 */
    private int audioPatchSize = 4;
    
    /** 音频编码器层数，默认6层 */
    private int audioEncoderLayers = 6;
    
    /** 音频编码器隐藏维度，默认512 */
    private int audioHiddenSize = 512;
    
    /** 最大音频长度（秒），默认30秒 */
    private int maxAudioLengthSeconds = 30;
    
    // ==================== 多模态融合配置 ====================
    
    /** 融合策略，默认混合融合 */
    private FusionStrategy fusionStrategy = FusionStrategy.HYBRID_FUSION;
    
    /** 是否启用跨模态注意力，默认启用 */
    private boolean enableCrossModalAttention = true;
    
    /** 跨模态注意力头数，默认12头 */
    private int crossModalHeads = 12;
    
    /** 模态dropout概率，默认0.1 */
    private double modalityDropout = 0.1;
    
    // ==================== MoE配置(可选) ====================
    
    /** 是否启用MoE架构，默认false */
    private boolean enableMoE = false;
    
    /** 专家数量，默认8个专家 */
    private int numExperts = 8;
    
    /** Top-K选择数量，默认选择2个专家 */
    private int expertTopK = 2;
    
    /** 路由噪声因子，默认0.1 */
    private float expertNoiseFactor = 0.1f;
    
    /** 专家隐藏层倍数（相对于hiddenSize），默认4倍 */
    private int expertHiddenMultiplier = 4;
    
    /** 是否启用负载均衡，默认true */
    private boolean expertLoadBalance = true;
    
    /** 重要性损失系数，默认0.01 */
    private float expertImportanceCoef = 0.01f;
    
    /** 负载损失系数，默认0.01 */
    private float expertLoadCoef = 0.01f;
    
    // ==================== Dropout配置 ====================
    
    /** 残差dropout概率，默认0.1 */
    private double dropoutRate = 0.1;
    
    /** 注意力dropout概率，默认0.1 */
    private double attentionDropout = 0.1;
    
    /** 嵌入dropout概率，默认0.1 */
    private double embeddingDropout = 0.1;
    
    // ==================== 归一化配置 ====================
    
    /** RMSNorm epsilon，默认1e-6 */
    private double rmsNormEps = 1e-6;
    
    // ==================== RoPE配置 ====================
    
    /** RoPE theta值，默认10000.0 */
    private double ropeTheta = 10000.0;
    
    // ==================== 特殊标记配置 ====================
    
    /** Padding token ID */
    private int padTokenId = 0;
    
    /** 开始序列token ID */
    private int bosTokenId = 1;
    
    /** 结束序列token ID */
    private int eosTokenId = 2;
    
    /** 图像开始token ID */
    private int imageStartTokenId = 3;
    
    /** 图像结束token ID */
    private int imageEndTokenId = 4;
    
    /** 音频开始token ID */
    private int audioStartTokenId = 5;
    
    /** 音频结束token ID */
    private int audioEndTokenId = 6;
    
    /**
     * 默认构造函数
     */
    public Qwen3OmniConfig() {
        updateDerivedParams();
    }
    
    // ==================== 预设配置工厂方法 ====================
    
    /**
     * 创建Tiny配置（教学用，最小规模）
     * 配置：512维, 6层, 8头, 224x224图像
     */
    public static Qwen3OmniConfig createTinyConfig() {
        Qwen3OmniConfig config = new Qwen3OmniConfig();
        config.setHiddenSize(512);
        config.setIntermediateSize(1408);
        config.setNumHiddenLayers(6);
        config.setNumAttentionHeads(8);
        config.setNumKeyValueHeads(8);
        config.setImageSize(224);
        config.setImageEncoderLayers(4);
        config.setImageHiddenSize(384);
        config.setAudioEncoderLayers(4);
        config.setAudioHiddenSize(384);
        config.updateDerivedParams();
        return config;
    }
    
    /**
     * 创建Small配置（实验用）
     * 配置：768维, 12层, 12头, 384x384图像
     */
    public static Qwen3OmniConfig createSmallConfig() {
        Qwen3OmniConfig config = new Qwen3OmniConfig();
        config.setHiddenSize(768);
        config.setIntermediateSize(2112);
        config.setNumHiddenLayers(12);
        config.setNumAttentionHeads(12);
        config.setNumKeyValueHeads(12);
        config.setImageSize(384);
        config.setImageEncoderLayers(6);
        config.setImageHiddenSize(512);
        config.setAudioEncoderLayers(6);
        config.setAudioHiddenSize(512);
        config.updateDerivedParams();
        return config;
    }
    
    /**
     * 创建Base配置（标准规模）
     * 配置：1024维, 16层, 16头, 512x512图像
     */
    public static Qwen3OmniConfig createBaseConfig() {
        Qwen3OmniConfig config = new Qwen3OmniConfig();
        config.setHiddenSize(1024);
        config.setIntermediateSize(2816);
        config.setNumHiddenLayers(16);
        config.setNumAttentionHeads(16);
        config.setNumKeyValueHeads(16);
        config.setImageSize(512);
        config.setImageEncoderLayers(8);
        config.setImageHiddenSize(768);
        config.setAudioEncoderLayers(8);
        config.setAudioHiddenSize(768);
        config.updateDerivedParams();
        return config;
    }
    
    // ==================== 辅助方法 ====================
    
    /**
     * 更新派生参数
     */
    public void updateDerivedParams() {
        // 计算图像patch数量
        this.numImagePatches = (imageSize / patchSize) * (imageSize / patchSize);
    }
    
    /**
     * 获取每个注意力头的维度
     */
    public int getHeadDim() {
        return hiddenSize / numAttentionHeads;
    }
    
    /**
     * 获取键值组数
     */
    public int getNumKeyValueGroups() {
        return numAttentionHeads / numKeyValueHeads;
    }
    
    /**
     * 估算模型参数量
     */
    public long estimateParameterCount() {
        // 文本嵌入: vocabSize * hiddenSize
        long textEmbedding = (long) vocabSize * hiddenSize;
        
        // 图像编码器参数
        long imagePatchEmbedding = (long) (patchSize * patchSize * imageChannels) * imageHiddenSize;
        long imageEncoder = (long) imageEncoderLayers * 12L * imageHiddenSize * imageHiddenSize;
        long imageProjection = (long) imageHiddenSize * hiddenSize;
        
        // 音频编码器参数
        long audioEncoder = (long) audioEncoderLayers * 12L * audioHiddenSize * audioHiddenSize;
        long audioProjection = (long) audioHiddenSize * hiddenSize;
        
        // Qwen3主干参数（每层约12 * hiddenSize^2）
        long backboneParams = (long) numHiddenLayers * 12L * hiddenSize * hiddenSize;
        
        // 跨模态注意力参数
        long crossModalParams = (long) numHiddenLayers * 4L * hiddenSize * hiddenSize;
        
        // 输出头参数
        long outputHeads = (long) hiddenSize * vocabSize;
        
        return textEmbedding + imagePatchEmbedding + imageEncoder + imageProjection
             + audioEncoder + audioProjection + backboneParams + crossModalParams + outputHeads;
    }
    
    /**
     * 格式化参数量显示
     */
    public String formatParameters() {
        long params = estimateParameterCount();
        if (params >= 1_000_000_000) {
            return String.format("%.2fB", params / 1_000_000_000.0);
        } else if (params >= 1_000_000) {
            return String.format("%.2fM", params / 1_000_000.0);
        } else {
            return String.format("%,d", params);
        }
    }
    
    /**
     * 验证配置有效性
     */
    public void validate() {
        if (hiddenSize % numAttentionHeads != 0) {
            throw new IllegalArgumentException(
                "hiddenSize必须能被numAttentionHeads整除: " + hiddenSize + " % " + numAttentionHeads + " != 0"
            );
        }
        
        if (numAttentionHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException(
                "numAttentionHeads必须能被numKeyValueHeads整除: " + numAttentionHeads + " % " + numKeyValueHeads + " != 0"
            );
        }
        
        if (imageSize % patchSize != 0) {
            throw new IllegalArgumentException(
                "imageSize必须能被patchSize整除: " + imageSize + " % " + patchSize + " != 0"
            );
        }
        
        if (numImagePatches != (imageSize / patchSize) * (imageSize / patchSize)) {
            throw new IllegalArgumentException(
                "numImagePatches计算错误，请调用updateDerivedParams()"
            );
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "Qwen3OmniConfig{\n" +
            "  基础配置: hiddenSize=%d, numLayers=%d, numHeads=%d, intermediateSize=%d\n" +
            "  文本配置: vocabSize=%d, maxSeqLen=%d\n" +
            "  图像配置: imageSize=%dx%d, patchSize=%dx%d, numPatches=%d, encoderLayers=%d\n" +
            "  音频配置: sampleRate=%d, melBins=%d, encoderLayers=%d, maxLength=%ds\n" +
            "  融合策略: %s, crossModalAttn=%b\n" +
            "  参数量: %s\n" +
            "}",
            hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize,
            vocabSize, maxPositionEmbeddings,
            imageSize, imageSize, patchSize, patchSize, numImagePatches, imageEncoderLayers,
            audioSampleRate, melBins, audioEncoderLayers, maxAudioLengthSeconds,
            fusionStrategy, enableCrossModalAttention,
            formatParameters()
        );
    }
    
    // ==================== Getter和Setter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }
    
    public int getIntermediateSize() {
        return intermediateSize;
    }
    
    public void setIntermediateSize(int intermediateSize) {
        this.intermediateSize = intermediateSize;
    }
    
    public int getNumHiddenLayers() {
        return numHiddenLayers;
    }
    
    public void setNumHiddenLayers(int numHiddenLayers) {
        this.numHiddenLayers = numHiddenLayers;
    }
    
    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }
    
    public void setNumAttentionHeads(int numAttentionHeads) {
        this.numAttentionHeads = numAttentionHeads;
    }
    
    public int getNumKeyValueHeads() {
        return numKeyValueHeads;
    }
    
    public void setNumKeyValueHeads(int numKeyValueHeads) {
        this.numKeyValueHeads = numKeyValueHeads;
    }
    
    public int getMaxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }
    
    public void setMaxPositionEmbeddings(int maxPositionEmbeddings) {
        this.maxPositionEmbeddings = maxPositionEmbeddings;
    }
    
    public int getImageSize() {
        return imageSize;
    }
    
    public void setImageSize(int imageSize) {
        this.imageSize = imageSize;
    }
    
    public int getPatchSize() {
        return patchSize;
    }
    
    public void setPatchSize(int patchSize) {
        this.patchSize = patchSize;
    }
    
    public int getImageChannels() {
        return imageChannels;
    }
    
    public void setImageChannels(int imageChannels) {
        this.imageChannels = imageChannels;
    }
    
    public int getNumImagePatches() {
        return numImagePatches;
    }
    
    public int getImageEncoderLayers() {
        return imageEncoderLayers;
    }
    
    public void setImageEncoderLayers(int imageEncoderLayers) {
        this.imageEncoderLayers = imageEncoderLayers;
    }
    
    public int getImageHiddenSize() {
        return imageHiddenSize;
    }
    
    public void setImageHiddenSize(int imageHiddenSize) {
        this.imageHiddenSize = imageHiddenSize;
    }
    
    public int getAudioSampleRate() {
        return audioSampleRate;
    }
    
    public void setAudioSampleRate(int audioSampleRate) {
        this.audioSampleRate = audioSampleRate;
    }
    
    public int getMelBins() {
        return melBins;
    }
    
    public void setMelBins(int melBins) {
        this.melBins = melBins;
    }
    
    public int getAudioFrameLengthMs() {
        return audioFrameLengthMs;
    }
    
    public void setAudioFrameLengthMs(int audioFrameLengthMs) {
        this.audioFrameLengthMs = audioFrameLengthMs;
    }
    
    public int getAudioFrameShiftMs() {
        return audioFrameShiftMs;
    }
    
    public void setAudioFrameShiftMs(int audioFrameShiftMs) {
        this.audioFrameShiftMs = audioFrameShiftMs;
    }
    
    public int getAudioPatchSize() {
        return audioPatchSize;
    }
    
    public void setAudioPatchSize(int audioPatchSize) {
        this.audioPatchSize = audioPatchSize;
    }
    
    public int getAudioEncoderLayers() {
        return audioEncoderLayers;
    }
    
    public void setAudioEncoderLayers(int audioEncoderLayers) {
        this.audioEncoderLayers = audioEncoderLayers;
    }
    
    public int getAudioHiddenSize() {
        return audioHiddenSize;
    }
    
    public void setAudioHiddenSize(int audioHiddenSize) {
        this.audioHiddenSize = audioHiddenSize;
    }
    
    public int getMaxAudioLengthSeconds() {
        return maxAudioLengthSeconds;
    }
    
    public void setMaxAudioLengthSeconds(int maxAudioLengthSeconds) {
        this.maxAudioLengthSeconds = maxAudioLengthSeconds;
    }
    
    public FusionStrategy getFusionStrategy() {
        return fusionStrategy;
    }
    
    public void setFusionStrategy(FusionStrategy fusionStrategy) {
        this.fusionStrategy = fusionStrategy;
    }
    
    public boolean isEnableCrossModalAttention() {
        return enableCrossModalAttention;
    }
    
    public void setEnableCrossModalAttention(boolean enableCrossModalAttention) {
        this.enableCrossModalAttention = enableCrossModalAttention;
    }
    
    public int getCrossModalHeads() {
        return crossModalHeads;
    }
    
    public void setCrossModalHeads(int crossModalHeads) {
        this.crossModalHeads = crossModalHeads;
    }
    
    public double getModalityDropout() {
        return modalityDropout;
    }
    
    public void setModalityDropout(double modalityDropout) {
        this.modalityDropout = modalityDropout;
    }
    
    public double getDropoutRate() {
        return dropoutRate;
    }
    
    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
    }
    
    public double getAttentionDropout() {
        return attentionDropout;
    }
    
    public void setAttentionDropout(double attentionDropout) {
        this.attentionDropout = attentionDropout;
    }
    
    public double getEmbeddingDropout() {
        return embeddingDropout;
    }
    
    public void setEmbeddingDropout(double embeddingDropout) {
        this.embeddingDropout = embeddingDropout;
    }
    
    public double getRmsNormEps() {
        return rmsNormEps;
    }
    
    public void setRmsNormEps(double rmsNormEps) {
        this.rmsNormEps = rmsNormEps;
    }
    
    public double getRopeTheta() {
        return ropeTheta;
    }
    
    public void setRopeTheta(double ropeTheta) {
        this.ropeTheta = ropeTheta;
    }
    
    public int getPadTokenId() {
        return padTokenId;
    }
    
    public void setPadTokenId(int padTokenId) {
        this.padTokenId = padTokenId;
    }
    
    public int getBosTokenId() {
        return bosTokenId;
    }
    
    public void setBosTokenId(int bosTokenId) {
        this.bosTokenId = bosTokenId;
    }
    
    public int getEosTokenId() {
        return eosTokenId;
    }
    
    public void setEosTokenId(int eosTokenId) {
        this.eosTokenId = eosTokenId;
    }
    
    public int getImageStartTokenId() {
        return imageStartTokenId;
    }
    
    public void setImageStartTokenId(int imageStartTokenId) {
        this.imageStartTokenId = imageStartTokenId;
    }
    
    public int getImageEndTokenId() {
        return imageEndTokenId;
    }
    
    public void setImageEndTokenId(int imageEndTokenId) {
        this.imageEndTokenId = imageEndTokenId;
    }
    
    public int getAudioStartTokenId() {
        return audioStartTokenId;
    }
    
    public void setAudioStartTokenId(int audioStartTokenId) {
        this.audioStartTokenId = audioStartTokenId;
    }
    
    public int getAudioEndTokenId() {
        return audioEndTokenId;
    }
    
    public void setAudioEndTokenId(int audioEndTokenId) {
        this.audioEndTokenId = audioEndTokenId;
    }
    
    // ========== MoE Getters/Setters ==========
    
    public boolean isEnableMoE() {
        return enableMoE;
    }
    
    public void setEnableMoE(boolean enableMoE) {
        this.enableMoE = enableMoE;
    }
    
    public int getNumExperts() {
        return numExperts;
    }
    
    public void setNumExperts(int numExperts) {
        this.numExperts = numExperts;
    }
    
    public int getExpertTopK() {
        return expertTopK;
    }
    
    public void setExpertTopK(int expertTopK) {
        this.expertTopK = expertTopK;
    }
    
    public float getExpertNoiseFactor() {
        return expertNoiseFactor;
    }
    
    public void setExpertNoiseFactor(float expertNoiseFactor) {
        this.expertNoiseFactor = expertNoiseFactor;
    }
    
    public int getExpertHiddenMultiplier() {
        return expertHiddenMultiplier;
    }
    
    public void setExpertHiddenMultiplier(int expertHiddenMultiplier) {
        this.expertHiddenMultiplier = expertHiddenMultiplier;
    }
    
    public int getExpertHiddenSize() {
        return hiddenSize * expertHiddenMultiplier;
    }
    
    public boolean isExpertLoadBalance() {
        return expertLoadBalance;
    }
    
    public void setExpertLoadBalance(boolean expertLoadBalance) {
        this.expertLoadBalance = expertLoadBalance;
    }
    
    public float getExpertImportanceCoef() {
        return expertImportanceCoef;
    }
    
    public void setExpertImportanceCoef(float expertImportanceCoef) {
        this.expertImportanceCoef = expertImportanceCoef;
    }
    
    public float getExpertLoadCoef() {
        return expertLoadCoef;
    }
    
    public void setExpertLoadCoef(float expertLoadCoef) {
        this.expertLoadCoef = expertLoadCoef;
    }
}
