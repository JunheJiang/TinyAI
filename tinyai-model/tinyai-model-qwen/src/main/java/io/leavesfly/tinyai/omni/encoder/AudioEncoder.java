package io.leavesfly.tinyai.omni.encoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;
import io.leavesfly.tinyai.nnet.v2.layer.transformer.TransformerEncoderLayer;
import io.leavesfly.tinyai.omni.config.Qwen3OmniConfig;

import java.util.ArrayList;
import java.util.List;

/**
 * 音频编码器
 * 
 * 将音频波形转换为特征表示,类似于Vision Transformer处理图像的方式。
 * 
 * 架构流程:
 * 音频波形 [batch, num_samples]
 *   ↓ MelSpectrogram
 * Mel频谱图 [batch, mel_bins, time_frames]
 *   ↓ AudioPatchEmbedding
 * Audio Patches [batch, num_patches, audio_hidden_size]
 *   ↓ + PositionalEncoding
 * Positioned Patches
 *   ↓ Transformer Layers
 * 音频特征 [batch, num_patches, audio_hidden_size]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class AudioEncoder extends Module {
    
    private final Qwen3OmniConfig config;
    private final MelSpectrogram melTransform;
    private final Linear patchEmbedding;
    private final Dropout embeddingDropout;
    private final List<TransformerEncoderLayer> encoderLayers;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Qwen3Omni配置
     */
    public AudioEncoder(String name, Qwen3OmniConfig config) {
        super(name);
        this.config = config;
        
        // Mel频谱转换器
        this.melTransform = new MelSpectrogram(
            config.getAudioSampleRate(),
            config.getMelBins(),
            config.getAudioFrameLengthMs(),
            config.getAudioFrameShiftMs()
        );
        
        // 音频Patch嵌入(简化版:直接线性投影)
        // 输入: [batch, mel_bins * patch_size] 
        // 输出: [batch, audio_hidden_size]
        int patchInputSize = config.getMelBins() * config.getAudioPatchSize();
        this.patchEmbedding = new Linear(
            name + "_patch_emb",
            patchInputSize,
            config.getAudioHiddenSize(),
            true
        );
        registerModule("patch_emb", patchEmbedding);
        
        // 嵌入Dropout
        this.embeddingDropout = new Dropout(
            name + "_emb_dropout",
            (float) config.getEmbeddingDropout()
        );
        registerModule("emb_dropout", embeddingDropout);
        
        // Transformer编码器层
        this.encoderLayers = new ArrayList<>();
        for (int i = 0; i < config.getAudioEncoderLayers(); i++) {
            TransformerEncoderLayer layer = new TransformerEncoderLayer(
                name + "_encoder_" + i,
                config.getAudioHiddenSize(),
                config.getNumAttentionHeads(),
                config.getAudioHiddenSize() * 4,
                (float) config.getDropoutRate(),
                true
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
     * @param inputs inputs[0]为音频波形 [batch, num_samples]
     * @return 音频特征 [batch, num_patches, audio_hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("AudioEncoder需要输入音频波形");
        }
        
        Variable audioWaveform = inputs[0];
        int batchSize = audioWaveform.size(0);
        
        // 1. 转换为Mel频谱图
        List<NdArray> melSpecs = new ArrayList<>();
        for (int b = 0; b < batchSize; b++) {
            NdArray waveform = extractBatch(audioWaveform.getValue(), b);
            NdArray melSpec = melTransform.transform(waveform);
            melSpecs.add(melSpec);
        }
        
        // 2. 分割成patches并嵌入
        Variable patches = createAudioPatches(melSpecs, batchSize);
        
        // 3. 应用Dropout
        Variable x = embeddingDropout.forward(patches);
        
        // 4. 通过Transformer层
        for (TransformerEncoderLayer layer : encoderLayers) {
            x = layer.forward(x);
        }
        
        return x;
    }
    
    /**
     * 提取batch中的单个样本
     */
    private NdArray extractBatch(NdArray batchData, int batchIdx) {
        int[] shape = batchData.getShape().getShapeDims();
        int numSamples = shape[1];
        
        float[] sampleData = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            sampleData[i] = batchData.get(batchIdx, i);
        }
        
        return NdArray.of(sampleData, Shape.of(numSamples));
    }
    
    /**
     * 创建音频patches并嵌入
     * 
     * 将Mel频谱图分割成patches,然后通过线性层嵌入
     */
    private Variable createAudioPatches(List<NdArray> melSpecs, int batchSize) {
        int patchSize = config.getAudioPatchSize();
        int melBins = config.getMelBins();
        
        // 计算patch数量
        int timeFrames = melSpecs.get(0).getShape().getShapeDims()[1];
        int numPatches = timeFrames / patchSize;
        
        // 创建patches矩阵 [batch, num_patches, mel_bins * patch_size]
        int patchInputSize = melBins * patchSize;
        float[][][] patchesData = new float[batchSize][numPatches][patchInputSize];
        
        for (int b = 0; b < batchSize; b++) {
            NdArray melSpec = melSpecs.get(b);
            
            for (int p = 0; p < numPatches; p++) {
                int timeStart = p * patchSize;
                
                // 提取patch
                for (int m = 0; m < melBins; m++) {
                    for (int t = 0; t < patchSize && (timeStart + t) < timeFrames; t++) {
                        patchesData[b][p][m * patchSize + t] = melSpec.get(m, timeStart + t);
                    }
                }
            }
        }
        
        // 转换为NdArray
        int totalSize = batchSize * numPatches * patchInputSize;
        float[] flatData = new float[totalSize];
        int idx = 0;
        for (int b = 0; b < batchSize; b++) {
            for (int p = 0; p < numPatches; p++) {
                for (int i = 0; i < patchInputSize; i++) {
                    flatData[idx++] = patchesData[b][p][i];
                }
            }
        }
        
        NdArray patchesArray = NdArray.of(flatData, Shape.of(batchSize, numPatches, patchInputSize));
        Variable patches = new Variable(patchesArray);
        
        // 通过线性层嵌入
        // 需要对每个patch单独应用线性层
        // [batch, num_patches, patch_input_size] -> [batch, num_patches, audio_hidden_size]
        return applyPatchEmbedding(patches, batchSize, numPatches, patchInputSize);
    }
    
    /**
     * 对patches应用嵌入
     */
    private Variable applyPatchEmbedding(Variable patches, int batchSize, int numPatches, int patchInputSize) {
        int audioHiddenSize = config.getAudioHiddenSize();
        float[] outputData = new float[batchSize * numPatches * audioHiddenSize];
        
        NdArray patchesData = patches.getValue();
        
        // 对每个patch应用线性变换
        for (int b = 0; b < batchSize; b++) {
            for (int p = 0; p < numPatches; p++) {
                // 提取单个patch
                float[] patchInput = new float[patchInputSize];
                for (int i = 0; i < patchInputSize; i++) {
                    patchInput[i] = patchesData.get(b, p, i);
                }
                
                // 应用线性层
                NdArray patchInputArray = NdArray.of(patchInput, Shape.of(1, patchInputSize));
                Variable patchVar = new Variable(patchInputArray);
                Variable embedded = patchEmbedding.forward(patchVar);
                
                // 存储结果
                NdArray embeddedData = embedded.getValue();
                for (int i = 0; i < audioHiddenSize; i++) {
                    outputData[(b * numPatches + p) * audioHiddenSize + i] = embeddedData.get(0, i);
                }
            }
        }
        
        return new Variable(NdArray.of(outputData, Shape.of(batchSize, numPatches, audioHiddenSize)));
    }
    
    public Qwen3OmniConfig getConfig() {
        return config;
    }
    
    public int getNumLayers() {
        return encoderLayers.size();
    }
}
