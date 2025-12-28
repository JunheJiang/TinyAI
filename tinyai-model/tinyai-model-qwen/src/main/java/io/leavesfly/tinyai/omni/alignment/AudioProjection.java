package io.leavesfly.tinyai.omni.alignment;

/**
 * 音频特征投影
 * 
 * 将音频编码器的输出投影到统一的隐藏空间。
 * 
 * 输入: [batch, num_audio_patches, audio_hidden_size]
 * 输出: [batch, num_audio_patches, hidden_size]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class AudioProjection extends ModalityAlignment {
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param audioHiddenSize 音频编码器输出维度
     * @param hiddenSize 统一隐藏空间维度
     */
    public AudioProjection(String name, int audioHiddenSize, int hiddenSize) {
        super(name, audioHiddenSize, hiddenSize);
    }
}
