package io.leavesfly.tinyai.omni.encoder;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Mel频谱图转换器
 * 
 * 将音频波形转换为Mel频谱图表示,这是音频处理的标准预处理步骤。
 * 
 * Mel频谱图特性:
 * - 频率轴使用Mel标度,更符合人耳听觉特性
 * - 时间轴通过短时傅里叶变换(STFT)获得
 * - 输出为二维矩阵: [mel_bins, time_frames]
 * 
 * 工作流程:
 * 1. 分帧: 将音频分成重叠的帧
 * 2. 加窗: 对每帧应用汉明窗
 * 3. FFT: 快速傅里叶变换
 * 4. Mel滤波器组: 应用Mel滤波器
 * 5. 对数变换: log(mel_spec + epsilon)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MelSpectrogram {
    
    private final int sampleRate;
    private final int melBins;
    private final int frameLengthMs;
    private final int frameShiftMs;
    private final int fftSize;
    
    /**
     * 构造函数
     * 
     * @param sampleRate 采样率(Hz)
     * @param melBins Mel频谱bin数量
     * @param frameLengthMs 帧长度(毫秒)
     * @param frameShiftMs 帧移位(毫秒)
     */
    public MelSpectrogram(int sampleRate, int melBins, int frameLengthMs, int frameShiftMs) {
        this.sampleRate = sampleRate;
        this.melBins = melBins;
        this.frameLengthMs = frameLengthMs;
        this.frameShiftMs = frameShiftMs;
        this.fftSize = computeFFTSize(sampleRate, frameLengthMs);
    }
    
    /**
     * 计算FFT大小(取大于等于帧长度的最小2的幂)
     */
    private int computeFFTSize(int sampleRate, int frameLengthMs) {
        int frameSamples = sampleRate * frameLengthMs / 1000;
        int fftSize = 1;
        while (fftSize < frameSamples) {
            fftSize *= 2;
        }
        return fftSize;
    }
    
    /**
     * 将音频波形转换为Mel频谱图
     * 
     * @param waveform 音频波形 [num_samples]
     * @return Mel频谱图 [mel_bins, time_frames]
     */
    public NdArray transform(NdArray waveform) {
        int[] shape = waveform.getShape().getShapeDims();
        if (shape.length != 1) {
            throw new IllegalArgumentException("waveform必须是1维数组,当前维度: " + shape.length);
        }
        
        float[] audioData = waveform.getArray();
        int numSamples = audioData.length;
        
        // 计算帧数
        int frameLengthSamples = sampleRate * frameLengthMs / 1000;
        int frameShiftSamples = sampleRate * frameShiftMs / 1000;
        int numFrames = (numSamples - frameLengthSamples) / frameShiftSamples + 1;
        
        if (numFrames <= 0) {
            throw new IllegalArgumentException(
                "音频太短,无法提取帧。需要至少 " + frameLengthSamples + " 个采样点"
            );
        }
        
        // 创建Mel频谱图矩阵
        float[][] melSpec = new float[melBins][numFrames];
        
        // 简化实现:使用能量谱近似Mel频谱
        // 生产环境应使用完整的STFT + Mel滤波器组实现
        for (int t = 0; t < numFrames; t++) {
            int startIdx = t * frameShiftSamples;
            
            // 提取当前帧
            float[] frame = extractFrame(audioData, startIdx, frameLengthSamples);
            
            // 应用汉明窗
            applyHammingWindow(frame);
            
            // 计算能量谱(简化版)
            float[] powerSpectrum = computePowerSpectrum(frame, fftSize);
            
            // 应用Mel滤波器(简化版)
            float[] melFrame = applyMelFilterBank(powerSpectrum);
            
            // 对数变换
            for (int i = 0; i < melBins; i++) {
                melSpec[i][t] = (float) Math.log(melFrame[i] + 1e-10);
            }
        }
        
        // 转换为NdArray
        return convertToNdArray(melSpec);
    }
    
    /**
     * 提取音频帧
     */
    private float[] extractFrame(float[] audio, int startIdx, int frameLength) {
        float[] frame = new float[frameLength];
        for (int i = 0; i < frameLength && (startIdx + i) < audio.length; i++) {
            frame[i] = audio[startIdx + i];
        }
        return frame;
    }
    
    /**
     * 应用汉明窗
     */
    private void applyHammingWindow(float[] frame) {
        int n = frame.length;
        for (int i = 0; i < n; i++) {
            float window = (float) (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1)));
            frame[i] *= window;
        }
    }
    
    /**
     * 计算功率谱(简化版FFT)
     */
    private float[] computePowerSpectrum(float[] frame, int fftSize) {
        // 简化实现:使用能量分布近似
        // 生产环境应使用真实FFT算法
        int numBins = fftSize / 2 + 1;
        float[] powerSpec = new float[numBins];
        
        // 计算每个频段的能量
        int frameLength = frame.length;
        int binSize = frameLength / numBins;
        
        for (int i = 0; i < numBins && i * binSize < frameLength; i++) {
            float energy = 0.0f;
            int start = i * binSize;
            int end = Math.min(start + binSize, frameLength);
            
            for (int j = start; j < end; j++) {
                energy += frame[j] * frame[j];
            }
            
            powerSpec[i] = energy / (end - start);
        }
        
        return powerSpec;
    }
    
    /**
     * 应用Mel滤波器组(简化版)
     */
    private float[] applyMelFilterBank(float[] powerSpectrum) {
        // 简化实现:线性插值到mel_bins
        // 生产环境应使用三角Mel滤波器
        float[] melOutput = new float[melBins];
        
        int spectrumLength = powerSpectrum.length;
        for (int i = 0; i < melBins; i++) {
            float ratio = (float) i / melBins;
            int idx = (int) (ratio * spectrumLength);
            
            if (idx < spectrumLength) {
                melOutput[i] = powerSpectrum[idx];
            }
        }
        
        return melOutput;
    }
    
    /**
     * 转换二维数组为NdArray
     */
    private NdArray convertToNdArray(float[][] melSpec) {
        int melBins = melSpec.length;
        int timeFrames = melSpec[0].length;
        
        float[] flatData = new float[melBins * timeFrames];
        for (int i = 0; i < melBins; i++) {
            for (int j = 0; j < timeFrames; j++) {
                flatData[i * timeFrames + j] = melSpec[i][j];
            }
        }
        
        return NdArray.of(flatData, Shape.of(melBins, timeFrames));
    }
    
    // ==================== Getter方法 ====================
    
    public int getSampleRate() {
        return sampleRate;
    }
    
    public int getMelBins() {
        return melBins;
    }
    
    public int getFrameLengthMs() {
        return frameLengthMs;
    }
    
    public int getFrameShiftMs() {
        return frameShiftMs;
    }
    
    public int getFftSize() {
        return fftSize;
    }
}
