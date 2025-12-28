package io.leavesfly.tinyai.omni.alignment;

/**
 * 图像特征投影
 * 
 * 将图像编码器的输出投影到统一的隐藏空间。
 * 
 * 输入: [batch, num_patches, image_hidden_size]
 * 输出: [batch, num_patches, hidden_size]
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ImageProjection extends ModalityAlignment {
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param imageHiddenSize 图像编码器输出维度
     * @param hiddenSize 统一隐藏空间维度
     */
    public ImageProjection(String name, int imageHiddenSize, int hiddenSize) {
        super(name, imageHiddenSize, hiddenSize);
    }
}
