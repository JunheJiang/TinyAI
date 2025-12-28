package io.leavesfly.tinyai.omni.encoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

/**
 * 2D位置编码
 * 
 * 为图像patches提供位置信息,使模型能够感知patch的空间位置。
 * 使用可学习的位置嵌入,在训练过程中自动学习最优的位置表示。
 * 
 * 输入: [batch, num_patches, hidden_size]
 * 输出: [1, num_patches, hidden_size] (可广播)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Position2D extends Module {
    
    private final int numPatches;
    private final int hiddenSize;
    private final Parameter positionEmbedding;
    
    public Position2D(String name, int numPatches, int hiddenSize) {
        super(name);
        this.numPatches = numPatches;
        this.hiddenSize = hiddenSize;
        
        NdArray posEmbData = NdArray.of(Shape.of(1, numPatches, hiddenSize));
        this.positionEmbedding = registerParameter("pos_emb", new Parameter(posEmbData));
        
        init();
    }
    
    @Override
    public void resetParameters() {
        NdArray data = positionEmbedding.data();
        float[] array = data.getArray();
        
        java.util.Random random = new java.util.Random(42);
        for (int i = 0; i < array.length; i++) {
            array[i] = (float) (random.nextGaussian() * 0.02);
        }
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        return positionEmbedding;
    }
    
    public int getNumPatches() {
        return numPatches;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
}
