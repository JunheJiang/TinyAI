package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;

/**
 * GPT-3 Token嵌入层（完全基于V2 Module）
 * 
 * 负责将离散的token ID转换为连续的向量表示，包括：
 * 1. Token嵌入：将词汇ID映射到向量空间
 * 2. 位置嵌入：为每个序列位置学习位置向量
 * 3. Dropout正则化
 * 
 * @author leavesfly
 * @version 2.0 - 完全基于V2 API
 */
public class GPT3TokenEmbedding extends Module {
    
    private final int vocabSize;
    private final int embeddingDim;
    private final int maxPositions;
    private final float dropoutProb;
    
    // V2 参数
    private Parameter tokenEmbedding;      // Token嵌入矩阵 (vocabSize, embeddingDim)
    private Parameter positionEmbedding;   // 位置嵌入矩阵 (maxPositions, embeddingDim)
    
    // Dropout层
    private Dropout dropout;
    
    /**
     * 构造GPT-3 Token嵌入层
     * 
     * @param name 层名称
     * @param config GPT-3配置
     */
    public GPT3TokenEmbedding(String name, GPT3Config config) {
        super(name);
        
        this.vocabSize = config.getVocabSize();
        this.embeddingDim = config.getNEmbd();
        this.maxPositions = config.getNPositions();
        this.dropoutProb = (float) config.getEmbdPdrop();
        
        // 初始化参数和层
        initializeParameters(config);
    }
    
    /**
     * 初始化参数
     */
    private void initializeParameters(GPT3Config config) {
        // 1. 初始化Token嵌入矩阵
        NdArray tokenEmbedData = NdArray.likeRandomN(Shape.of(vocabSize, embeddingDim))
            .mulNum((float) config.getInitializerRange());
        tokenEmbedding = new Parameter(tokenEmbedData);
        registerParameter("token_embedding", tokenEmbedding);
        
        // 2. 初始化位置嵌入矩阵
        NdArray positionEmbedData = NdArray.likeRandomN(Shape.of(maxPositions, embeddingDim))
            .mulNum((float) config.getInitializerRange());
        positionEmbedding = new Parameter(positionEmbedData);
        registerParameter("position_embedding", positionEmbedding);
        
        // 3. 初始化Dropout
        dropout = new Dropout("embedding_dropout", dropoutProb);
        registerModule("dropout", dropout);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        Variable tokenIds = inputs[0];  // shape: (batchSize, sequenceLength)
        NdArray tokenData = tokenIds.getValue();
        
        int batchSize = tokenData.getShape().getDimension(0);
        int sequenceLength = tokenData.getShape().getDimension(1);
        
        // 验证序列长度
        if (sequenceLength > maxPositions) {
            throw new IllegalArgumentException(
                String.format("输入序列长度(%d)超过最大位置数(%d)", sequenceLength, maxPositions)
            );
        }
        
        // 1. 获取Token嵌入
        Variable tokenEmbeds = getTokenEmbeddings(tokenData, batchSize, sequenceLength);
        
        // 2. 获取位置嵌入
        Variable positionEmbeds = getPositionEmbeddings(sequenceLength, batchSize);
        
        // 3. 相加组合Token和位置嵌入
        Variable combined = tokenEmbeds.add(positionEmbeds);
        
        // 4. 应用Dropout
        Variable result = dropout.forward(combined);
        
        return result;
    }
    
    /**
     * 获取Token嵌入
     * 
     * @param tokenIds Token ID数组
     * @param batchSize 批次大小
     * @param sequenceLength 序列长度
     * @return Token嵌入变量
     */
    private Variable getTokenEmbeddings(NdArray tokenIds, int batchSize, int sequenceLength) {
        NdArray embeddings = NdArray.of(Shape.of(batchSize, sequenceLength, embeddingDim));
        NdArray tokenEmbedData = tokenEmbedding.data();
        
        // 对每个token ID查找对应的嵌入向量
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < sequenceLength; s++) {
                int tokenId = (int) tokenIds.get(b, s);
                
                // 验证token ID的有效性
                if (tokenId < 0 || tokenId >= vocabSize) {
                    throw new IllegalArgumentException(
                        String.format("Token ID %d out of vocabulary range [0, %d)", tokenId, vocabSize)
                    );
                }
                
                // 复制对应的嵌入向量
                for (int d = 0; d < embeddingDim; d++) {
                    float embeddingValue = tokenEmbedData.get(tokenId, d);
                    embeddings.set(embeddingValue, b, s, d);
                }
            }
        }
        
        return new Variable(embeddings);
    }
    
    /**
     * 获取位置嵌入
     * 
     * @param sequenceLength 序列长度
     * @param batchSize 批次大小
     * @return 位置嵌入变量
     */
    private Variable getPositionEmbeddings(int sequenceLength, int batchSize) {
        NdArray posEmbeds = NdArray.of(Shape.of(batchSize, sequenceLength, embeddingDim));
        NdArray positionEmbedData = positionEmbedding.data();
        
        // 为每个位置添加位置嵌入
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < sequenceLength; s++) {
                // 复制对应位置的嵌入向量
                for (int d = 0; d < embeddingDim; d++) {
                    float positionValue = positionEmbedData.get(s, d);
                    posEmbeds.set(positionValue, b, s, d);
                }
            }
        }
        
        return new Variable(posEmbeds);
    }
    
    // ==================== Getter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public int getEmbeddingDim() {
        return embeddingDim;
    }
    
    public int getMaxPositions() {
        return maxPositions;
    }
    
    public float getDropoutProb() {
        return dropoutProb;
    }
    
    public Parameter getTokenEmbedding() {
        return tokenEmbedding;
    }
    
    public Parameter getPositionEmbedding() {
        return positionEmbedding;
    }
    
    @Override
    public String toString() {
        return String.format("GPT3TokenEmbedding{name='%s', vocabSize=%d, embeddingDim=%d, maxPositions=%d, dropout=%.3f}",
            name, vocabSize, embeddingDim, maxPositions, dropoutProb);
    }
}
