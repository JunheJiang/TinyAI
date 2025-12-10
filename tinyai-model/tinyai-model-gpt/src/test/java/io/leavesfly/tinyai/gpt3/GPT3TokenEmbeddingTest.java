package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

/**
 * GPT3TokenEmbedding 单元测试
 * 
 * 测试覆盖：
 * 1. 嵌入层创建
 * 2. Token嵌入
 * 3. 位置嵌入
 * 4. 组合嵌入
 * 5. Dropout应用
 * 6. 边界条件
 * 7. 异常处理
 * 
 * @author leavesfly
 */
public class GPT3TokenEmbeddingTest {
    
    private GPT3TokenEmbedding embedding;
    private GPT3Config config;
    
    @Before
    public void setUp() {
        config = GPT3Config.createSmallConfig();
        embedding = new GPT3TokenEmbedding("test_embedding", config);
    }
    
    // ==================== 基础创建测试 ====================
    
    @Test
    public void testEmbeddingCreation() {
        assertNotNull("嵌入层不应为null", embedding);
        assertEquals("test_embedding", embedding.getName());
    }
    
    @Test
    public void testEmbeddingDimensions() {
        assertEquals("词汇表大小应为50257", 50257, embedding.getVocabSize());
        assertEquals("嵌入维度应为768", 768, embedding.getEmbeddingDim());
        assertEquals("最大位置数应为2048", 2048, embedding.getMaxPositions());
        assertEquals("Dropout概率应为0.1", 0.1f, embedding.getDropoutProb(), 0.001f);
    }
    
    @Test
    public void testParameterInitialization() {
        assertNotNull("Token嵌入参数不应为null", embedding.getTokenEmbedding());
        assertNotNull("位置嵌入参数不应为null", embedding.getPositionEmbedding());
        
        // 验证参数形状
        Shape tokenEmbedShape = embedding.getTokenEmbedding().data().getShape();
        assertEquals("Token嵌入应为2维", 2, tokenEmbedShape.getDimNum());
        assertEquals(50257, tokenEmbedShape.getDimension(0));
        assertEquals(768, tokenEmbedShape.getDimension(1));
        
        Shape posEmbedShape = embedding.getPositionEmbedding().data().getShape();
        assertEquals("位置嵌入应为2维", 2, posEmbedShape.getDimNum());
        assertEquals(2048, posEmbedShape.getDimension(0));
        assertEquals(768, posEmbedShape.getDimension(1));
    }
    
    // ==================== 前向传播测试 ====================
    
    @Test
    public void testForwardPassSingleBatch() {
        // 创建输入: batch_size=1, seq_len=10
        NdArray tokenIds = NdArray.of(Shape.of(1, 10));
        
        Variable output = embedding.forward(new Variable(tokenIds));
        
        assertNotNull("输出不应为null", output);
        assertNotNull("输出值不应为null", output.getValue());
        
        // 验证输出形状: (batch_size, seq_len, embedding_dim)
        Shape outputShape = output.getValue().getShape();
        assertEquals(3, outputShape.getDimNum());
        assertEquals(1, outputShape.getDimension(0));
        assertEquals(10, outputShape.getDimension(1));
        assertEquals(768, outputShape.getDimension(2));
    }
    
    @Test
    public void testForwardPassMultipleBatches() {
        // 创建输入: batch_size=4, seq_len=20
        NdArray tokenIds = NdArray.of(Shape.of(4, 20));
        
        Variable output = embedding.forward(new Variable(tokenIds));
        
        Shape outputShape = output.getValue().getShape();
        assertEquals(4, outputShape.getDimension(0));
        assertEquals(20, outputShape.getDimension(1));
        assertEquals(768, outputShape.getDimension(2));
    }
    
    @Test
    public void testForwardPassLongSequence() {
        // 测试较长序列
        NdArray tokenIds = NdArray.of(Shape.of(2, 512));
        
        Variable output = embedding.forward(new Variable(tokenIds));
        
        assertEquals(512, output.getValue().getShape().getDimension(1));
    }
    
    // ==================== Token ID范围测试 ====================
    
    @Test
    public void testValidTokenIds() {
        NdArray tokenIds = NdArray.of(Shape.of(1, 5));
        
        // 设置有效的token IDs (0-50256范围内)
        tokenIds.set(0, 0, 0);      // 最小值
        tokenIds.set(100, 0, 1);    // 中间值
        tokenIds.set(50256, 0, 2);  // 最大值
        tokenIds.set(1000, 0, 3);
        tokenIds.set(25000, 0, 4);
        
        Variable output = embedding.forward(new Variable(tokenIds));
        assertNotNull(output);
        assertEquals(5, output.getValue().getShape().getDimension(1));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidTokenIdNegative() {
        NdArray tokenIds = NdArray.of(Shape.of(1, 5));
        tokenIds.set(-1, 0, 0);  // 无效的负数token ID
        
        embedding.forward(new Variable(tokenIds));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidTokenIdTooLarge() {
        NdArray tokenIds = NdArray.of(Shape.of(1, 5));
        tokenIds.set(50257, 0, 0);  // 超出词汇表范围
        
        embedding.forward(new Variable(tokenIds));
    }
    
    // ==================== 序列长度测试 ====================
    
    @Test
    public void testMinimalSequenceLength() {
        // 最小序列长度1
        NdArray tokenIds = NdArray.of(Shape.of(1, 1));
        tokenIds.set(100, 0, 0);
        
        Variable output = embedding.forward(new Variable(tokenIds));
        
        assertNotNull(output);
        assertEquals(1, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testMaxSequenceLength() {
        // 最大序列长度2048
        NdArray tokenIds = NdArray.of(Shape.of(1, 2048));
        
        Variable output = embedding.forward(new Variable(tokenIds));
        
        assertNotNull(output);
        assertEquals(2048, output.getValue().getShape().getDimension(1));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSequenceTooLong() {
        // 超过最大序列长度
        NdArray tokenIds = NdArray.of(Shape.of(1, 2049));
        
        embedding.forward(new Variable(tokenIds));
    }
    
    // ==================== 批次大小测试 ====================
    
    @Test
    public void testSingleBatch() {
        NdArray tokenIds = NdArray.of(Shape.of(1, 10));
        
        Variable output = embedding.forward(new Variable(tokenIds));
        
        assertEquals(1, output.getValue().getShape().getDimension(0));
    }
    
    @Test
    public void testMultipleBatches() {
        NdArray tokenIds = NdArray.of(Shape.of(8, 10));
        
        Variable output = embedding.forward(new Variable(tokenIds));
        
        assertEquals(8, output.getValue().getShape().getDimension(0));
    }
    
    @Test
    public void testLargeBatch() {
        NdArray tokenIds = NdArray.of(Shape.of(32, 64));
        
        Variable output = embedding.forward(new Variable(tokenIds));
        
        assertEquals(32, output.getValue().getShape().getDimension(0));
    }
    
    // ==================== 不同配置测试 ====================
    
    @Test
    public void testCustomVocabSize() {
        GPT3Config customConfig = new GPT3Config();
        customConfig.setVocabSize(32000);
        customConfig.setNEmbd(512);
        customConfig.setNPositions(1024);
        
        GPT3TokenEmbedding customEmbedding = new GPT3TokenEmbedding("custom", customConfig);
        
        assertEquals(32000, customEmbedding.getVocabSize());
        assertEquals(512, customEmbedding.getEmbeddingDim());
        assertEquals(1024, customEmbedding.getMaxPositions());
    }
    
    @Test
    public void testCustomDropout() {
        GPT3Config customConfig = new GPT3Config();
        customConfig.setEmbdPdrop(0.2);
        
        GPT3TokenEmbedding customEmbedding = new GPT3TokenEmbedding("custom", customConfig);
        
        assertEquals(0.2f, customEmbedding.getDropoutProb(), 0.001f);
    }
    
    @Test
    public void testZeroDropout() {
        GPT3Config customConfig = new GPT3Config();
        customConfig.setEmbdPdrop(0.0);
        
        GPT3TokenEmbedding customEmbedding = new GPT3TokenEmbedding("custom", customConfig);
        
        assertEquals(0.0f, customEmbedding.getDropoutProb(), 0.001f);
    }
    
    // ==================== 异常处理测试 ====================
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidInputDimensions1D() {
        // 1维输入（应该是2维）
        NdArray tokenIds = NdArray.of(Shape.of(10));
        
        embedding.forward(new Variable(tokenIds));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidInputDimensions3D() {
        // 3维输入（应该是2维）
        NdArray tokenIds = NdArray.of(Shape.of(2, 10, 5));
        
        embedding.forward(new Variable(tokenIds));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNullInput() {
        embedding.forward((Variable) null);
    }
    
    // ==================== 一致性测试 ====================
    
    @Test
    public void testConsecutiveForwardPasses() {
        NdArray tokenIds = NdArray.of(Shape.of(2, 10));
        
        // 多次前向传播应该产生相同形状的输出
        Variable output1 = embedding.forward(new Variable(tokenIds));
        Variable output2 = embedding.forward(new Variable(tokenIds));
        Variable output3 = embedding.forward(new Variable(tokenIds));
        
        assertEquals(output1.getValue().getShape().toString(),
                    output2.getValue().getShape().toString());
        assertEquals(output2.getValue().getShape().toString(),
                    output3.getValue().getShape().toString());
    }
    
    @Test
    public void testDifferentSequenceLengths() {
        // 同一嵌入层应该能处理不同序列长度
        NdArray input1 = NdArray.of(Shape.of(1, 10));
        NdArray input2 = NdArray.of(Shape.of(1, 50));
        NdArray input3 = NdArray.of(Shape.of(1, 100));
        
        Variable output1 = embedding.forward(new Variable(input1));
        Variable output2 = embedding.forward(new Variable(input2));
        Variable output3 = embedding.forward(new Variable(input3));
        
        assertEquals(10, output1.getValue().getShape().getDimension(1));
        assertEquals(50, output2.getValue().getShape().getDimension(1));
        assertEquals(100, output3.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testDifferentBatchSizes() {
        // 同一嵌入层应该能处理不同批次大小
        NdArray input1 = NdArray.of(Shape.of(1, 10));
        NdArray input2 = NdArray.of(Shape.of(4, 10));
        NdArray input3 = NdArray.of(Shape.of(16, 10));
        
        Variable output1 = embedding.forward(new Variable(input1));
        Variable output2 = embedding.forward(new Variable(input2));
        Variable output3 = embedding.forward(new Variable(input3));
        
        assertEquals(1, output1.getValue().getShape().getDimension(0));
        assertEquals(4, output2.getValue().getShape().getDimension(0));
        assertEquals(16, output3.getValue().getShape().getDimension(0));
    }
    
    // ==================== ToString测试 ====================
    
    @Test
    public void testToString() {
        String str = embedding.toString();
        assertNotNull("toString不应为null", str);
        assertTrue("toString应包含类名", str.contains("GPT3TokenEmbedding"));
        assertTrue("toString应包含名称", str.contains("test_embedding"));
        assertTrue("toString应包含词汇表大小", str.contains("50257"));
        assertTrue("toString应包含嵌入维度", str.contains("768"));
    }
}
