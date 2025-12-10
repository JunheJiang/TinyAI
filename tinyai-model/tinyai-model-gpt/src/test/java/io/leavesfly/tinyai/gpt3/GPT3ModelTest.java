package io.leavesfly.tinyai.gpt3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

/**
 * GPT3Model 单元测试
 * 
 * 测试覆盖：
 * 1. 模型创建（工厂方法）- 仅测试Small和Medium模型实例化，Large和XL仅测试配置
 * 2. 前向传播
 * 3. 文本生成
 * 4. 模型信息输出
 * 5. 边界条件测试
 * 6. 异常处理
 * 
 * 内存使用说明：
 * - 所有测试在默认JVM堆设置下运行（通常为512MB-1GB）
 * - 序列长度限制在1024以内，批次大小限制在8以内
 * - 注意力矩阵内存复杂度：O(batch_size * n_heads * seq_len^2)
 * - 例：1批次×1024序列长度 ≈ 193MB，16批次×64序列长度 ≈ 50MB
 * 
 * 注意事项：
 * - Large模型（1.3B参数）和XL模型（175B参数）在单元测试中会导致OutOfMemoryError
 * - 这些超大模型仅测试配置验证和参数估算，不进行实际实例化
 * - 生产环境使用超大模型需要配置足够的JVM堆内存（例如：-Xmx32g或更大）
 * - 测试最大序列长度2048需要至少-Xmx4g的JVM堆内存
 * 
 * @author leavesfly
 */
public class GPT3ModelTest {
    
    private GPT3Model smallModel;
    
    @Before
    public void setUp() {
        // 每个测试前创建小型模型
        smallModel = GPT3Model.createSmallModel("test-gpt3-small");
    }
    
    // ==================== 模型创建测试 ====================
    
    @Test
    public void testCreateSmallModel() {
        GPT3Model model = GPT3Model.createSmallModel("gpt3-small");
        assertNotNull("小型模型不应为null", model);
        assertEquals("gpt3-small", model.getName());
        
        GPT3Config config = model.getConfig();
        assertEquals(768, config.getNEmbd());
        assertEquals(12, config.getNLayer());
        assertEquals(12, config.getNHead());
    }
    
    @Test
    public void testCreateMediumModel() {
        GPT3Model model = GPT3Model.createMediumModel("gpt3-medium");
        assertNotNull("中型模型不应为null", model);
        assertEquals("gpt3-medium", model.getName());
        
        GPT3Config config = model.getConfig();
        assertEquals(1024, config.getNEmbd());
        assertEquals(24, config.getNLayer());
    }
    
    @Test
    public void testLargeModelConfig() {
        // 注意：由于Large模型（1.3B参数）会导致OOM，仅测试配置而不实例化模型
        GPT3Config config = GPT3Config.createLargeConfig();
        assertNotNull("大型配置不应为null", config);
        assertEquals(2048, config.getNEmbd());
        assertEquals(32, config.getNHead());
        assertEquals(24, config.getNLayer());
        
        // 验证参数估算在合理范围内
        long paramCount = config.estimateParameterCount();
        assertTrue("大型模型参数应在1.2B-1.6B之间", 
            paramCount > 1_200_000_000L && paramCount < 1_600_000_000L);
    }
    
    @Test
    public void testXLModelConfig() {
        // 注意：由于XL模型（175B参数）会导致OOM，仅测试配置而不实例化模型
        GPT3Config config = GPT3Config.createXLConfig();
        assertNotNull("超大型配置不应为null", config);
        assertEquals(12288, config.getNEmbd());
        assertEquals(96, config.getNLayer());
        assertEquals(96, config.getNHead());
        
        // 验证参数估算在合理范围内
        long paramCount = config.estimateParameterCount();
        assertTrue("超大型模型参数应在170B-185B之间", 
            paramCount > 170_000_000_000L && paramCount < 185_000_000_000L);
    }
    
    @Test
    public void testCustomModelCreation() {
        GPT3Config customConfig = new GPT3Config();
        customConfig.setNEmbd(512);
        customConfig.setNLayer(6);
        customConfig.setNHead(8);
        customConfig.setNInner(2048);
        
        GPT3Model model = new GPT3Model("custom-gpt3", customConfig);
        assertNotNull(model);
        assertEquals("custom-gpt3", model.getName());
        assertEquals(512, model.getConfig().getNEmbd());
    }
    
    // ==================== 前向传播测试 ====================
    
    @Test
    public void testForwardPassSingleBatch() {
        // 创建输入: batch_size=1, seq_len=10
        NdArray tokenIds = NdArray.of(Shape.of(1, 10));
        
        // 填充一些token IDs (0-50256范围内)
        for (int i = 0; i < 10; i++) {
            tokenIds.set(i * 100, 0, i);
        }
        
        Variable input = new Variable(tokenIds);
        Variable output = smallModel.forward(input);
        
        assertNotNull("输出不应为null", output);
        assertNotNull("输出值不应为null", output.getValue());
        
        // 验证输出形状: (batch_size, seq_len, vocab_size)
        Shape outputShape = output.getValue().getShape();
        assertEquals("输出批次大小应为1", 1, outputShape.getDimension(0));
        assertEquals("输出序列长度应为10", 10, outputShape.getDimension(1));
        assertEquals("输出词汇表大小应为50257", 50257, outputShape.getDimension(2));
    }
    
    @Test
    public void testForwardPassMultipleBatches() {
        // 创建输入: batch_size=4, seq_len=20
        NdArray tokenIds = NdArray.of(Shape.of(4, 20));
        
        Variable input = new Variable(tokenIds);
        Variable output = smallModel.forward(input);
        
        assertNotNull(output);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(4, outputShape.getDimension(0));
        assertEquals(20, outputShape.getDimension(1));
        assertEquals(50257, outputShape.getDimension(2));
    }
    
    @Test
    public void testForwardPassLongSequence() {
        // 测试较长序列512（单批次内存友好）
        NdArray tokenIds = NdArray.of(Shape.of(2, 512));
        
        Variable input = new Variable(tokenIds);
        Variable output = smallModel.forward(input);
        
        assertNotNull(output);
        assertEquals(512, output.getValue().getShape().getDimension(1));
    }
    
    // ==================== 文本生成测试 ====================
    
    @Test
    public void testGenerateSequenceBasic() {
        // 创建prompt: batch_size=1, seq_len=5
        NdArray promptIds = NdArray.of(Shape.of(1, 5));
        
        // 生成10个新token
        NdArray generated = smallModel.generateSequence(promptIds, 10);
        
        assertNotNull("生成结果不应为null", generated);
        
        // 验证生成序列长度: 原始5 + 新生成10 = 15
        assertEquals("生成序列长度应为15", 15, generated.getShape().getDimension(1));
    }
    
    @Test
    public void testGenerateSequenceZeroTokens() {
        NdArray promptIds = NdArray.of(Shape.of(1, 10));
        
        // 生成0个新token（应该返回原始prompt）
        NdArray generated = smallModel.generateSequence(promptIds, 0);
        
        assertNotNull(generated);
        assertEquals(10, generated.getShape().getDimension(1));
    }
    
    @Test
    public void testGenerateSequenceMultipleBatches() {
        // 测试批量生成
        NdArray promptIds = NdArray.of(Shape.of(3, 8));
        
        NdArray generated = smallModel.generateSequence(promptIds, 20);
        
        assertNotNull(generated);
        assertEquals("批次大小应保持为3", 3, generated.getShape().getDimension(0));
        assertEquals("生成序列长度应为28", 28, generated.getShape().getDimension(1));
    }
    
    // ==================== 边界条件测试 ====================
    
    @Test
    public void testMinimalSequenceLength() {
        // 测试最小序列长度1
        NdArray tokenIds = NdArray.of(Shape.of(1, 1));
        tokenIds.set(100, 0, 0);
        
        Variable input = new Variable(tokenIds);
        Variable output = smallModel.forward(input);
        
        assertNotNull(output);
        assertEquals(1, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testLongSequenceLength() {
        // 测试较长序列长度1024（注意：2048会导致OOM）
        // 最大序列长度2048的测试需要足够的内存（如-Xmx4g）
        NdArray tokenIds = NdArray.of(Shape.of(1, 1024));
        
        Variable input = new Variable(tokenIds);
        Variable output = smallModel.forward(input);
        
        assertNotNull(output);
        assertEquals(1024, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testSingleBatch() {
        NdArray tokenIds = NdArray.of(Shape.of(1, 10));
        
        Variable input = new Variable(tokenIds);
        Variable output = smallModel.forward(input);
        
        assertNotNull(output);
        assertEquals(1, output.getValue().getShape().getDimension(0));
    }
    
    @Test
    public void testLargeBatch() {
        // 测试较大批次（降低序列长度以避免OOM）
        NdArray tokenIds = NdArray.of(Shape.of(8, 32));
        
        Variable input = new Variable(tokenIds);
        Variable output = smallModel.forward(input);
        
        assertNotNull(output);
        assertEquals(8, output.getValue().getShape().getDimension(0));
    }
    
    // ==================== 异常处理测试 ====================
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidInputDimensions() {
        // 创建1维输入（应该是2维）
        NdArray tokenIds = NdArray.of(Shape.of(10));
        Variable input = new Variable(tokenIds);
        
        // 应该抛出异常
        smallModel.forward(input);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSequenceTooLong() {
        // 创建超过最大长度的序列（2048+1）
        NdArray tokenIds = NdArray.of(Shape.of(1, 2049));
        Variable input = new Variable(tokenIds);
        
        // 应该抛出异常
        smallModel.forward(input);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNullInput() {
        // 传入null应抛出异常
        smallModel.forward((Variable[]) null);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testEmptyInput() {
        // 传入空数组应抛出异常
        smallModel.forward(new Variable[0]);
    }
    
    // ==================== 模型信息测试 ====================
    
    @Test
    public void testModelName() {
        assertEquals("test-gpt3-small", smallModel.getName());
    }
    
    @Test
    public void testModelConfig() {
        GPT3Config config = smallModel.getConfig();
        assertNotNull("配置不应为null", config);
        
        // 验证小型模型配置
        assertEquals(768, config.getNEmbd());
        assertEquals(12, config.getNLayer());
        assertEquals(12, config.getNHead());
        assertEquals(3072, config.getNInner());
    }
    
    @Test
    public void testPrintModelInfo() {
        // 测试打印模型信息不抛出异常
        try {
            smallModel.printModelInfo();
        } catch (Exception e) {
            fail("打印模型信息不应抛出异常: " + e.getMessage());
        }
    }
    
    // ==================== 模型一致性测试 ====================
    
    @Test
    public void testConsecutiveForwardPasses() {
        NdArray tokenIds = NdArray.of(Shape.of(1, 10));
        Variable input = new Variable(tokenIds);
        
        // 执行两次前向传播
        Variable output1 = smallModel.forward(input);
        Variable output2 = smallModel.forward(input);
        
        // 两次输出形状应该相同
        assertEquals(output1.getValue().getShape().toString(), 
                    output2.getValue().getShape().toString());
    }
    
    @Test
    public void testDifferentBatchSizes() {
        // 测试相同模型可以处理不同批次大小
        NdArray input1 = NdArray.of(Shape.of(1, 10));
        NdArray input2 = NdArray.of(Shape.of(4, 10));
        NdArray input3 = NdArray.of(Shape.of(8, 10));
        
        Variable output1 = smallModel.forward(new Variable(input1));
        Variable output2 = smallModel.forward(new Variable(input2));
        Variable output3 = smallModel.forward(new Variable(input3));
        
        assertEquals(1, output1.getValue().getShape().getDimension(0));
        assertEquals(4, output2.getValue().getShape().getDimension(0));
        assertEquals(8, output3.getValue().getShape().getDimension(0));
    }
    
    @Test
    public void testDifferentSequenceLengths() {
        // 测试相同模型可以处理不同序列长度
        NdArray input1 = NdArray.of(Shape.of(1, 10));
        NdArray input2 = NdArray.of(Shape.of(1, 50));
        NdArray input3 = NdArray.of(Shape.of(1, 100));
        
        Variable output1 = smallModel.forward(new Variable(input1));
        Variable output2 = smallModel.forward(new Variable(input2));
        Variable output3 = smallModel.forward(new Variable(input3));
        
        assertEquals(10, output1.getValue().getShape().getDimension(1));
        assertEquals(50, output2.getValue().getShape().getDimension(1));
        assertEquals(100, output3.getValue().getShape().getDimension(1));
    }
}
