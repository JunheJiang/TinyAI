package io.leavesfly.tinyai.gpt3;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * GPT3Config 单元测试
 * 
 * 测试覆盖：
 * 1. 默认配置创建
 * 2. 预设配置工厂方法
 * 3. 自定义配置
 * 4. 配置验证
 * 5. 参数估算
 * 6. 边界条件
 * 7. 异常处理
 * 
 * @author leavesfly
 */
public class GPT3ConfigTest {
    
    // ==================== 基础配置测试 ====================
    
    @Test
    public void testDefaultConfig() {
        GPT3Config config = new GPT3Config();
        
        // 验证默认值
        assertEquals("默认词汇表大小应为50257", 50257, config.getVocabSize());
        assertEquals("默认序列长度应为2048", 2048, config.getNPositions());
        assertEquals("默认嵌入维度应为768", 768, config.getNEmbd());
        assertEquals("默认层数应为12", 12, config.getNLayer());
        assertEquals("默认注意力头数应为12", 12, config.getNHead());
        assertEquals("默认FFN维度应为3072", 3072, config.getNInner());
        assertEquals("默认激活函数应为gelu", "gelu", config.getActivationFunction());
        
        // 验证dropout配置
        assertEquals("默认残差dropout应为0.1", 0.1, config.getResidPdrop(), 0.001);
        assertEquals("默认嵌入dropout应为0.1", 0.1, config.getEmbdPdrop(), 0.001);
        assertEquals("默认注意力dropout应为0.1", 0.1, config.getAttnPdrop(), 0.001);
        
        // 验证GPT-3特有配置
        assertTrue("默认应启用并行注意力", config.isParallelAttention());
        assertFalse("默认不启用RoPE", config.isUseRotaryEmbedding());
        assertFalse("默认不启用稀疏注意力", config.isSparseAttention());
    }
    
    @Test
    public void testSmallConfig() {
        GPT3Config config = GPT3Config.createSmallConfig();
        
        assertEquals(768, config.getNEmbd());
        assertEquals(12, config.getNLayer());
        assertEquals(12, config.getNHead());
        assertEquals(3072, config.getNInner());
        assertTrue(config.isParallelAttention());
        
        // 验证配置有效性
        try {
            config.validate();
        } catch (Exception e) {
            fail("小型配置应该有效");
        }
    }
    
    @Test
    public void testMediumConfig() {
        GPT3Config config = GPT3Config.createMediumConfig();
        
        assertEquals(1024, config.getNEmbd());
        assertEquals(24, config.getNLayer());
        assertEquals(16, config.getNHead());
        assertEquals(4096, config.getNInner());
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("中型配置应该有效");
        }
    }
    
    @Test
    public void testLargeConfig() {
        GPT3Config config = GPT3Config.createLargeConfig();
        
        assertEquals(2048, config.getNEmbd());
        assertEquals(24, config.getNLayer());
        assertEquals(32, config.getNHead());
        assertEquals(8192, config.getNInner());
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("大型配置应该有效");
        }
    }
    
    @Test
    public void testXLConfig() {
        GPT3Config config = GPT3Config.createXLConfig();
        
        assertEquals(12288, config.getNEmbd());
        assertEquals(96, config.getNLayer());
        assertEquals(96, config.getNHead());
        assertEquals(49152, config.getNInner());
        assertTrue(config.isSparseAttention());
        assertTrue(config.isGradientCheckpointing());
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("超大型配置应该有效");
        }
    }
    
    // ==================== 配置修改测试 ====================
    
    @Test
    public void testCustomConfig() {
        GPT3Config config = new GPT3Config();
        
        // 修改基础配置
        config.setVocabSize(32000);
        config.setNPositions(1024);
        config.setNEmbd(512);
        config.setNLayer(8);
        config.setNHead(8);
        config.setNInner(2048);
        
        // 验证修改生效
        assertEquals(32000, config.getVocabSize());
        assertEquals(1024, config.getNPositions());
        assertEquals(512, config.getNEmbd());
        assertEquals(8, config.getNLayer());
        assertEquals(8, config.getNHead());
        assertEquals(2048, config.getNInner());
    }
    
    // ==================== 配置验证测试 ====================
    
    @Test
    public void testInvalidVocabSize() {
        GPT3Config config = new GPT3Config();
        config.setVocabSize(0);
        
        try {
            config.validate();
            fail("词汇表大小为0应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("词汇表大小"));
        }
    }
    
    @Test
    public void testInvalidPositions() {
        GPT3Config config = new GPT3Config();
        config.setNPositions(-1);
        
        try {
            config.validate();
            fail("位置数为负应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("最大位置数"));
        }
    }
    
    @Test
    public void testEmbedDimNotDivisibleByHeads() {
        GPT3Config config = new GPT3Config();
        config.setNEmbd(100);
        config.setNHead(7);
        
        try {
            config.validate();
            fail("嵌入维度不能被头数整除应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("整除"));
        }
    }
    
    // ==================== 参数估算测试 ====================
    
    @Test
    public void testSmallModelParameterEstimation() {
        GPT3Config config = GPT3Config.createSmallConfig();
        long paramCount = config.estimateParameterCount();
        
        // 小型模型大约125M参数 (实际约163M)
        assertTrue("小型模型参数应在150M-200M之间，实际: " + paramCount,
            paramCount > 150_000_000 && paramCount < 200_000_000);
    }
    
    @Test
    public void testMediumModelParameterEstimation() {
        GPT3Config config = GPT3Config.createMediumConfig();
        long paramCount = config.estimateParameterCount();
        
        // 中型模型大约350M参数 (根据实际计算调整范围)
        assertTrue("中型模型参数应在350M-450M之间，实际: " + paramCount,
            paramCount > 350_000_000 && paramCount < 450_000_000);
    }
    
    @Test
    public void testLargeModelParameterEstimation() {
        GPT3Config config = GPT3Config.createLargeConfig();
        long paramCount = config.estimateParameterCount();
        
        // 大型模型大约1.3B参数 (根据实际计算调整范围)
        assertTrue("大型模型参数应在1.2B-1.6B之间，实际: " + paramCount,
            paramCount > 1_200_000_000L && paramCount < 1_600_000_000L);
    }
    
    @Test
    public void testXLModelParameterEstimation() {
        GPT3Config config = GPT3Config.createXLConfig();
        long paramCount = config.estimateParameterCount();
        
        // 超大型模型大约175B参数 (根据实际计算调整范围)
        assertTrue("超大型模型参数应在170B-185B之间，实际: " + paramCount,
            paramCount > 170_000_000_000L && paramCount < 185_000_000_000L);
    }
    
    // ==================== 边界条件测试 ====================
    
    @Test
    public void testMinimalValidConfig() {
        GPT3Config config = new GPT3Config();
        config.setVocabSize(1000);
        config.setNPositions(128);
        config.setNEmbd(64);
        config.setNLayer(2);
        config.setNHead(4);
        config.setNInner(256);
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("最小有效配置应通过验证");
        }
    }
    
    @Test
    public void testZeroDropout() {
        GPT3Config config = new GPT3Config();
        config.setResidPdrop(0.0);
        config.setEmbdPdrop(0.0);
        config.setAttnPdrop(0.0);
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("0 dropout应该有效");
        }
    }
    
    @Test
    public void testInvalidDropoutNegative() {
        GPT3Config config = new GPT3Config();
        config.setResidPdrop(-0.1);
        
        try {
            config.validate();
            fail("负数dropout应抛出异常");
        } catch (IllegalArgumentException e) {
            // 预期异常
        }
    }
    
    @Test
    public void testInvalidDropoutTooLarge() {
        GPT3Config config = new GPT3Config();
        config.setResidPdrop(1.0);
        
        try {
            config.validate();
            fail("dropout=1.0应抛出异常");
        } catch (IllegalArgumentException e) {
            // 预期异常
        }
    }
    
    // ==================== GPT-3特有配置测试 ====================
    
    @Test
    public void testParallelAttentionConfig() {
        GPT3Config config = new GPT3Config();
        
        config.setParallelAttention(false);
        assertFalse(config.isParallelAttention());
        
        config.setParallelAttention(true);
        assertTrue(config.isParallelAttention());
    }
    
    @Test
    public void testSparseAttentionConfig() {
        GPT3Config config = new GPT3Config();
        
        config.setSparseAttention(true);
        config.setSparseLocalWindow(128);
        config.setSparseStrideSize(64);
        
        assertTrue(config.isSparseAttention());
        assertEquals(128, config.getSparseLocalWindow());
        assertEquals(64, config.getSparseStrideSize());
    }
    
    @Test
    public void testRotaryEmbeddingConfig() {
        GPT3Config config = new GPT3Config();
        
        config.setUseRotaryEmbedding(true);
        config.setRotaryPct(0.25);
        config.setRotaryBase(10000.0);
        
        assertTrue(config.isUseRotaryEmbedding());
        assertEquals(0.25, config.getRotaryPct(), 0.001);
        assertEquals(10000.0, config.getRotaryBase(), 0.001);
    }
}
