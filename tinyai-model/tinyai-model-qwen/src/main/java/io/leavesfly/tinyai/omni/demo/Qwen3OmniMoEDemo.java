package io.leavesfly.tinyai.omni.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.omni.config.ModalityType;
import io.leavesfly.tinyai.omni.config.Qwen3OmniConfig;
import io.leavesfly.tinyai.omni.moe.Qwen3OmniMoELayer;

/**
 * Qwen3-Omni MoE功能演示程序
 * 
 * 展示如何使用MoE来提升多模态模型的效率:
 * 1. 参数扩展但计算量可控
 * 2. 模态感知的专家路由
 * 3. 负载均衡和统计
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3OmniMoEDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("Qwen3-Omni MoE功能演示");
        System.out.println("=".repeat(70));
        System.out.println();
        
        demo1_BasicMoE();
        System.out.println();
        
        demo2_ModalityAwareMoE();
        System.out.println();
        
        demo3_EfficiencyComparison();
        System.out.println();
        
        demo4_ExpertStatistics();
    }
    
    /**
     * Demo 1: 基础MoE功能
     */
    private static void demo1_BasicMoE() {
        System.out.println("【Demo 1】基础MoE功能");
        System.out.println("-".repeat(70));
        
        // 创建配置（启用MoE）
        Qwen3OmniConfig config = Qwen3OmniConfig.createTinyConfig();
        config.setEnableMoE(true);
        config.setNumExperts(4);
        config.setExpertTopK(2);
        
        System.out.println("配置信息:");
        System.out.println("  - 隐藏维度: " + config.getHiddenSize());
        System.out.println("  - 专家数量: " + config.getNumExperts());
        System.out.println("  - Top-K选择: " + config.getExpertTopK());
        System.out.println("  - 专家隐藏维度: " + config.getExpertHiddenSize());
        
        // 创建MoE层
        Qwen3OmniMoELayer moeLayer = new Qwen3OmniMoELayer("moe_demo", config);
        
        // 创建输入 [batch=2, seq_len=8, hidden=512]
        int batchSize = 2;
        int seqLen = 8;
        int hiddenSize = config.getHiddenSize();
        
        NdArray inputArray = NdArray.randn(Shape.of(batchSize, seqLen, hiddenSize));
        Variable input = new Variable(inputArray);
        
        // 前向传播
        System.out.println();
        System.out.println("前向传播:");
        System.out.println("  输入形状: " + input.getValue().getShape());
        
        Variable output = moeLayer.forward(input);
        
        System.out.println("  输出形状: " + output.getValue().getShape());
        System.out.println("  ✅ MoE层成功处理输入!");
    }
    
    /**
     * Demo 2: 模态感知的MoE路由
     */
    private static void demo2_ModalityAwareMoE() {
        System.out.println("【Demo 2】模态感知的MoE路由");
        System.out.println("-".repeat(70));
        
        // 创建配置
        Qwen3OmniConfig config = Qwen3OmniConfig.createSmallConfig();
        config.setEnableMoE(true);
        config.setNumExperts(6);  // 6个专家,每种模态倾向使用2个
        config.setExpertTopK(2);
        
        Qwen3OmniMoELayer moeLayer = new Qwen3OmniMoELayer("modality_aware_moe", config);
        
        // 创建输入
        int batchSize = 1;
        int seqLen = 16;
        int hiddenSize = config.getHiddenSize();
        
        NdArray inputArray = NdArray.randn(Shape.of(batchSize, seqLen, hiddenSize));
        Variable input = new Variable(inputArray);
        
        // 测试不同模态
        System.out.println("测试不同模态的专家路由:");
        
        for (ModalityType modality : ModalityType.values()) {
            moeLayer.resetStats();  // 重置统计
            
            // 使用模态信息计算
            Qwen3OmniMoELayer.MoEOutput output = moeLayer.computeMoE(input, modality);
            
            System.out.println();
            System.out.println("  模态: " + modality.getDescription());
            System.out.println("  负载均衡损失: " + String.format("%.6f", output.loadBalanceLoss));
            
            // 显示前3个token的专家选择
            int[][][] indices = output.topKResult.indices;
            float[][][] weights = output.topKResult.weights;
            
            System.out.print("  前3个token选择的专家: ");
            for (int t = 0; t < Math.min(3, seqLen); t++) {
                System.out.print("[");
                for (int k = 0; k < config.getExpertTopK(); k++) {
                    System.out.print("E" + indices[0][t][k]);
                    if (k < config.getExpertTopK() - 1) {
                        System.out.print(",");
                    }
                }
                System.out.print("] ");
            }
            System.out.println();
        }
    }
    
    /**
     * Demo 3: 效率对比
     */
    private static void demo3_EfficiencyComparison() {
        System.out.println("【Demo 3】MoE效率对比");
        System.out.println("-".repeat(70));
        
        // 创建两个配置对比
        Qwen3OmniConfig denseConfig = Qwen3OmniConfig.createSmallConfig();
        denseConfig.setEnableMoE(false);
        
        Qwen3OmniConfig moeConfig = Qwen3OmniConfig.createSmallConfig();
        moeConfig.setEnableMoE(true);
        moeConfig.setNumExperts(8);
        moeConfig.setExpertTopK(2);
        
        System.out.println("配置对比:");
        System.out.println();
        
        System.out.println("【稠密模型】");
        System.out.println("  - 隐藏维度: " + denseConfig.getHiddenSize());
        System.out.println("  - FFN中间维度: " + denseConfig.getIntermediateSize());
        System.out.println("  - 估算参数量: " + denseConfig.formatParameters());
        
        System.out.println();
        System.out.println("【MoE模型】");
        System.out.println("  - 隐藏维度: " + moeConfig.getHiddenSize());
        System.out.println("  - 专家数量: " + moeConfig.getNumExperts());
        System.out.println("  - 每个专家FFN维度: " + moeConfig.getExpertHiddenSize());
        System.out.println("  - Top-K激活: " + moeConfig.getExpertTopK() + " 个专家");
        System.out.println("  - 估算参数量: " + moeConfig.formatParameters());
        
        // 计算MoE的参数扩展和激活比例
        long denseFFNParams = 2L * moeConfig.getHiddenSize() * moeConfig.getIntermediateSize();
        long moeFFNParams = (long) moeConfig.getNumExperts() * 2L * 
                           moeConfig.getHiddenSize() * moeConfig.getExpertHiddenSize();
        long moeActiveFFNParams = (long) moeConfig.getExpertTopK() * 2L * 
                                 moeConfig.getHiddenSize() * moeConfig.getExpertHiddenSize();
        
        double paramExpansion = (double) moeFFNParams / denseFFNParams;
        double activationRatio = (double) moeActiveFFNParams / moeFFNParams * 100;
        
        System.out.println();
        System.out.println("【效率分析】");
        System.out.println("  - FFN参数扩展: " + String.format("%.1fx", paramExpansion));
        System.out.println("  - 稀疏激活比例: " + String.format("%.1f%%", activationRatio));
        System.out.println("  - 效率提升: 参数增加" + String.format("%.1fx", paramExpansion) + 
                         ", 但每次仅激活" + String.format("%.1f%%", activationRatio));
        System.out.println();
        System.out.println("  ✅ MoE实现了 \"大容量+高效率\" 的平衡!");
    }
    
    /**
     * Demo 4: 专家使用统计
     */
    private static void demo4_ExpertStatistics() {
        System.out.println("【Demo 4】专家使用统计");
        System.out.println("-".repeat(70));
        
        // 创建配置
        Qwen3OmniConfig config = Qwen3OmniConfig.createTinyConfig();
        config.setEnableMoE(true);
        config.setNumExperts(4);
        config.setExpertTopK(2);
        config.setExpertLoadBalance(true);
        
        Qwen3OmniMoELayer moeLayer = new Qwen3OmniMoELayer("stats_moe", config);
        
        // 模拟多轮前向传播
        int batchSize = 2;
        int seqLen = 10;
        int hiddenSize = config.getHiddenSize();
        int numIterations = 5;
        
        System.out.println("模拟 " + numIterations + " 轮前向传播:");
        System.out.println();
        
        for (int iter = 0; iter < numIterations; iter++) {
            NdArray inputArray = NdArray.randn(Shape.of(batchSize, seqLen, hiddenSize));
            Variable input = new Variable(inputArray);
            
            Qwen3OmniMoELayer.MoEOutput output = moeLayer.computeMoE(input, null);
            
            System.out.println("  第" + (iter + 1) + "轮 - 负载均衡损失: " + 
                             String.format("%.6f", output.loadBalanceLoss));
        }
        
        // 显示最终统计
        System.out.println();
        System.out.println("最终专家使用统计:");
        Qwen3OmniMoELayer.ExpertUsageStats stats = moeLayer.getUsageStats();
        
        long[] counts = stats.getUsageCount();
        float[] rates = stats.getUsageRate();
        
        for (int i = 0; i < config.getNumExperts(); i++) {
            System.out.println(String.format("  Expert%d: 使用%d次, 使用率%.2f%%",
                i, counts[i], rates[i] * 100));
        }
        
        System.out.println();
        System.out.println("  总调用: " + stats.getTotalCalls() + " 次");
        
        // 计算使用率的标准差（评估负载均衡效果）
        float avgRate = 1.0f / config.getNumExperts();
        float variance = 0.0f;
        for (float rate : rates) {
            float diff = rate - avgRate;
            variance += diff * diff;
        }
        float stdDev = (float) Math.sqrt(variance);
        
        System.out.println("  负载均衡度（标准差）: " + String.format("%.4f", stdDev));
        System.out.println("  （标准差越小,负载越均衡）");
        
        if (stdDev < 0.05f) {
            System.out.println();
            System.out.println("  ✅ 负载均衡效果优秀!");
        }
    }
}
