package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Qwen3 MoE功能演示程序
 * 
 * 展示如何使用Qwen3的MoE架构:
 * 1. 创建启用MoE的Qwen3模型
 * 2. 演示MoE的参数效率优势
 * 3. 查看专家使用统计
 * 4. 对比MoE vs 标准FFN
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3MoEDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("Qwen3 MoE功能演示");
        System.out.println("=".repeat(70));
        System.out.println();
        
        demo1_BasicMoE();
        System.out.println();
        
        demo2_ParameterEfficiency();
        System.out.println();
        
        demo3_ExpertStatistics();
        System.out.println();
        
        demo4_ComparisonWithStandard();
    }
    
    /**
     * Demo 1: 基础MoE功能
     */
    private static void demo1_BasicMoE() {
        System.out.println("【Demo 1】基础MoE功能");
        System.out.println("-".repeat(70));
        
        // 创建启用MoE的配置
        Qwen3Config config = Qwen3Config.createSmallConfig();
        config.setEnableMoE(true);
        config.setNumExperts(8);
        config.setExpertTopK(2);
        config.validate();
        
        System.out.println("✅ 创建MoE配置:");
        System.out.println("  - 专家数量: " + config.getNumExperts());
        System.out.println("  - Top-K选择: " + config.getExpertTopK());
        System.out.println("  - 激活率: " + (config.getExpertTopK() * 100.0 / config.getNumExperts()) + "%");
        System.out.println();
        
        // 创建MoE模型
        Qwen3Model model = new Qwen3Model("qwen3-moe", config);
        System.out.println("✅ 创建MoE模型:");
        model.printModelInfo();
    }
    
    /**
     * Demo 2: 参数效率对比
     */
    private static void demo2_ParameterEfficiency() {
        System.out.println("【Demo 2】参数效率对比");
        System.out.println("-".repeat(70));
        
        // 标准FFN配置
        Qwen3Config standardConfig = Qwen3Config.createSmallConfig();
        standardConfig.setEnableMoE(false);
        long standardParams = standardConfig.estimateParameterCount();
        
        // MoE配置
        Qwen3Config moeConfig = Qwen3Config.createSmallConfig();
        moeConfig.setEnableMoE(true);
        moeConfig.setNumExperts(8);
        moeConfig.setExpertTopK(2);
        long moeParams = moeConfig.estimateParameterCount();
        
        System.out.println("【标准FFN模型】");
        System.out.println("  总参数量: " + formatParams(standardParams));
        System.out.println("  激活参数: " + formatParams(standardParams) + " (100%)");
        System.out.println();
        
        System.out.println("【MoE模型】");
        System.out.println("  总参数量: " + formatParams(moeParams));
        double activationRate = moeConfig.getExpertTopK() * 100.0 / moeConfig.getNumExperts();
        long activatedParams = (long) (moeParams * activationRate / 100.0);
        System.out.println("  激活参数: " + formatParams(activatedParams) + 
                          String.format(" (%.1f%%)", activationRate));
        System.out.println();
        
        System.out.println("【效率分析】");
        double paramExpansion = moeParams * 1.0 / standardParams;
        System.out.println(String.format("  参数扩展: %.2fx", paramExpansion));
        System.out.println(String.format("  计算效率: ~%.1fx (相对于稠密模型)", 
            1.0 / (activationRate / 100.0)));
        System.out.println("  优势: 参数容量大幅提升,但推理成本接近标准模型");
    }
    
    /**
     * Demo 3: 专家使用统计
     */
    private static void demo3_ExpertStatistics() {
        System.out.println("【Demo 3】专家使用统计");
        System.out.println("-".repeat(70));
        
        // 创建MoE模型
        Qwen3Config config = Qwen3Config.createSmallConfig();
        config.setEnableMoE(true);
        config.setNumExperts(8);
        config.setExpertTopK(2);
        
        Qwen3Model model = new Qwen3Model("qwen3-moe", config);
        
        // 运行一些推理
        System.out.println("运行推理测试...");
        int batchSize = 2;
        int seqLen = 10;
        int[][] inputData = new int[batchSize][seqLen];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                inputData[b][t] = (int) (Math.random() * 1000);
            }
        }
        
        Variable input = new Variable(NdArray.of(inputData));
        model.forward(input);
        
        // 获取第一层的统计信息
        System.out.println();
        System.out.println("专家使用统计 (第0层):");
        Qwen3Block block = (Qwen3Block) model.getModule();
        Qwen3TransformerBlock layer0 = block.getLayers().get(0);
        Qwen3MoEBlock.ExpertUsageStats stats = layer0.getMoEStats();
        
        if (stats != null) {
            System.out.println(stats);
        } else {
            System.out.println("  (未启用MoE)");
        }
    }
    
    /**
     * Demo 4: MoE vs 标准FFN对比
     */
    private static void demo4_ComparisonWithStandard() {
        System.out.println("【Demo 4】MoE vs 标准FFN对比");
        System.out.println("-".repeat(70));
        
        // 准备测试数据
        int batchSize = 2;
        int seqLen = 8;
        int[][] inputData = new int[batchSize][seqLen];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                inputData[b][t] = (int) (Math.random() * 1000);
            }
        }
        Variable input = new Variable(NdArray.of(inputData));
        
        // 创建标准模型
        Qwen3Config standardConfig = Qwen3Config.createSmallConfig();
        standardConfig.setEnableMoE(false);
        Qwen3Model standardModel = new Qwen3Model("qwen3-standard", standardConfig);
        
        // 创建MoE模型
        Qwen3Config moeConfig = Qwen3Config.createSmallConfig();
        moeConfig.setEnableMoE(true);
        moeConfig.setNumExperts(8);
        moeConfig.setExpertTopK(2);
        Qwen3Model moeModel = new Qwen3Model("qwen3-moe", moeConfig);
        
        System.out.println("【标准FFN模型】");
        System.out.println("  架构: Transformer + SwiGLU FFN");
        System.out.println("  参数量: " + formatParams(standardConfig.estimateParameterCount()));
        
        // 运行推理
        long startTime = System.currentTimeMillis();
        Variable standardOutput = standardModel.forward(input);
        long standardTime = System.currentTimeMillis() - startTime;
        
        Shape outputShape = standardOutput.getValue().getShape();
        System.out.println("  输出形状: " + outputShape);
        System.out.println("  推理时间: " + standardTime + "ms");
        System.out.println();
        
        System.out.println("【MoE模型】");
        System.out.println("  架构: Transformer + MoE(8专家,Top-2)");
        System.out.println("  参数量: " + formatParams(moeConfig.estimateParameterCount()));
        
        // 运行推理
        startTime = System.currentTimeMillis();
        Variable moeOutput = moeModel.forward(input);
        long moeTime = System.currentTimeMillis() - startTime;
        
        outputShape = moeOutput.getValue().getShape();
        System.out.println("  输出形状: " + outputShape);
        System.out.println("  推理时间: " + moeTime + "ms");
        System.out.println();
        
        System.out.println("【对比总结】");
        System.out.println("  ✅ MoE优势:");
        System.out.println("    - 参数容量更大 (约" + 
            String.format("%.1fx", moeConfig.estimateParameterCount() * 1.0 / standardConfig.estimateParameterCount()) + 
            ")");
        System.out.println("    - 每次仅激活25%参数,推理效率高");
        System.out.println("    - 支持任务专门化,不同专家处理不同模式");
        System.out.println("  ✅ 标准FFN优势:");
        System.out.println("    - 实现更简单,训练更稳定");
        System.out.println("    - 无需负载均衡等复杂机制");
    }
    
    /**
     * 格式化参数数量
     */
    private static String formatParams(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.format("%d", count);
        }
    }
}
