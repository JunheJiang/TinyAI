package io.leavesfly.tinyai.omni.demo;

import io.leavesfly.tinyai.omni.config.Qwen3OmniConfig;
import io.leavesfly.tinyai.omni.model.Qwen3OmniModel;

/**
 * Qwen3-Omni 演示程序
 * 
 * 展示Qwen3-Omni全模态大模型的基本功能
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3OmniDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("Qwen3-Omni 全模态大模型演示程序");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // 演示1: 创建不同规模的模型
        demo1_CreateModels();
        System.out.println();
        
        // 演示2: 配置管理
        demo2_ConfigManagement();
        System.out.println();
        
        // 演示3: 模型信息
        demo3_ModelInfo();
        System.out.println();
        
        System.out.println("=".repeat(80));
        System.out.println("演示完成!");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 演示1: 创建不同规模的模型
     */
    private static void demo1_CreateModels() {
        System.out.println("【演示1】创建不同规模的Qwen3-Omni模型");
        System.out.println("-".repeat(80));
        
        try {
            // 创建Tiny模型
            System.out.println("1. 创建Tiny模型(教学用)...");
            Qwen3OmniModel tinyModel = Qwen3OmniModel.createTinyModel("qwen3-omni-tiny");
            System.out.println("   ✓ Tiny模型创建成功");
            System.out.println("   - " + tinyModel.getConfigSummary());
            
            // 创建Small模型
            System.out.println("\n2. 创建Small模型(实验用)...");
            Qwen3OmniModel smallModel = Qwen3OmniModel.createSmallModel("qwen3-omni-small");
            System.out.println("   ✓ Small模型创建成功");
            System.out.println("   - " + smallModel.getConfigSummary());
            
            // 创建Base模型
            System.out.println("\n3. 创建Base模型(标准规模)...");
            Qwen3OmniModel baseModel = Qwen3OmniModel.createBaseModel("qwen3-omni-base");
            System.out.println("   ✓ Base模型创建成功");
            System.out.println("   - " + baseModel.getConfigSummary());
            
            // 创建自定义配置模型
            System.out.println("\n4. 创建自定义配置模型...");
            Qwen3OmniConfig customConfig = new Qwen3OmniConfig();
            customConfig.setHiddenSize(512);
            customConfig.setNumHiddenLayers(8);
            customConfig.setNumAttentionHeads(8);
            customConfig.setNumKeyValueHeads(8);
            customConfig.setImageSize(224);
            customConfig.updateDerivedParams();
            customConfig.validate();
            
            Qwen3OmniModel customModel = new Qwen3OmniModel("qwen3-omni-custom", customConfig);
            System.out.println("   ✓ 自定义模型创建成功");
            System.out.println("   - 隐藏维度: " + customConfig.getHiddenSize());
            System.out.println("   - 层数: " + customConfig.getNumHiddenLayers());
            System.out.println("   - 参数量: " + customConfig.formatParameters());
            
        } catch (Exception e) {
            System.err.println("   ✗ 创建模型失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示2: 配置管理
     */
    private static void demo2_ConfigManagement() {
        System.out.println("【演示2】配置管理与验证");
        System.out.println("-".repeat(80));
        
        try {
            // 创建并验证配置
            System.out.println("1. 创建自定义配置...");
            Qwen3OmniConfig config = new Qwen3OmniConfig();
            
            // 基础配置
            config.setVocabSize(32000);
            config.setHiddenSize(768);
            config.setIntermediateSize(2112);
            config.setNumHiddenLayers(12);
            config.setNumAttentionHeads(12);
            config.setNumKeyValueHeads(12);
            config.setMaxPositionEmbeddings(2048);
            
            // 图像配置
            config.setImageSize(384);
            config.setPatchSize(16);
            config.setImageEncoderLayers(6);
            config.setImageHiddenSize(512);
            
            // 音频配置
            config.setAudioSampleRate(16000);
            config.setMelBins(80);
            config.setAudioEncoderLayers(6);
            config.setAudioHiddenSize(512);
            
            // 更新派生参数
            config.updateDerivedParams();
            
            System.out.println("   ✓ 配置创建完成");
            
            // 验证配置
            System.out.println("\n2. 验证配置有效性...");
            config.validate();
            System.out.println("   ✓ 配置验证通过");
            
            // 显示配置信息
            System.out.println("\n3. 配置详情:");
            System.out.println(config);
            
            // 显示计算属性
            System.out.println("\n4. 计算属性:");
            System.out.println("   - 头维度: " + config.getHeadDim());
            System.out.println("   - 键值组数: " + config.getNumKeyValueGroups());
            System.out.println("   - 图像Patch数: " + config.getNumImagePatches());
            System.out.println("   - 估算参数量: " + config.formatParameters());
            
        } catch (Exception e) {
            System.err.println("   ✗ 配置管理失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示3: 模型信息
     */
    private static void demo3_ModelInfo() {
        System.out.println("【演示3】模型信息查看");
        System.out.println("-".repeat(80));
        
        try {
            // 创建模型
            Qwen3OmniModel model = Qwen3OmniModel.createSmallModel("qwen3-omni-demo");
            
            // 打印模型信息
            System.out.println("1. 模型基本信息:");
            model.printModelInfo();
            
            // 打印模型toString
            System.out.println("\n2. 模型toString:");
            System.out.println(model);
            
        } catch (Exception e) {
            System.err.println("   ✗ 查看模型信息失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
