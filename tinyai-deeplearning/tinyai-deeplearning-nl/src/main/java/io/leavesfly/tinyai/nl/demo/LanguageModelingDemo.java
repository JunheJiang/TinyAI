package io.leavesfly.tinyai.nl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nl.block.MultiFrequencyAttention;
import io.leavesfly.tinyai.nl.block.NestedLearningBlock;
import io.leavesfly.tinyai.nl.model.NestedLearningModel;

/**
 * 语言建模演示
 * 展示嵌套学习在序列建模中的应用
 * 
 * @author TinyAI Team
 */
public class LanguageModelingDemo {
    
    public static void main(String[] args) {
        System.out.println("=== 语言建模演示 ===\n");
        
        // 1. 创建模型
        NestedLearningBlock block = new NestedLearningBlock("lm-block", 3);
        block.init();
        NestedLearningModel model = new NestedLearningModel("lm-model", block);
        
        // 2. 创建多频率注意力机制
        MultiFrequencyAttention attention = new MultiFrequencyAttention("mfa", 3, 64);
        attention.init();
        
        System.out.println("创建了语言模型，包含：");
        System.out.println("  - 3层嵌套优化层级");
        System.out.println("  - 多频率注意力机制");
        
        // 3. 模拟处理文本序列
        System.out.println("\n处理示例序列：");
        String[] tokens = {"hello", "world", "this", "is", "nested", "learning"};
        
        for (int i = 0; i < tokens.length; i++) {
            System.out.println("  Token " + (i+1) + ": " + tokens[i]);
            
            // 创建token embedding（简化）
            NdArray tokenEmb = NdArray.of(new float[]{(float)i, (float)(i*2), (float)(i*3)}).reshape(Shape.of(1, 3));
            Variable input = new Variable(tokenEmb);
            
            // 前向传播
            Variable output = model.forward(input);
            
            // 更新注意力记忆
            for (int freq = 0; freq < 3; freq++) {
                attention.updateMemory(freq, input, output);
            }
        }
        
        // 4. 打印统计信息
        System.out.println("\n模型状态：");
        System.out.println(block.getLevelStatistics());
        
        System.out.println("\n注意力机制状态：");
        System.out.println("  频率数量: " + attention.getNumFrequencies());
        for (int i = 0; i < attention.getNumFrequencies(); i++) {
            var memory = attention.getMemory(i);
            if (memory != null) {
                System.out.println("  频率" + i + " 记忆数: " + memory.getCurrentSize());
            }
        }
        
        System.out.println("\n=== 演示完成 ===");
    }
}
