package io.leavesfly.tinyai.nl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nl.block.NestedLearningBlock;
import io.leavesfly.tinyai.nl.core.NestedOptimizationLevel;
import io.leavesfly.tinyai.nl.model.NestedLearningModel;

/**
 * 嵌套学习演示
 * 展示多层级优化的基本概念
 * 
 * @author TinyAI Team
 */
public class NestedLearningDemo {
    
    public static void main(String[] args) {
        System.out.println("=== 嵌套学习演示 ===\n");
        
        // 1. 创建嵌套学习块
        int numLevels = 3;
        NestedLearningBlock block = new NestedLearningBlock("demo-block", numLevels);
        System.out.println("创建了" + numLevels + "层嵌套优化层级");
        
        // 2. 初始化块
        block.init();
        
        // 3. 创建模型
        NestedLearningModel model = new NestedLearningModel("demo-model", block);
        System.out.println("创建了嵌套学习模型\n");
        
        // 4. 模拟训练过程
        System.out.println("开始模拟训练过程：");
        for (int step = 0; step < 100; step++) {
            // 创建模拟输入
            NdArray inputData = NdArray.of(new float[]{1.0f, 2.0f, 3.0f}).reshape(Shape.of(1, 3));
            Variable input = new Variable(inputData);
            
            // 前向传播
            Variable output = model.forward(input);
            
            // 每10步打印一次信息
            if (step % 10 == 0) {
                System.out.println("\n步骤 " + step + ":");
                for (int i = 0; i < numLevels; i++) {
                    if (block.shouldUpdateLevel(i)) {
                        System.out.println("  层级 " + i + " 需要更新");
                    }
                }
            }
        }
        
        // 5. 打印统计信息
        System.out.println("\n" + block.getLevelStatistics());
        System.out.println(model.getMemorySystem().getStatistics());
        
        System.out.println("\n=== 演示完成 ===");
    }
}
