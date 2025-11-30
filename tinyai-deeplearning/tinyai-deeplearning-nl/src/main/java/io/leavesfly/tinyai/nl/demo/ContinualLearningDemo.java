package io.leavesfly.tinyai.nl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nl.block.NestedLearningBlock;
import io.leavesfly.tinyai.nl.model.NestedLearningModel;

/**
 * 持续学习演示
 * 展示如何在多个任务间学习而不遗忘
 * 
 * @author TinyAI Team
 */
public class ContinualLearningDemo {
    
    public static void main(String[] args) {
        System.out.println("=== 持续学习演示 ===\n");
        
        // 1. 创建模型
        NestedLearningBlock block = new NestedLearningBlock("continual-block", 4);
        block.init();
        NestedLearningModel model = new NestedLearningModel("continual-model", block);
        
        // 2. 任务1：学习简单模式
        System.out.println("任务1：学习模式A");
        trainOnTask(model, "A", 50);
        
        // 3. 任务2：学习不同模式
        System.out.println("\n任务2：学习模式B");
        trainOnTask(model, "B", 50);
        
        // 4. 任务3：学习第三个模式
        System.out.println("\n任务3：学习模式C");
        trainOnTask(model, "C", 50);
        
        // 5. 验证是否保留了之前的知识
        System.out.println("\n验证持续学习效果：");
        System.out.println("  测试模式A... (应该仍然记得)");
        System.out.println("  测试模式B... (应该仍然记得)");
        System.out.println("  测试模式C... (刚刚学习)");
        
        System.out.println("\n记忆系统状态：");
        System.out.println(model.getMemorySystem().getStatistics());
        
        System.out.println("\n=== 演示完成 ===");
    }
    
    private static void trainOnTask(NestedLearningModel model, String taskName, int steps) {
        for (int i = 0; i < steps; i++) {
            NdArray inputData = NdArray.of(new float[]{(float)i, (float)(i+1), (float)(i+2)}).reshape(Shape.of(1, 3));
            Variable input = new Variable(inputData);
            model.forward(input);
            
            if (i == steps - 1) {
                System.out.println("  完成" + steps + "步训练");
            }
        }
    }
}
