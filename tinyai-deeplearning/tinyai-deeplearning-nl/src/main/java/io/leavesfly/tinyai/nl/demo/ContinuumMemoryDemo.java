package io.leavesfly.tinyai.nl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nl.memory.ContinuumMemorySystem;
import io.leavesfly.tinyai.nl.memory.MemoryType;

/**
 * 连续体记忆系统演示
 * 展示多时间尺度记忆管理
 * 
 * @author TinyAI Team
 */
public class ContinuumMemoryDemo {
    
    public static void main(String[] args) {
        System.out.println("=== 连续体记忆系统演示 ===\n");
        
        // 1. 创建记忆系统
        ContinuumMemorySystem memorySystem = new ContinuumMemorySystem();
        System.out.println("创建了连续体记忆系统");
        System.out.println(memorySystem.getStatistics());
        
        // 2. 存储记忆
        System.out.println("\n开始存储记忆：");
        for (int i = 0; i < 20; i++) {
            NdArray keyData = NdArray.of(new float[]{(float)i, (float)(i*2)}).reshape(io.leavesfly.tinyai.ndarr.Shape.of(1, 2));
            NdArray valueData = NdArray.of(new float[]{(float)(i*10)}).reshape(io.leavesfly.tinyai.ndarr.Shape.of(1, 1));
            
            Variable key = new Variable(keyData);
            Variable value = new Variable(valueData);
            
            // 存储到短期记忆
            memorySystem.store(key, value);
            
            if (i % 5 == 0) {
                System.out.println("  已存储 " + (i+1) + " 条记忆");
            }
        }
        
        // 3. 模拟时间推移，执行记忆整合
        System.out.println("\n模拟时间推移：");
        for (int step = 0; step < 200; step++) {
            memorySystem.update(step);
            
            if (step % 50 == 0 && step > 0) {
                System.out.println("\n步骤 " + step + ":");
                System.out.println(memorySystem.getStatistics());
            }
        }
        
        // 4. 检索记忆
        System.out.println("\n检索记忆测试：");
        NdArray queryData = NdArray.of(new float[]{5.0f, 10.0f}).reshape(io.leavesfly.tinyai.ndarr.Shape.of(1, 2));
        Variable query = new Variable(queryData);
        Variable retrieved = memorySystem.retrieve(query);
        
        if (retrieved != null) {
            System.out.println("成功检索到相关记忆");
        } else {
            System.out.println("未找到相关记忆");
        }
        
        System.out.println("\n=== 演示完成 ===");
    }
}
