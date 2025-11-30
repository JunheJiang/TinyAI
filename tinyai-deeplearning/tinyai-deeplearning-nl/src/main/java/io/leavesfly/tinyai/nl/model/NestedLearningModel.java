package io.leavesfly.tinyai.nl.model;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nl.block.NestedLearningBlock;
import io.leavesfly.tinyai.nl.memory.ContinuumMemorySystem;
import io.leavesfly.tinyai.nl.optimizer.DeepOptimizer;

/**
 * 嵌套学习模型（NestedLearningModel）
 * 实现基于嵌套优化的完整模型
 * 
 * @author TinyAI Team
 */
public class NestedLearningModel {
    
    private String name;
    private NestedLearningBlock block;
    private ContinuumMemorySystem memorySystem;
    private DeepOptimizer optimizer;
    
    public NestedLearningModel(String name, NestedLearningBlock block) {
        this.name = name;
        this.block = block;
        this.memorySystem = new ContinuumMemorySystem();
    }
    
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            return null;
        }
        
        Variable output = block.layerForward(inputs);
        
        // 更新层级状态
        block.updateLevels();
        
        // 更新记忆系统
        if (memorySystem != null) {
            memorySystem.update(block.getCurrentStep());
        }
        
        return output;
    }
    
    public void setOptimizer(DeepOptimizer optimizer) {
        this.optimizer = optimizer;
    }
    
    public NestedLearningBlock getBlock() {
        return block;
    }
    
    public ContinuumMemorySystem getMemorySystem() {
        return memorySystem;
    }
}
