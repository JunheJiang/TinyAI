package io.leavesfly.tinyai.nl.model;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nl.block.NestedLearningBlock;
import io.leavesfly.tinyai.nl.block.SelfModifyingBlock;

/**
 * Hope模型
 * 实现完整的嵌套学习范式，包括自修改能力
 * 
 * @author TinyAI Team
 */
public class HopeModel extends NestedLearningModel {
    
    private SelfModifyingBlock selfModifyingBlock;
    private boolean enableSelfModification;
    
    public HopeModel(String name, NestedLearningBlock block, SelfModifyingBlock selfModBlock) {
        super(name, block);
        this.selfModifyingBlock = selfModBlock;
        this.enableSelfModification = true;
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        Variable output = super.forward(inputs);
        
        // 如果启用自修改，应用自修改块
        if (enableSelfModification && selfModifyingBlock != null && output != null) {
            output = selfModifyingBlock.layerForward(output);
        }
        
        return output;
    }
    
    public void evaluatePerformance(float performance) {
        if (selfModifyingBlock != null) {
            selfModifyingBlock.evaluateAndModify(performance);
        }
    }
    
    public void setEnableSelfModification(boolean enable) {
        this.enableSelfModification = enable;
    }
    
    public SelfModifyingBlock getSelfModifyingBlock() {
        return selfModifyingBlock;
    }
}
