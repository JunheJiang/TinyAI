package io.leavesfly.tinyai.nl.block;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v1.Block;
import io.leavesfly.tinyai.nnet.v1.LayerAble;

/**
 * 自修改块（SelfModifyingBlock）
 * 实现可以动态修改自身结构和参数的神经网络块
 * 
 * <p>该块能够根据输入数据或性能反馈动态调整自身的网络结构、
 * 激活函数或超参数，模拟生物神经系统的可塑性。</p>
 * 
 * @author TinyAI Team
 */
public class SelfModifyingBlock extends Block {
    
    /**
     * 修改阈值
     */
    private float modificationThreshold;
    
    /**
     * 性能历史
     */
    private float[] performanceHistory;
    
    /**
     * 历史记录索引
     */
    private int historyIndex;
    
    /**
     * 是否启用自修改
     */
    private boolean enableSelfModification;
    
    /**
     * 修改计数器
     */
    private int modificationCount;
    
    public SelfModifyingBlock(String name, Shape inputShape) {
        super(name, inputShape);
        this.modificationThreshold = 0.1f;
        this.performanceHistory = new float[100];
        this.historyIndex = 0;
        this.enableSelfModification = true;
        this.modificationCount = 0;
    }
    
    public SelfModifyingBlock(String name) {
        this(name, null);
    }
    
    @Override
    public void init() {
        for (LayerAble layer : layers) {
            layer.init();
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            return null;
        }
        
        Variable x = inputs[0];
        
        if (!layers.isEmpty()) {
            Variable y = layers.get(0).layerForward(x);
            for (int i = 1; i < layers.size(); i++) {
                y = layers.get(i).layerForward(y);
            }
            return y;
        }
        
        return x;
    }
    
    /**
     * 根据性能反馈决定是否修改结构
     */
    public void evaluateAndModify(float performance) {
        if (!enableSelfModification) {
            return;
        }
        
        performanceHistory[historyIndex % performanceHistory.length] = performance;
        historyIndex++;
        
        if (shouldModify()) {
            modifyStructure();
        }
    }
    
    /**
     * 判断是否应该修改
     */
    private boolean shouldModify() {
        if (historyIndex < 10) {
            return false;
        }
        
        int recent = Math.min(10, historyIndex);
        float recentAvg = 0.0f;
        for (int i = 0; i < recent; i++) {
            int idx = (historyIndex - 1 - i) % performanceHistory.length;
            recentAvg += performanceHistory[idx];
        }
        recentAvg /= recent;
        
        return recentAvg < modificationThreshold;
    }
    
    /**
     * 修改网络结构
     */
    private void modifyStructure() {
        modificationCount++;
        // 简化实现：这里可以添加层、删除层、改变参数等
    }
    
    public int getModificationCount() {
        return modificationCount;
    }
    
    public void setModificationThreshold(float threshold) {
        this.modificationThreshold = threshold;
    }
    
    public void setEnableSelfModification(boolean enable) {
        this.enableSelfModification = enable;
    }
}
