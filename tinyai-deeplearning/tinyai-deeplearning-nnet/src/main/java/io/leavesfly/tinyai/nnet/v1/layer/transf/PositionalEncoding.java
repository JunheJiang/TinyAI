package io.leavesfly.tinyai.nnet.v1.layer.transf;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v1.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * 位置编码层实现
 * <p>
 * 位置编码用于给输入序列添加位置信息，因为Transformer本身没有循环或卷积结构来获取位置信息。
 * 使用正弦和余弦函数来生成位置编码：
 * <p>
 * PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 * <p>
 * 其中 pos 是位置，i 是维度
 */
public class PositionalEncoding extends Layer {

    private int maxSeqLength;
    private NdArray posEncoding;
    private boolean dropout;
    private double dropoutRate;


    public PositionalEncoding(String name) {
        super(name);
    }

    /**
     * 构造位置编码层
     *
     * @param name         层名称
     * @param dModel       模型维度
     * @param maxSeqLength 最大序列长度
     * @param dropoutRate  dropout比率
     */
    public PositionalEncoding(String name, int dModel, int maxSeqLength, double dropoutRate) {
        super(name);
        this.maxSeqLength = maxSeqLength;
        this.dropoutRate = dropoutRate;
        this.dropout = dropoutRate > 0.0;
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            int dModel = inputShape.getDimension(2);

            // 创建位置编码矩阵
            posEncoding = NdArray.zeros(Shape.of(maxSeqLength, dModel));

            // 计算位置编码
            for (int pos = 0; pos < maxSeqLength; pos++) {
                for (int i = 0; i < dModel; i++) {
                    double angle = pos / Math.pow(10000, (double) (2 * (i / 2)) / dModel);
                    if (i % 2 == 0) {
                        // 偶数维度使用sin
                        posEncoding.set((float) Math.sin(angle), pos, i);
                    } else {
                        // 奇数维度使用cos
                        posEncoding.set((float) Math.cos(angle), pos, i);
                    }
                }
            }

            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        //todo
        return null;
    }

}