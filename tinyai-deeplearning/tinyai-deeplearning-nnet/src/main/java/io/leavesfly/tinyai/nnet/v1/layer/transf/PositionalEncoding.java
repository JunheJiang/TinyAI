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
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("PositionalEncoding layer requires input");
        }
        
        Variable input = inputs[0];
        NdArray inputData = input.getValue();
        Shape inputShape = inputData.getShape();
        
        // 输入形状: (batch_size, seq_len, d_model)
        if (inputShape.getDimNum() != 3) {
            throw new IllegalArgumentException(
                String.format("PositionalEncoding expects 3D input (batch, seq, d_model), got %dD", 
                    inputShape.getDimNum())
            );
        }
        
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        int dModel = inputShape.getDimension(2);
        
        // 验证序列长度
        if (seqLen > maxSeqLength) {
            throw new IllegalArgumentException(
                String.format("Sequence length %d exceeds maximum %d", seqLen, maxSeqLength)
            );
        }
        
        // ============================================================================
        // 使用Variable算子构造位置编码，避免NdArray直接操作
        // ============================================================================
        
        // 将预计算的位置编码转为Variable
        Variable posEncVar = new Variable(posEncoding);
        posEncVar.setRequireGrad(false);  // 位置编码不需要梯度
        
        // 截取对应长度: [从0到seqLen)
        // 使用slice操作截取前seqLen行
        int[] rowSlices = new int[seqLen];
        for (int i = 0; i < seqLen; i++) {
            rowSlices[i] = i;
        }
        Variable posEncSlice = posEncVar.getItem(rowSlices, null);  // (seq_len, d_model)
        
        // 广播到batch维度: (seq_len, d_model) -> (1, seq_len, d_model) -> (batch, seq_len, d_model)
        posEncSlice = posEncSlice.reshape(Shape.of(1, seqLen, dModel));
        posEncSlice = posEncSlice.broadcastTo(Shape.of(batchSize, seqLen, dModel));
        
        // 将位置编码加到输入
        Variable output = input.add(posEncSlice);
        
        // TODO: 当Variable支持dropout时，添加: if (dropout) { output = output.dropout(dropoutRate); }
        
        return output;
    }

}