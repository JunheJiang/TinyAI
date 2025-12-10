package io.leavesfly.tinyai.nnet.v2.layer.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;

/**
 * 词嵌入层：将离散索引映射为连续向量表示。
 * <p>
 * 仅支持1D/2D索引输入：
 * - (seq_len,)
 * - (batch_size, seq_len)
 * 输出形状：
 * - 1D输入 -> (seq_len, embedding_dim)
 * - 2D输入 -> (batch_size, seq_len, embedding_dim)，若 seq_len==1 则压缩为 (batch_size, embedding_dim)
 */
public class Embedding extends Module {

    private final int numEmbeddings;
    private final int embeddingDim;
    private final Parameter weight;

    public Embedding(String name, int numEmbeddings, int embeddingDim) {
        super(name);
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

        NdArray weightData = NdArray.likeRandomN(Shape.of(numEmbeddings, embeddingDim));
        this.weight = registerParameter("weight", new Parameter(weightData));

        init();
    }

    @Override
    public void resetParameters() {
        // 使用较小方差的正态分布初始化嵌入
        Initializers.normal(weight.data(), 0f, 0.01f);
    }

    @Override
    public Variable forward(Variable... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Embedding requires one input indices Variable");
        }
        Variable indices = inputs[0];
        // 使用Variable的形状属性
        int dim = indices.ndim();

        if (dim == 1) {
            // 1D索引：直接使用getItem
            // 这里仍然需要获取NdArray进行索引转换，属于复杂索引操作
            NdArray idxValue = indices.getValue();
            int[] slices = NdArrayUtil.toInt(idxValue.getMatrix()[0]);
            return weight.getItem(slices, null);
        } else if (dim == 2) {
            // 使用Variable的size()方法
            int batchSize = indices.size(0);
            int seqLen = indices.size(1);

            // 批量处理：为每个样本获取embedding
            NdArray idxValue = indices.getValue();
            NdArray result = NdArray.zeros(Shape.of(batchSize, seqLen, embeddingDim));
            for (int i = 0; i < batchSize; i++) {
                int[] slices = NdArrayUtil.toInt(idxValue.getMatrix()[i]);
                Variable embRow = weight.getItem(slices, null);
                // 这里需要直接操作NdArray来填充结果，是合理的使用场景
                NdArray embVal = embRow.getValue();
                for (int j = 0; j < seqLen; j++) {
                    for (int k = 0; k < embeddingDim; k++) {
                        result.set(embVal.get(j, k), i, j, k);
                    }  
                }
            }

            if (seqLen == 1) {
                // 使用Variable的reshape方法
                return new Variable(result).reshape(Shape.of(batchSize, embeddingDim));
            }
            return new Variable(result);
        } else {
            throw new IllegalArgumentException(
                    String.format("Embedding only supports 1D or 2D index tensors, got %dD", dim));
        }
    }

    public Parameter getWeight() {
        return weight;
    }

    public int getNumEmbeddings() {
        return numEmbeddings;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }

    @Override
    public String toString() {
        return "Embedding{" +
                "name='" + name + '\'' +
                ", numEmbeddings=" + numEmbeddings +
                ", embeddingDim=" + embeddingDim +
                '}';
    }
}

