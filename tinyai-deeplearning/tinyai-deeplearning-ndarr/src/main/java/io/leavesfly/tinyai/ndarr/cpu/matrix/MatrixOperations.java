package io.leavesfly.tinyai.ndarr.cpu.matrix;

import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;

/**
 * 矩阵运算操作类
 * <p>提供矩阵乘法、切片等矩阵相关操作</p>
 */
public class MatrixOperations {

    /**
     * 矩阵内积运算（矩阵乘法）
     *
     * <p>执行标准的矩阵乘法运算，要求第一个矩阵的列数等于第二个矩阵的行数</p>
     *
     * @param left  左操作数数组
     * @param right 右操作数数组
     * @return 矩阵乘法结果
     * @throws IllegalArgumentException 当数组不是矩阵或维度不匹配时抛出
     */
    public static NdArrayCpu dot(NdArrayCpu left, NdArrayCpu right) {
        // 对于多维数组，我们只支持最后两个维度的矩阵乘法
        if (left.shape.getDimNum() >= 2 && right.shape.getDimNum() >= 2) {
            int leftLastDim = left.shape.getDimension(left.shape.getDimNum() - 1);
            int leftSecondLastDim = left.shape.getDimension(left.shape.getDimNum() - 2);
            int rightLastDim = right.shape.getDimension(right.shape.getDimNum() - 1);
            int rightSecondLastDim = right.shape.getDimension(right.shape.getDimNum() - 2);

            if (leftLastDim != rightSecondLastDim) {
                throw new IllegalArgumentException(String.format("矩阵乘法维度不匹配：%s × %s，第一个矩阵的列数(%d)必须等于第二个矩阵的行数(%d)", left.shape, right.shape, leftLastDim, rightSecondLastDim));
            }

            // 计算结果形状
            int[] newDims = new int[Math.max(left.shape.getDimNum(), right.shape.getDimNum())];
            int maxDimNum = Math.max(left.shape.getDimNum(), right.shape.getDimNum());

            // 复制前面的维度
            for (int i = 0; i < maxDimNum - 2; i++) {
                int leftDim = (i < left.shape.getDimNum() - 2) ? left.shape.getDimension(i) : 1;
                int rightDim = (i < right.shape.getDimNum() - 2) ? right.shape.getDimension(i) : 1;
                newDims[i] = Math.max(leftDim, rightDim);
            }

            // 设置最后两个维度
            newDims[maxDimNum - 2] = leftSecondLastDim;
            newDims[maxDimNum - 1] = rightLastDim;

            NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

            // 实现多维数组的矩阵乘法逻辑
            int batchSize = result.shape.size() / (leftSecondLastDim * rightLastDim);

            // 支持对前置维度的广播
            int leftBatchSize = left.shape.size() / (leftSecondLastDim * leftLastDim);
            int rightBatchSize = right.shape.size() / (rightSecondLastDim * rightLastDim);

            for (int batch = 0; batch < batchSize; batch++) {
                int leftBatchIndex = (leftBatchSize == 1) ? 0 : batch % leftBatchSize;
                int rightBatchIndex = (rightBatchSize == 1) ? 0 : batch % rightBatchSize;

                for (int i = 0; i < leftSecondLastDim; i++) {
                    for (int j = 0; j < rightLastDim; j++) {
                        float sum = 0f;
                        for (int k = 0; k < leftLastDim; k++) {
                            int leftIndex = leftBatchIndex * (leftSecondLastDim * leftLastDim) + i * leftLastDim + k;
                            int rightIndex = rightBatchIndex * (rightSecondLastDim * rightLastDim) + k * rightLastDim + j;
                            sum += left.buffer[leftIndex] * right.buffer[rightIndex];
                        }
                        int resultIndex = batch * (leftSecondLastDim * rightLastDim) + i * rightLastDim + j;
                        result.buffer[resultIndex] = sum;
                    }
                }
            }
            return result;
        }

        throw new IllegalArgumentException("操作需要至少二维数组");
    }

    /**
     * 获取数组的子集（切片操作）
     *
     * @param array      数组
     * @param rowSlices  行索引数组，null表示选择所有行
     * @param colSlices  列索引数组，null表示选择所有列
     * @return 切片结果数组
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    public static NdArrayCpu getItem(NdArrayCpu array, int[] rowSlices, int[] colSlices) {
        // 对于多维数组，我们只支持最后两个维度的切片操作
        if (array.shape.getDimNum() >= 2) {
            int lastDimSize = array.shape.getDimension(array.shape.getDimNum() - 1);
            int secondLastDimSize = array.shape.getDimension(array.shape.getDimNum() - 2);

            if (rowSlices != null && colSlices != null) {
                if (rowSlices.length != colSlices.length) {
                    throw new IllegalArgumentException(String.format("行索引数组长度(%d)必须等于列索引数组长度(%d)", rowSlices.length, colSlices.length));
                }

                NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(1, colSlices.length));
                for (int i = 0; i < colSlices.length; i++) {
                    int index = rowSlices[i] * lastDimSize + colSlices[i];
                    result.buffer[i] = array.buffer[index];
                }
                return result;
            }

            if (colSlices == null) {
                colSlices = NdArrayUtil.getSeq(lastDimSize);
            }
            if (rowSlices == null) {
                rowSlices = NdArrayUtil.getSeq(secondLastDimSize);
            }

            NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(rowSlices.length, colSlices.length));
            for (int i = 0; i < rowSlices.length; i++) {
                for (int j = 0; j < colSlices.length; j++) {
                    int index = rowSlices[i] * lastDimSize + colSlices[j];
                    result.buffer[i * result.getShape().getColumn() + j] = array.buffer[index];
                }
            }
            return result;
        }

        throw new IllegalArgumentException("操作需要至少二维数组");
    }

    /**
     * 设置数组的子集（切片赋值操作）
     *
     * @param array      数组
     * @param rowSlices  行索引数组，null表示选择所有行
     * @param colSlices  列索引数组，null表示选择所有列
     * @param data       要设置的数据
     * @return 当前数组实例
     * @throws IllegalArgumentException 当数组不是矩阵或参数不合法时抛出
     */
    public static NdArrayCpu setItem(NdArrayCpu array, int[] rowSlices, int[] colSlices, float[] data) {
        // 对于多维数组，我们只支持最后两个维度的切片操作
        if (array.shape.getDimNum() >= 2) {
            int lastDimSize = array.shape.getDimension(array.shape.getDimNum() - 1);

            if (rowSlices != null && colSlices != null) {
                if (rowSlices.length != colSlices.length) {
                    throw new IllegalArgumentException(String.format("行索引数组长度(%d)必须等于列索引数组长度(%d)", rowSlices.length, colSlices.length));
                }

                for (int i = 0; i < colSlices.length; i++) {
                    int index = rowSlices[i] * lastDimSize + colSlices[i];
                    array.buffer[index] = data[i];
                }
                return array;
            }
        }

        throw new IllegalArgumentException("操作需要至少二维数组");
    }
}

