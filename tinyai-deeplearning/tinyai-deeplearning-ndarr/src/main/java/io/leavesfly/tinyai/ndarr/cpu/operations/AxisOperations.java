package io.leavesfly.tinyai.ndarr.cpu.operations;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayValidator;

/**
 * 轴操作类
 * <p>提供沿指定轴进行的各种操作，包括最大值、最小值、argMax等</p>
 * <p>注意：当前实现主要支持最后两个轴的优化操作</p>
 */
public class AxisOperations {

    /**
     * 沿指定轴查找最大值的索引
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最大值索引数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu argMax(NdArrayCpu array, int axis) {
        ArrayValidator.validateAxis(axis, array.shape.getDimNum());

        int normalizedAxis = axis < 0 ? axis + array.shape.getDimNum() : axis;
        int dimNum = array.shape.getDimNum();

        // 对于多维数组，我们只支持最后两个轴的查找
        if (normalizedAxis == dimNum - 2) {
            return argMaxSecondLastAxis(array);
        } else if (normalizedAxis == dimNum - 1) {
            return argMaxLastAxis(array);
        }
        throw new IllegalArgumentException(
                String.format("不支持的轴参数: %d，仅支持 %d(列) 或 %d(行)", axis, dimNum - 2, dimNum - 1)
        );
    }

    /**
     * 沿倒数第二个轴查找最大值的索引（按行查找每列的最大值索引）
     */
    private static NdArrayCpu argMaxSecondLastAxis(NdArrayCpu array) {
        int dimNum = array.shape.getDimNum();
        int[] newDims = new int[dimNum];
        for (int i = 0; i < dimNum - 2; i++) {
            newDims[i] = array.shape.getDimension(i);
        }
        newDims[dimNum - 2] = 1;
        newDims[dimNum - 1] = array.shape.getDimension(dimNum - 1);

        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        int lastDimSize = array.shape.getDimension(dimNum - 1);
        int secondLastDimSize = array.shape.getDimension(dimNum - 2);
        int batchSize = array.shape.size() / (lastDimSize * secondLastDimSize);

        for (int batch = 0; batch < batchSize; batch++) {
            for (int j = 0; j < lastDimSize; j++) {
                float maxValue = Float.NEGATIVE_INFINITY;
                int maxIndex = -1;
                for (int i = 0; i < secondLastDimSize; i++) {
                    int index = batch * (lastDimSize * secondLastDimSize) + i * lastDimSize + j;
                    if (maxValue < array.buffer[index]) {
                        maxValue = array.buffer[index];
                        maxIndex = i;
                    }
                }
                int resultIndex = batch * (lastDimSize * 1) + 0 * lastDimSize + j;
                result.buffer[resultIndex] = maxIndex;
            }
        }
        return result;
    }

    /**
     * 沿最后一个轴查找最大值的索引（按列查找每行的最大值索引）
     */
    private static NdArrayCpu argMaxLastAxis(NdArrayCpu array) {
        int dimNum = array.shape.getDimNum();
        int[] newDims = new int[dimNum];
        for (int i = 0; i < dimNum - 2; i++) {
            newDims[i] = array.shape.getDimension(i);
        }
        newDims[dimNum - 2] = array.shape.getDimension(dimNum - 2);
        newDims[dimNum - 1] = 1;

        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        int lastDimSize = array.shape.getDimension(dimNum - 1);
        int secondLastDimSize = array.shape.getDimension(dimNum - 2);
        int batchSize = array.shape.size() / (lastDimSize * secondLastDimSize);

        for (int batch = 0; batch < batchSize; batch++) {
            for (int i = 0; i < secondLastDimSize; i++) {
                float maxValue = Float.NEGATIVE_INFINITY;
                int maxIndex = -1;
                for (int j = 0; j < lastDimSize; j++) {
                    int index = batch * (lastDimSize * secondLastDimSize) + i * lastDimSize + j;
                    if (maxValue < array.buffer[index]) {
                        maxValue = array.buffer[index];
                        maxIndex = j;
                    }
                }
                int resultIndex = batch * (1 * secondLastDimSize) + i * 1 + 0;
                result.buffer[resultIndex] = maxIndex;
            }
        }
        return result;
    }

    /**
     * 沿指定轴查找最大值（优化版本，仅支持最后两个轴）
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最大值数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu max(NdArrayCpu array, int axis) {
        ArrayValidator.validateAxis(axis, array.shape.getDimNum());

        int normalizedAxis = axis < 0 ? axis + array.shape.getDimNum() : axis;
        int dimNum = array.shape.getDimNum();

        // 对于多维数组，我们只支持最后两个轴的查找
        if (normalizedAxis == dimNum - 1) {
            return maxLastAxis(array);
        } else if (normalizedAxis == dimNum - 2) {
            return maxSecondLastAxis(array);
        }
        throw new IllegalArgumentException(
                String.format("不支持的轴参数: %d，仅支持 %d(列) 或 %d(行)", axis, dimNum - 2, dimNum - 1)
        );
    }

    /**
     * 沿最后一个轴查找最大值（按列查找每行的最大值）
     */
    private static NdArrayCpu maxLastAxis(NdArrayCpu array) {
        int dimNum = array.shape.getDimNum();
        int[] newDims = new int[dimNum];
        for (int i = 0; i < dimNum - 1; i++) {
            newDims[i] = array.shape.getDimension(i);
        }
        newDims[dimNum - 1] = 1;

        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        int lastDimSize = array.shape.getDimension(dimNum - 1);
        int secondLastDimSize = array.shape.getDimension(dimNum - 2);
        int batchSize = array.shape.size() / (lastDimSize * secondLastDimSize);

        for (int batch = 0; batch < batchSize; batch++) {
            for (int i = 0; i < secondLastDimSize; i++) {
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < lastDimSize; j++) {
                    int index = batch * (lastDimSize * secondLastDimSize) + i * lastDimSize + j;
                    if (max < array.buffer[index]) {
                        max = array.buffer[index];
                    }
                }
                int resultIndex = batch * (secondLastDimSize * 1) + i * 1 + 0;
                result.buffer[resultIndex] = max;
            }
        }
        return result;
    }

    /**
     * 沿倒数第二个轴查找最大值（按行查找每列的最大值）
     */
    private static NdArrayCpu maxSecondLastAxis(NdArrayCpu array) {
        int dimNum = array.shape.getDimNum();
        int[] newDims = new int[dimNum];
        for (int i = 0; i < dimNum - 2; i++) {
            newDims[i] = array.shape.getDimension(i);
        }
        newDims[dimNum - 2] = 1;
        newDims[dimNum - 1] = array.shape.getDimension(dimNum - 1);

        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        int lastDimSize = array.shape.getDimension(dimNum - 1);
        int secondLastDimSize = array.shape.getDimension(dimNum - 2);
        int batchSize = array.shape.size() / (lastDimSize * secondLastDimSize);

        for (int batch = 0; batch < batchSize; batch++) {
            for (int j = 0; j < lastDimSize; j++) {
                float max = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < secondLastDimSize; i++) {
                    int index = batch * (lastDimSize * secondLastDimSize) + i * lastDimSize + j;
                    if (max < array.buffer[index]) {
                        max = array.buffer[index];
                    }
                }
                int resultIndex = batch * (1 * lastDimSize) + 0 * lastDimSize + j;
                result.buffer[resultIndex] = max;
            }
        }
        return result;
    }

    /**
     * 沿指定轴查找最小值（优化版本，仅支持最后两个轴）
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最小值数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu min(NdArrayCpu array, int axis) {
        ArrayValidator.validateAxis(axis, array.shape.getDimNum());

        int normalizedAxis = axis < 0 ? axis + array.shape.getDimNum() : axis;
        int dimNum = array.shape.getDimNum();

        // 对于多维数组，我们只支持最后两个轴的查找
        if (normalizedAxis == dimNum - 1) {
            return minLastAxis(array);
        } else if (normalizedAxis == dimNum - 2) {
            return minSecondLastAxis(array);
        }
        throw new IllegalArgumentException(
                String.format("不支持的轴参数: %d，仅支持 %d(列) 或 %d(行)", axis, dimNum - 2, dimNum - 1)
        );
    }

    /**
     * 沿最后一个轴查找最小值（按列查找每行的最小值）
     */
    private static NdArrayCpu minLastAxis(NdArrayCpu array) {
        int dimNum = array.shape.getDimNum();
        int[] newDims = new int[dimNum];
        for (int i = 0; i < dimNum - 1; i++) {
            newDims[i] = array.shape.getDimension(i);
        }
        newDims[dimNum - 1] = 1;

        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        int lastDimSize = array.shape.getDimension(dimNum - 1);
        int secondLastDimSize = array.shape.getDimension(dimNum - 2);
        int batchSize = array.shape.size() / (lastDimSize * secondLastDimSize);

        for (int batch = 0; batch < batchSize; batch++) {
            for (int i = 0; i < secondLastDimSize; i++) {
                float min = Float.MAX_VALUE;
                for (int j = 0; j < lastDimSize; j++) {
                    int index = batch * (lastDimSize * secondLastDimSize) + i * lastDimSize + j;
                    if (min > array.buffer[index]) {
                        min = array.buffer[index];
                    }
                }
                int resultIndex = batch * (secondLastDimSize * 1) + i * 1 + 0;
                result.buffer[resultIndex] = min;
            }
        }
        return result;
    }

    /**
     * 沿倒数第二个轴查找最小值（按行查找每列的最小值）
     */
    private static NdArrayCpu minSecondLastAxis(NdArrayCpu array) {
        int dimNum = array.shape.getDimNum();
        int[] newDims = new int[dimNum];
        for (int i = 0; i < dimNum - 2; i++) {
            newDims[i] = array.shape.getDimension(i);
        }
        newDims[dimNum - 2] = 1;
        newDims[dimNum - 1] = array.shape.getDimension(dimNum - 1);

        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDims));

        int lastDimSize = array.shape.getDimension(dimNum - 1);
        int secondLastDimSize = array.shape.getDimension(dimNum - 2);
        int batchSize = array.shape.size() / (lastDimSize * secondLastDimSize);

        for (int batch = 0; batch < batchSize; batch++) {
            for (int j = 0; j < lastDimSize; j++) {
                float min = Float.MAX_VALUE;
                for (int i = 0; i < secondLastDimSize; i++) {
                    int index = batch * (lastDimSize * secondLastDimSize) + i * lastDimSize + j;
                    if (min > array.buffer[index]) {
                        min = array.buffer[index];
                    }
                }
                int resultIndex = batch * (1 * lastDimSize) + 0 * lastDimSize + j;
                result.buffer[resultIndex] = min;
            }
        }
        return result;
    }
}

