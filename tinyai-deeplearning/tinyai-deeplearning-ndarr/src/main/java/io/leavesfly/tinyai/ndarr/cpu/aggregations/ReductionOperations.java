package io.leavesfly.tinyai.ndarr.cpu.aggregations;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayValidator;

/**
 * 聚合操作类
 * <p>提供各种聚合运算功能，包括求和、均值、方差、最大值、最小值等</p>
 */
public class ReductionOperations {

    /**
     * 元素累和运算，计算数组所有元素的总和
     *
     * @param array 数组
     * @return 所有元素的总和（标量）
     */
    public static NdArrayCpu sum(NdArrayCpu array) {
        float sum = 0f;
        for (float value : array.buffer) {
            sum += value;
        }
        return new NdArrayCpu(sum);
    }

    /**
     * 按轴聚合的通用方法，沿指定轴进行聚合运算
     *
     * @param array        数组
     * @param axis         聚合轴
     * @param operation    聚合操作函数
     * @param operationName 操作名称，用于错误提示
     * @return 聚合结果数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    private static NdArrayCpu axisOperation(NdArrayCpu array, int axis, java.util.function.Function<float[], Float> operation, String operationName) {
        ArrayValidator.validateAxis(axis, array.shape.getDimNum());

        // 计算新形状
        int[] newDimensions = new int[array.shape.getDimNum() - 1];
        int newIndex = 0;
        for (int i = 0; i < array.shape.getDimNum(); i++) {
            if (i != axis) {
                newDimensions[newIndex++] = array.shape.getDimension(i);
            }
        }

        ShapeCpu newShape = ShapeCpu.of(newDimensions);
        NdArrayCpu result = new NdArrayCpu(newShape);

        // 计算聚合
        int totalElementsInAxis = array.shape.getDimension(axis);

        // 遍历所有非axis维度的组合
        int[] indices = new int[array.shape.getDimNum()];
        int[] resultIndices = new int[newShape.getDimNum()];

        for (int i = 0; i < newShape.size(); i++) {
            // 将结果索引转换为多维索引
            int temp = i;
            for (int dim = newShape.getDimNum() - 1; dim >= 0; dim--) {
                resultIndices[dim] = temp % newShape.getDimension(dim);
                temp /= newShape.getDimension(dim);
            }

            // 构建完整的索引数组
            int resultIndex = 0;
            for (int dim = 0; dim < array.shape.getDimNum(); dim++) {
                if (dim == axis) {
                    indices[dim] = 0; // axis维度将在下面循环中变化
                } else {
                    indices[dim] = resultIndices[resultIndex++];
                }
            }

            // 计算沿axis维度的聚合
            float[] dataToAggregate = new float[totalElementsInAxis];
            for (int j = 0; j < totalElementsInAxis; j++) {
                indices[axis] = j;
                dataToAggregate[j] = array.get(indices);
            }
            result.buffer[i] = operation.apply(dataToAggregate);
        }
        return result;
    }

    /**
     * 矩阵均值运算，沿指定轴计算均值
     *
     * @param array 数组
     * @param axis  聚合轴，axis=0表示按列计算均值，axis=1表示按行计算均值
     * @return 均值运算结果数组
     */
    public static NdArrayCpu mean(NdArrayCpu array, int axis) {
        return axisOperation(array, axis, values -> {
            float sum = 0f;
            for (float value : values) {
                sum += value;
            }
            return sum / values.length;
        }, "均值计算");
    }

    /**
     * 矩阵方差运算，沿指定轴计算方差
     *
     * @param array 数组
     * @param axis  聚合轴，axis=0表示按列计算方差，axis=1表示按行计算方差
     * @return 方差运算结果数组
     */
    public static NdArrayCpu var(NdArrayCpu array, int axis) {
        return axisOperation(array, axis, values -> {
            // 计算均值
            float mean = 0f;
            for (float value : values) {
                mean += value;
            }
            mean /= values.length;

            // 计算方差
            float variance = 0f;
            for (float value : values) {
                variance += (value - mean) * (value - mean);
            }
            return variance / values.length;
        }, "方差计算");
    }

    /**
     * 矩阵累和运算，沿指定轴计算累和
     *
     * @param array 数组
     * @param axis  聚合轴，axis=0表示按列累和，axis=1表示按行累和
     * @return 累和运算结果数组
     */
    public static NdArrayCpu sum(NdArrayCpu array, int axis) {
        return axisOperation(array, axis, values -> {
            float sum = 0f;
            for (float value : values) {
                sum += value;
            }
            return sum;
        }, "累和计算");
    }

    /**
     * 沿指定轴查找最大值
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最大值数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu max(NdArrayCpu array, int axis) {
        return axisOperation(array, axis, values -> {
            float max = Float.NEGATIVE_INFINITY;
            for (float value : values) {
                if (value > max) {
                    max = value;
                }
            }
            return max;
        }, "最大值计算");
    }

    /**
     * 沿指定轴查找最小值
     *
     * @param array 数组
     * @param axis  查找轴
     * @return 最小值数组
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static NdArrayCpu min(NdArrayCpu array, int axis) {
        return axisOperation(array, axis, values -> {
            float min = Float.MAX_VALUE;
            for (float value : values) {
                if (value < min) {
                    min = value;
                }
            }
            return min;
        }, "最小值计算");
    }

    /**
     * 查找数组中的最大值（全局最大值）
     *
     * @param array 数组
     * @return 数组中的最大值
     */
    public static float max(NdArrayCpu array) {
        float max = Float.NEGATIVE_INFINITY;
        for (float value : array.buffer) {
            if (max < value) {
                max = value;
            }
        }
        return max;
    }
}

