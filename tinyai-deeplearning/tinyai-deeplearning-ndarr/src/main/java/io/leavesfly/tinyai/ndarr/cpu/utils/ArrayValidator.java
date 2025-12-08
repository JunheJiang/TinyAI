package io.leavesfly.tinyai.ndarr.cpu.utils;

import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 数组验证工具类
 * <p>提供各种数组和形状验证功能</p>
 */
public class ArrayValidator {

    /**
     * 验证数据长度与形状大小是否匹配
     *
     * @param dataLength 数据数组长度
     * @param shapeSize  形状指定的总元素数量
     * @throws IllegalArgumentException 当长度不匹配时抛出
     */
    public static void validateDataShape(int dataLength, int shapeSize) {
        if (dataLength != shapeSize) {
            throw new IllegalArgumentException(String.format("数据长度 %d 与形状大小 %d 不匹配", dataLength, shapeSize));
        }
    }

    /**
     * 验证两个数组的形状是否兼容（完全相同）
     *
     * @param shape1        第一个数组形状
     * @param shape2        第二个数组形状
     * @param operationName 操作名称，用于错误提示
     * @throws IllegalArgumentException 当形状不一致时抛出
     */
    public static void validateShapeCompatibility(Shape shape1, Shape shape2, String operationName) {
        if (!shape1.equals(shape2)) {
            throw new IllegalArgumentException(String.format("%s 操作要求形状一致：%s vs %s", operationName, shape1, shape2));
        }
    }

    /**
     * 验证轴参数的有效性
     *
     * @param axis    轴参数
     * @param dimNum  维度数量
     * @throws IllegalArgumentException 当轴参数无效时抛出
     */
    public static void validateAxis(int axis, int dimNum) {
        if (axis < 0 || axis >= dimNum) {
            throw new IllegalArgumentException(String.format("轴参数%d超出范围[0,%d)", axis, dimNum));
        }
    }

    /**
     * 验证转置维度顺序的有效性
     *
     * @param order  维度顺序数组
     * @param dimNum 维度数量
     * @throws IllegalArgumentException 当维度顺序无效时抛出
     */
    public static void validateTransposeOrder(int[] order, int dimNum) {
        if (order.length != dimNum) {
            throw new IllegalArgumentException("转置维度数量不匹配");
        }

        boolean[] used = new boolean[dimNum];
        for (int dim : order) {
            if (dim < 0 || dim >= dimNum || used[dim]) {
                throw new IllegalArgumentException("转置维度顺序无效");
            }
            used[dim] = true;
        }
    }

    /**
     * 验证数组维度一致性
     *
     * @param array 数组对象
     * @throws IllegalArgumentException 当数组维度不一致时抛出
     */
    public static void validateArrayDimensions(Object array) {
        if (array == null) {
            throw new IllegalArgumentException("数组不能为null");
        }

        if (!array.getClass().isArray()) {
            throw new IllegalArgumentException("输入必须是数组类型");
        }

        // 递归验证各维度的一致性
        validateDimensionConsistency(array, 0, new java.util.ArrayList<>());
    }

    /**
     * 递归验证多维数组各维度大小的一致性
     *
     * <p>确保多维数组在每个维度上的大小是一致的，避免不规则数组</p>
     *
     * @param array         当前数组对象
     * @param depth         当前递归深度
     * @param expectedSizes 各层级的期望大小列表
     * @throws IllegalArgumentException 当数组维度不一致或包含null元素时抛出
     */
    private static void validateDimensionConsistency(Object array, int depth, java.util.List<Integer> expectedSizes) {
        if (array == null) {
            throw new IllegalArgumentException(String.format("第%d维度包含null元素", depth));
        }

        if (!array.getClass().isArray()) {
            return; // 到达最底层元素
        }

        int length = java.lang.reflect.Array.getLength(array);

        // 检查当前维度的大小是否与期望一致
        if (expectedSizes.size() <= depth) {
            // 第一次遇到这个深度，记录期望大小
            expectedSizes.add(length);
        } else if (!expectedSizes.get(depth).equals(length)) {
            // 发现大小不一致
            throw new IllegalArgumentException(String.format("第%d维度大小不一致：期望%d，实际%d", depth, expectedSizes.get(depth), length));
        }

        // 检查维度不能为0
        if (length == 0) {
            throw new IllegalArgumentException(String.format("第%d维度大小不能为0", depth));
        }

        // 递归检查下一级维度
        for (int i = 0; i < length; i++) {
            Object subArray = java.lang.reflect.Array.get(array, i);
            validateDimensionConsistency(subArray, depth + 1, expectedSizes);
        }
    }
}

