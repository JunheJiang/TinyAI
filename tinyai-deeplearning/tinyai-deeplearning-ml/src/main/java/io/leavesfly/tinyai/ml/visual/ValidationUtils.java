package io.leavesfly.tinyai.ml.visual;

import io.leavesfly.tinyai.ml.exception.ParameterMismatchException;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 输入验证工具类
 * 
 * @author TinyAI
 * @version 1.0
 */
public class ValidationUtils {
    
    /**
     * 验证对象非空
     * 
     * @param obj 要验证的对象
     * @param name 对象名称（用于错误消息）
     * @throws IllegalArgumentException 如果对象为null
     */
    public static void requireNonNull(Object obj, String name) {
        if (obj == null) {
            throw new IllegalArgumentException(name + " cannot be null");
        }
    }
    
    /**
     * 验证数值为正数
     * 
     * @param value 要验证的数值
     * @param name 数值名称（用于错误消息）
     * @throws IllegalArgumentException 如果数值不是正数
     */
    public static void requirePositive(int value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive, got: " + value);
        }
    }
    
    /**
     * 验证数值为正数（浮点数）
     * 
     * @param value 要验证的数值
     * @param name 数值名称（用于错误消息）
     * @throws IllegalArgumentException 如果数值不是正数
     */
    public static void requirePositive(float value, String name) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive, got: " + value);
        }
    }
    
    /**
     * 验证数值为非负数
     * 
     * @param value 要验证的数值
     * @param name 数值名称（用于错误消息）
     * @throws IllegalArgumentException 如果数值为负数
     */
    public static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, got: " + value);
        }
    }
    
    /**
     * 验证数值为非负数（浮点数）
     * 
     * @param value 要验证的数值
     * @param name 数值名称（用于错误消息）
     * @throws IllegalArgumentException 如果数值为负数
     */
    public static void requireNonNegative(float value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, got: " + value);
        }
    }
    
    /**
     * 验证数值在范围内
     * 
     * @param value 要验证的数值
     * @param min 最小值（包含）
     * @param max 最大值（包含）
     * @param name 数值名称（用于错误消息）
     * @throws IllegalArgumentException 如果数值不在范围内
     */
    public static void requireInRange(int value, int min, int max, String name) {
        if (value < min || value > max) {
            throw new IllegalArgumentException(
                String.format("%s must be in range [%d, %d], got: %d", name, min, max, value));
        }
    }
    
    /**
     * 验证形状匹配
     * 
     * @param expected 期望的形状
     * @param actual 实际的形状
     * @param context 上下文信息（用于错误消息）
     * @throws ParameterMismatchException 如果形状不匹配
     */
    public static void requireShapeMatch(Shape expected, Shape actual, String context) {
        if (expected == null || actual == null) {
            if (expected != actual) {
                throw new ParameterMismatchException(
                    context, expected, actual);
            }
            return;
        }
        
        if (!expected.equals(actual)) {
            throw new ParameterMismatchException(
                context, expected, actual);
        }
    }
    
    /**
     * 验证比例总和为1.0（允许小的浮点误差）
     * 
     * @param ratios 比例数组
     * @param names 比例名称数组
     * @throws IllegalArgumentException 如果比例总和不为1.0
     */
    public static void requireRatiosSumToOne(float[] ratios, String[] names) {
        float sum = 0f;
        for (float ratio : ratios) {
            sum += ratio;
        }
        
        if (Math.abs(sum - 1.0f) > 1e-5) {
            StringBuilder sb = new StringBuilder("Ratios must sum to 1.0, got: ");
            for (int i = 0; i < ratios.length; i++) {
                if (i > 0) sb.append(" + ");
                sb.append(names[i]).append("=").append(ratios[i]);
            }
            sb.append(" = ").append(sum);
            throw new IllegalArgumentException(sb.toString());
        }
    }
}

