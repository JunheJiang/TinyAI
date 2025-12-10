package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

/**
 * Conv2d性能测试
 * <p>
 * 验证Im2Col优化版本的性能提升
 * 
 * @author TinyAI Team
 */
public class Conv2dPerformanceTest {

    @Test
    public void testConv2dPerformance() {
        System.out.println("=== Conv2d性能测试 ===\n");
        
        // 测试配置
        int batchSize = 8;
        int inChannels = 64;
        int outChannels = 128;
        int height = 32;
        int width = 32;
        int kernelSize = 3;
        int stride = 1;
        int padding = 1;
        
        // 创建输入数据
        NdArray input = NdArray.randn(Shape.of(batchSize, inChannels, height, width));
        NdArray kernel = NdArray.randn(Shape.of(outChannels, inChannels, kernelSize, kernelSize));
        
        System.out.printf("输入形状: [%d, %d, %d, %d]%n", batchSize, inChannels, height, width);
        System.out.printf("卷积核形状: [%d, %d, %d, %d]%n", outChannels, inChannels, kernelSize, kernelSize);
        System.out.printf("参数: stride=%d, padding=%d%n%n", stride, padding);
        
        // 预热
        Conv2d conv = new Conv2d(stride, padding);
        for (int i = 0; i < 3; i++) {
            conv.forward(input, kernel);
        }
        
        // 性能测试
        int iterations = 10;
        long startTime = System.nanoTime();
        
        for (int i = 0; i < iterations; i++) {
            NdArray output = conv.forward(input, kernel);
        }
        
        long endTime = System.nanoTime();
        double avgTimeMs = (endTime - startTime) / 1_000_000.0 / iterations;
        
        System.out.printf("迭代次数: %d%n", iterations);
        System.out.printf("平均耗时: %.2f ms%n", avgTimeMs);
        System.out.printf("吞吐量: %.2f images/sec%n", 1000.0 * batchSize / avgTimeMs);
        
        // 验证输出形状
        NdArray output = conv.forward(input, kernel);
        int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
        int outWidth = (width + 2 * padding - kernelSize) / stride + 1;
        
        System.out.printf("%n预期输出形状: [%d, %d, %d, %d]%n", 
            batchSize, outChannels, outHeight, outWidth);
        System.out.printf("实际输出形状: %s%n", output.getShape());
        
        assert output.getShape().equals(Shape.of(batchSize, outChannels, outHeight, outWidth));
        System.out.println("\n✅ 形状验证通过！");
    }
    
    @Test
    public void testConv2dGradient() {
        System.out.println("=== Conv2d梯度测试 ===\n");
        
        // 小规模测试梯度计算
        int batchSize = 2;
        int inChannels = 3;
        int outChannels = 4;
        int height = 8;
        int width = 8;
        int kernelSize = 3;
        
        NdArray input = NdArray.randn(Shape.of(batchSize, inChannels, height, width));
        NdArray kernel = NdArray.randn(Shape.of(outChannels, inChannels, kernelSize, kernelSize));
        
        Conv2d conv = new Conv2d(1, 1);
        
        // 前向传播
        long startForward = System.nanoTime();
        NdArray output = conv.forward(input, kernel);
        long forwardTime = System.nanoTime() - startForward;
        
        // 反向传播
        NdArray yGrad = NdArray.ones(output.getShape());
        long startBackward = System.nanoTime();
        var grads = conv.backward(yGrad);
        long backwardTime = System.nanoTime() - startBackward;
        
        System.out.printf("前向传播耗时: %.3f ms%n", forwardTime / 1_000_000.0);
        System.out.printf("反向传播耗时: %.3f ms%n", backwardTime / 1_000_000.0);
        
        // 验证梯度形状
        NdArray inputGrad = grads.get(0);
        NdArray kernelGrad = grads.get(1);
        
        assert inputGrad.getShape().equals(input.getShape());
        assert kernelGrad.getShape().equals(kernel.getShape());
        
        System.out.println("✅ 梯度形状验证通过！");
    }
    
    @Test
    public void testDifferentConfigurations() {
        System.out.println("=== 不同配置性能对比 ===\n");
        
        // 测试不同stride和padding组合
        int[][] configs = {
            {1, 0},  // stride=1, padding=0
            {1, 1},  // stride=1, padding=1 (same padding)
            {2, 0},  // stride=2, padding=0
            {2, 1}   // stride=2, padding=1
        };
        
        int batchSize = 4;
        int inChannels = 32;
        int outChannels = 64;
        int height = 28;
        int width = 28;
        int kernelSize = 3;
        
        NdArray input = NdArray.randn(Shape.of(batchSize, inChannels, height, width));
        NdArray kernel = NdArray.randn(Shape.of(outChannels, inChannels, kernelSize, kernelSize));
        
        for (int[] config : configs) {
            int stride = config[0];
            int padding = config[1];
            
            Conv2d conv = new Conv2d(stride, padding);
            
            // 预热
            conv.forward(input, kernel);
            
            // 测试
            long startTime = System.nanoTime();
            int iterations = 20;
            for (int i = 0; i < iterations; i++) {
                conv.forward(input, kernel);
            }
            long avgTime = (System.nanoTime() - startTime) / iterations;
            
            int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
            int outWidth = (width + 2 * padding - kernelSize) / stride + 1;
            
            System.out.printf("stride=%d, padding=%d -> 输出[%d,%d,%d,%d], 耗时: %.2f ms%n",
                stride, padding, batchSize, outChannels, outHeight, outWidth,
                avgTime / 1_000_000.0);
        }
    }
}
