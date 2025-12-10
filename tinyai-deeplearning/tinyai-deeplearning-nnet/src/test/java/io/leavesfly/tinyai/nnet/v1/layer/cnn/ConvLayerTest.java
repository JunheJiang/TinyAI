package io.leavesfly.tinyai.nnet.v1.layer.cnn;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v1.ParameterV1;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * ConvLayer卷积层的单元测试
 * 
 * @author TinyAI Team
 */
public class ConvLayerTest {

    private boolean originalTrainMode;

    @Before
    public void setUp() {
        originalTrainMode = Config.train;
        Config.train = true; // 启用训练模式
    }

    @After
    public void tearDown() {
        Config.train = originalTrainMode;
    }

    // =============================================================================
    // 基础功能测试
    // =============================================================================

    @Test
    public void testConvLayerInitialization() {
        // 测试卷积层初始化
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 1, 1, true);
        
        // 验证参数
        assertEquals(3, conv.getInChannels());
        assertEquals(16, conv.getOutChannels());
        assertEquals(3, conv.getKernelHeight());
        assertEquals(3, conv.getKernelWidth());
        assertEquals(1, conv.getStride());
        assertEquals(1, conv.getPadding());
        assertTrue(conv.isUseBias());
        
        // 验证权重参数存在
        assertNotNull(conv.getWeight());
        assertNotNull(conv.getBias());
        
        // 验证权重形状
        assertEquals(Shape.of(16, 3, 3, 3), conv.getWeight().getValue().getShape());
        assertEquals(Shape.of(16), conv.getBias().getValue().getShape());
    }

    @Test
    public void testConvLayerWithoutBias() {
        // 测试不使用偏置的卷积层
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 1, 1, false);
        
        assertFalse(conv.isUseBias());
        assertNull(conv.getBias());
    }

    @Test
    public void testConvLayerNonSquareKernel() {
        // 测试非正方形卷积核
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 5, 1, 1, true);
        
        assertEquals(3, conv.getKernelHeight());
        assertEquals(5, conv.getKernelWidth());
        assertEquals(Shape.of(16, 3, 3, 5), conv.getWeight().getValue().getShape());
    }

    @Test
    public void testConvLayerForward() {
        // 测试前向传播
        ConvLayer conv = new ConvLayer("conv1", 1, 1, 2, 1, 0, false);
        
        // 手动设置权重为全1，方便验证
        NdArray weight = NdArray.ones(Shape.of(1, 1, 2, 2));
        conv.getWeight().setValue(weight);
        
        // 输入: [1, 1, 3, 3]
        NdArray input = NdArray.of(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        input = input.reshape(Shape.of(1, 1, 3, 3));
        
        Variable inputVar = new Variable(input);
        Variable output = conv.layerForward(inputVar);
        
        // 验证输出形状: (3-2)/1+1 = 2
        assertEquals(Shape.of(1, 1, 2, 2), output.getValue().getShape());
        
        // 验证输出值
        // output[0,0,0,0] = 1+2+4+5 = 12
        // output[0,0,0,1] = 2+3+5+6 = 16
        // output[0,0,1,0] = 4+5+7+8 = 24
        // output[0,0,1,1] = 5+6+8+9 = 28
        assertEquals(12f, output.getValue().get(0, 0, 0, 0), 1e-6);
        assertEquals(16f, output.getValue().get(0, 0, 0, 1), 1e-6);
        assertEquals(24f, output.getValue().get(0, 0, 1, 0), 1e-6);
        assertEquals(28f, output.getValue().get(0, 0, 1, 1), 1e-6);
    }

    @Test
    public void testConvLayerWithBias() {
        // 测试带偏置的卷积层
        ConvLayer conv = new ConvLayer("conv1", 1, 2, 2, 1, 0, true);
        
        // 设置权重
        NdArray weight = NdArray.zeros(Shape.of(2, 1, 2, 2));
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                weight.set(1f, 0, 0, i, j);
                weight.set(2f, 1, 0, i, j);
            }
        }
        conv.getWeight().setValue(weight);
        
        // 设置偏置
        NdArray bias = NdArray.of(new float[]{10f, 20f});
        conv.getBias().setValue(bias);
        
        // 输入
        NdArray input = NdArray.ones(Shape.of(1, 1, 2, 2));
        Variable inputVar = new Variable(input);
        Variable output = conv.layerForward(inputVar);
        
        // 验证输出
        // 第一个通道: 1*4 + 10 = 14
        // 第二个通道: 2*4 + 20 = 28
        assertEquals(14f, output.getValue().get(0, 0, 0, 0), 1e-6);
        assertEquals(28f, output.getValue().get(0, 1, 0, 0), 1e-6);
    }

    @Test
    public void testConvLayerWithPadding() {
        // 测试带padding的卷积（same padding）
        ConvLayer conv = new ConvLayer("conv1", 1, 1, 3, 1, 1, false);
        
        NdArray weight = NdArray.ones(Shape.of(1, 1, 3, 3));
        conv.getWeight().setValue(weight);
        
        // 输入: [1, 1, 3, 3]
        NdArray input = NdArray.ones(Shape.of(1, 1, 3, 3));
        Variable inputVar = new Variable(input);
        Variable output = conv.layerForward(inputVar);
        
        // 验证输出形状: (3+2*1-3)/1+1 = 3 (保持尺寸)
        assertEquals(Shape.of(1, 1, 3, 3), output.getValue().getShape());
    }

    @Test
    public void testConvLayerWithStride() {
        // 测试带stride的卷积（下采样）
        ConvLayer conv = new ConvLayer("conv1", 1, 1, 2, 2, 0, false);
        
        NdArray weight = NdArray.ones(Shape.of(1, 1, 2, 2));
        conv.getWeight().setValue(weight);
        
        // 输入: [1, 1, 4, 4]
        NdArray input = NdArray.ones(Shape.of(1, 1, 4, 4));
        Variable inputVar = new Variable(input);
        Variable output = conv.layerForward(inputVar);
        
        // 验证输出形状: (4-2)/2+1 = 2
        assertEquals(Shape.of(1, 1, 2, 2), output.getValue().getShape());
    }

    @Test
    public void testConvLayerMultiChannel() {
        // 测试多通道卷积
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 1, 1, true);
        
        // 输入: [2, 3, 32, 32]
        NdArray input = NdArray.ones(Shape.of(2, 3, 32, 32));
        Variable inputVar = new Variable(input);
        Variable output = conv.layerForward(inputVar);
        
        // 验证输出形状
        assertEquals(Shape.of(2, 16, 32, 32), output.getValue().getShape());
    }

    // =============================================================================
    // 计算图和反向传播测试
    // =============================================================================

    @Test
    public void testConvLayerBackward() {
        // 测试反向传播（计算图完整性）
        ConvLayer conv = new ConvLayer("conv1", 1, 1, 2, 1, 0, false);
        
        NdArray input = NdArray.ones(Shape.of(1, 1, 3, 3));
        Variable inputVar = new Variable(input);
        
        Variable output = conv.layerForward(inputVar);
        
        // 反向传播
        output.backward();
        
        // 验证输入梯度存在
        assertNotNull(inputVar.getGrad());
        assertEquals(input.getShape(), inputVar.getGrad().getShape());
    }

    @Test
    public void testConvLayerChainRule() {
        // 测试链式法则
        ConvLayer conv1 = new ConvLayer("conv1", 1, 4, 3, 1, 1, true);
        ConvLayer conv2 = new ConvLayer("conv2", 4, 8, 3, 1, 1, true);
        
        NdArray input = NdArray.ones(Shape.of(1, 1, 8, 8));
        Variable x = new Variable(input);
        
        // 构建两层卷积
        Variable h1 = conv1.layerForward(x);
        Variable h2 = conv2.layerForward(h1);
        Variable sum = h2.sum();
        
        // 反向传播
        sum.backward();
        
        // 验证梯度存在
        assertNotNull(x.getGrad());
    }

    @Test
    public void testConvLayerComputationGraphIntegrity() {
        // 测试计算图完整性
        ConvLayer conv = new ConvLayer("conv1", 1, 1, 2, 1, 0, false);
        
        NdArray input = NdArray.ones(Shape.of(1, 1, 3, 3));
        Variable inputVar = new Variable(input);
        
        Variable output = conv.layerForward(inputVar);
        
        // 验证计算图连接
        assertNotNull(output.getCreator());
    }

    // =============================================================================
    // 输出形状计算测试
    // =============================================================================

    @Test
    public void testComputeOutputShape() {
        // 测试输出形状计算
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 1, 1, true);
        
        // 输入: [2, 3, 32, 32]
        NdArray input = NdArray.ones(Shape.of(2, 3, 32, 32));
        Variable inputVar = new Variable(input);
        Variable output = conv.layerForward(inputVar);
        
        // 验证输出形状: [2, 16, 32, 32]
        assertEquals(Shape.of(2, 16, 32, 32), output.getValue().getShape());
    }

    @Test
    public void testComputeOutputShapeWithStride() {
        // 测试带stride的输出形状
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 2, 1, true);
        
        // 输入: [1, 3, 32, 32]
        NdArray input = NdArray.ones(Shape.of(1, 3, 32, 32));
        Variable inputVar = new Variable(input);
        Variable output = conv.layerForward(inputVar);
        
        // 验证输出形状: (32+2*1-3)/2+1 = 16
        assertEquals(Shape.of(1, 16, 16, 16), output.getValue().getShape());
    }

    @Test
    public void testComputeOutputShapeNoPadding() {
        // 测试无padding的输出形状
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 5, 1, 0, true);
        
        // 输入: [1, 3, 28, 28]
        NdArray input = NdArray.ones(Shape.of(1, 3, 28, 28));
        Variable inputVar = new Variable(input);
        Variable output = conv.layerForward(inputVar);
        
        // 验证输出形状: (28+0-5)/1+1 = 24
        assertEquals(Shape.of(1, 16, 24, 24), output.getValue().getShape());
    }

    // =============================================================================
    // 边界条件和异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testConvLayerInvalidInputDimension() {
        // 测试错误的输入维度
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 1, 1, true);
        
        NdArray input = NdArray.ones(Shape.of(3, 32, 32)); // 3D
        Variable inputVar = new Variable(input);
        
        conv.layerForward(inputVar);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConvLayerChannelMismatch() {
        // 测试通道数不匹配
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 1, 1, true);
        
        // 输入是1通道，但期望3通道
        NdArray input = NdArray.ones(Shape.of(1, 1, 32, 32));
        Variable inputVar = new Variable(input);
        
        conv.layerForward(inputVar);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConvLayerNullInput() {
        // 测试空输入
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 1, 1, true);
        conv.layerForward((Variable[]) null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConvLayerEmptyInput() {
        // 测试空数组输入
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 1, 1, true);
        conv.layerForward(new Variable[0]);
    }

    // =============================================================================
    // 参数访问测试
    // =============================================================================

    @Test
    public void testParameterAccess() {
        // 测试参数访问
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 3, 1, 1, true);
        
        ParameterV1 weight = conv.getWeight();
        ParameterV1 bias = conv.getBias();
        
        assertNotNull(weight);
        assertNotNull(bias);
        
        assertEquals("conv1_weight", weight.getName());
        assertEquals("conv1_bias", bias.getName());
    }

    @Test
    public void testParameterShapes() {
        // 测试参数形状
        ConvLayer conv = new ConvLayer("conv1", 3, 16, 5, 7, 1, 1, true);
        
        // 权重形状: [out_channels, in_channels, kernel_h, kernel_w]
        assertEquals(Shape.of(16, 3, 5, 7), conv.getWeight().getValue().getShape());
        
        // 偏置形状: [out_channels]
        assertEquals(Shape.of(16), conv.getBias().getValue().getShape());
    }

    // =============================================================================
    // 实际应用场景测试
    // =============================================================================

    @Test
    public void testCNNSequence() {
        // 测试典型的CNN序列
        ConvLayer conv1 = new ConvLayer("conv1", 3, 32, 3, 1, 1, true);
        ConvLayer conv2 = new ConvLayer("conv2", 32, 64, 3, 2, 1, true);
        ConvLayer conv3 = new ConvLayer("conv3", 64, 128, 3, 2, 1, true);
        
        // 输入: [1, 3, 32, 32]
        NdArray input = NdArray.ones(Shape.of(1, 3, 32, 32));
        Variable x = new Variable(input);
        
        // 前向传播
        Variable h1 = conv1.layerForward(x);  // [1, 32, 32, 32]
        assertEquals(Shape.of(1, 32, 32, 32), h1.getValue().getShape());
        
        Variable h2 = conv2.layerForward(h1); // [1, 64, 16, 16]
        assertEquals(Shape.of(1, 64, 16, 16), h2.getValue().getShape());
        
        Variable h3 = conv3.layerForward(h2); // [1, 128, 8, 8]
        assertEquals(Shape.of(1, 128, 8, 8), h3.getValue().getShape());
        
        // 反向传播
        Variable sum = h3.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
    }

    @Test
    public void testResidualConnection() {
        // 测试残差连接
        ConvLayer conv1 = new ConvLayer("conv1", 16, 16, 3, 1, 1, true);
        ConvLayer conv2 = new ConvLayer("conv2", 16, 16, 3, 1, 1, true);
        
        NdArray input = NdArray.ones(Shape.of(1, 16, 32, 32));
        Variable x = new Variable(input);
        
        // 残差块: y = x + conv2(conv1(x))
        Variable h1 = conv1.layerForward(x);
        Variable h2 = conv2.layerForward(h1);
        Variable y = x.add(h2);
        
        assertEquals(Shape.of(1, 16, 32, 32), y.getValue().getShape());
        
        // 反向传播
        y.backward();
        assertNotNull(x.getGrad());
    }
}
