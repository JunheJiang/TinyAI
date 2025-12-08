package io.leavesfly.tinyai.ndarr;

import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArrayCpu 关键行为补充测试（TDD 先行）
 */
public class NdArrayCpuBehaviorTest {

    @Test
    public void reshapeShouldShareUnderlyingBuffer() {
        NdArrayCpu original = (NdArrayCpu) NdArray.of(new float[]{1f, 2f, 3f, 4f});
        NdArrayCpu reshaped = (NdArrayCpu) original.reshape(Shape.of(2, 2));

        // 期望共享底层 buffer，避免不必要拷贝
        assertSame(original.getArray(), reshaped.getArray());

        // 修改 reshaped 的数据，原始视图同步变化
        reshaped.getArray()[3] = 99f;
        assertEquals(99f, original.getArray()[3], 1e-6f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void broadcastToShouldRejectIncompatibleShape() {
        NdArrayCpu src = (NdArrayCpu) NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        // 目标维度既不等于源，也不为 1，应该抛出异常
        src.broadcastTo(Shape.of(3, 3));
    }

    @Test(expected = IllegalArgumentException.class)
    public void sumToShouldRejectIncompatibleShape() {
        NdArrayCpu src = (NdArrayCpu) NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        // 目标形状与 sumTo 规则不兼容，应抛异常
        src.sumTo(Shape.of(2, 2));
    }

    @Test
    public void dotSupportsBatchBroadcastOnLeadingDims() {
        // a 形状 [2,1,3]
        float[][][] aData = {
                {{1f, 2f, 3f}},
                {{4f, 5f, 6f}}
        };
        // b 形状 [1,3,2]，在批次维度可广播
        float[][][] bData = {
                {
                        {1f, 0f},
                        {0f, 1f},
                        {1f, 1f}
                }
        };

        NdArrayCpu a = (NdArrayCpu) NdArray.of(aData);
        NdArrayCpu b = (NdArrayCpu) NdArray.of(bData);

        NdArrayCpu result = a.dot(b);
        assertEquals(Shape.of(2, 1, 2), result.getShape());

        // batch 0: [1,2,3] x [[1,0],[0,1],[1,1]] => [4,5]
        assertEquals(4f, result.get(0, 0, 0), 1e-6f);
        assertEquals(5f, result.get(0, 0, 1), 1e-6f);

        // batch 1: [4,5,6] x same matrix => [10,11]
        assertEquals(10f, result.get(1, 0, 0), 1e-6f);
        assertEquals(11f, result.get(1, 0, 1), 1e-6f);
    }

    @Test(expected = ArithmeticException.class)
    public void divShouldRejectDivideByZero() {
        NdArrayCpu a = (NdArrayCpu) NdArray.of(new float[][]{{1f, 2f}});
        NdArrayCpu b = (NdArrayCpu) NdArray.of(new float[][]{{0f, 1f}});
        a.div(b);
    }
}

