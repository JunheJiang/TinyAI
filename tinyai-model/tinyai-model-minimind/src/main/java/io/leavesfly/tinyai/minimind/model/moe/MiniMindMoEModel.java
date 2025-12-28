package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.attention.KVCache;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * MiniMind MoE 语言模型
 * <p>
 * 集成 Mixture of Experts 的轻量级 GPT 风格语言模型
 * 
 * 特点:
 * - 支持自回归文本生成
 * - 增量推理（KV-Cache）
 * - MoE 架构提升参数效率
 * - 负载均衡损失
 * - 专家使用统计
 * 
 * 
 * 
//  * MiniMindMoEModel (145M 参数)
// ├── MiniMindMoEBlock
// │   ├── TokenEmbedding (6400 × 512)
// │   ├── 8 × MiniMindMoETransformerLayer
// │   │   ├── LayerNorm
// │   │   ├── MultiHeadAttention (16 heads)
// │   │   ├── LayerNorm
// │   │   └── MoELayer (4 experts, top-2)
// │   │       ├── ExpertRouter
// │   │       └── 4 × ExpertNetwork
// │   ├── Final LayerNorm
// │   └── LM Head (512 × 6400)
//  * 
 * 
 * 
 * 模型规模:
 * - MoE: 145M 参数（4专家,每次激活2个）
 * 
 * 继承自 TinyAI Model 类
 *
 * @author leavesfly
 * @version 1.0
 */
public class MiniMindMoEModel extends Model {

    /**
     * 模型配置
     */
    private final MiniMindConfig config;

    /**
     * MoE 模型主体
     */
    private final MiniMindMoEBlock moeBlock;

    /**
     * 随机数生成器
     */
    private final Random random;

    /**
     * 构造 MiniMind MoE 模型
     *
     * @param name   模型名称
     * @param config 模型配置（必须启用 MoE）
     */
    public MiniMindMoEModel(String name, MiniMindConfig config) {
        super(name, new MiniMindMoEBlock(config));
        
        if (!config.isUseMoE()) {
            throw new IllegalArgumentException("Config must have MoE enabled (useMoE=true)");
        }
        
        this.config = config;
        this.moeBlock = (MiniMindMoEBlock) getModule();
        this.random = new Random(42);

        // 设置模型描述
        setDescription(String.format(
            "MiniMind MoE Language Model - %d experts, %d active, ~%dM parameters",
            config.getNumExperts(),
            config.getNumExpertsPerToken(),
            estimateParameters() / 1_000_000
        ));
    }

    /**
     * 创建 MoE 模型 (145M 参数)
     *
     * @param name 模型名称
     * @return MiniMind MoE 模型
     */
    public static MiniMindMoEModel create(String name) {
        return new MiniMindMoEModel(name, MiniMindConfig.createMoEConfig());
    }

    /**
     * 单次预测（不使用 KV-Cache） - 接受Variable
     *
     * @param tokenIds Token IDs Variable,形状 [batch_size, seq_len]
     * @return Logits,形状 [batch_size, seq_len, vocab_size]
     */
    public Variable predict(Variable tokenIds) {
        return moeBlock.forward(tokenIds);
    }

    /**
     * 单次预测（不使用 KV-Cache） - 接受NdArray
     *
     * @param tokenIds Token IDs,形状 [batch_size, seq_len]
     * @return Logits,形状 [batch_size, seq_len, vocab_size]
     */
    public Variable predict(NdArray tokenIds) {
        Variable input = new Variable(tokenIds);
        return moeBlock.forward(input);
    }

    /**
     * 带负载均衡损失的预测
     *
     * @param tokenIds Token IDs,形状 [batch_size, seq_len]
     * @return MoE 输出结果
     */
    public MiniMindMoEBlock.MoEOutput predictWithLoss(NdArray tokenIds) {
        Variable input = new Variable(tokenIds);
        return moeBlock.forwardWithCache(input, null, 0);
    }

    /**
     * 自回归文本生成（贪婪采样）
     *
     * @param promptTokens 提示词 Token IDs
     * @param maxNewTokens 最大生成 Token 数量
     * @return 生成的完整 Token 序列
     */
    public int[] generate(int[] promptTokens, int maxNewTokens) {
        return generate(promptTokens, maxNewTokens, 0.0f, 0, 0.0f);
    }

    /**
     * 自回归文本生成（支持多种采样策略）
     *
     * @param promptTokens 提示词 Token IDs
     * @param maxNewTokens 最大生成 Token 数量
     * @param temperature  温度参数（0.0 = 贪婪,越大越随机）
     * @param topK         Top-K 采样参数（0 = 禁用）
     * @param topP         Top-P 采样参数（0.0 = 禁用）
     * @return 生成的完整 Token 序列
     */
    public int[] generate(int[] promptTokens, int maxNewTokens, 
                         float temperature, int topK, float topP) {
        
        moeBlock.setTraining(false);  // 推理模式
        
        List<Integer> generatedTokens = new ArrayList<>();
        for (int token : promptTokens) {
            generatedTokens.add(token);
        }

        // 创建 KV-Cache
        List<KVCache> kvCaches = moeBlock.createKVCaches(1);

        // 首次前向传播（处理完整提示词）
        NdArray promptArray = NdArray.of(Shape.of(1, promptTokens.length));
        float[] promptBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) promptArray).buffer;
        for (int i = 0; i < promptTokens.length; i++) {
            promptBuffer[i] = promptTokens[i];
        }
        Variable promptVar = new Variable(promptArray);
        MiniMindMoEBlock.MoEOutput output = moeBlock.forwardWithCache(promptVar, kvCaches, 0);
        Variable logits = output.getOutput();

        // 从最后一个位置的 logits 采样
        int nextToken = sampleToken(logits, promptTokens.length - 1, temperature, topK, topP);
        generatedTokens.add(nextToken);

        // 自回归生成
        for (int step = 1; step < maxNewTokens; step++) {
            // 准备输入（只有一个新 token）
            NdArray tokenArray = NdArray.of(Shape.of(1, 1));
            float[] tokenBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) tokenArray).buffer;
            tokenBuffer[0] = nextToken;
            Variable tokenVar = new Variable(tokenArray);

            // 增量前向传播
            int currentPos = promptTokens.length + step - 1;
            output = moeBlock.forwardGeneration(tokenVar, kvCaches, currentPos);
            logits = output.getOutput();

            // 采样下一个 token
            nextToken = sampleToken(logits, 0, temperature, topK, topP);
            generatedTokens.add(nextToken);

            // 停止条件（可根据需要添加 EOS token 检查）
            if (nextToken == config.getVocabSize() - 1) {  // 假设最后一个是 EOS
                break;
            }
        }

        // 转换为数组
        int[] result = new int[generatedTokens.size()];
        for (int i = 0; i < generatedTokens.size(); i++) {
            result[i] = generatedTokens.get(i);
        }

        return result;
    }

    /**
     * 采样下一个 Token
     */
    private int sampleToken(Variable logits, int position, 
                           float temperature, int topK, float topP) {
        // 提取指定位置的 logits: [vocab_size]
        NdArray logitsData = logits.getValue();
        int[] shape = logitsData.getShape().getShapeDims();
        int vocabSize = shape[shape.length - 1];
        
        float[] logitsArray = new float[vocabSize];
        float[] logitsBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) logitsData).buffer;
        
        int offset = position * vocabSize;
        System.arraycopy(logitsBuffer, offset, logitsArray, 0, vocabSize);

        // 贪婪采样
        if (temperature == 0.0f || (topK == 0 && topP == 0.0f)) {
            return argmax(logitsArray);
        }

        // 温度缩放
        if (temperature != 1.0f) {
            for (int i = 0; i < vocabSize; i++) {
                logitsArray[i] /= temperature;
            }
        }

        // Softmax
        float[] probs = softmax(logitsArray);

        // Top-K 过滤
        if (topK > 0) {
            probs = applyTopK(probs, topK);
        }

        // Top-P 过滤
        if (topP > 0.0f && topP < 1.0f) {
            probs = applyTopP(probs, topP);
        }

        // 重新归一化
        float sum = 0.0f;
        for (float p : probs) {
            sum += p;
        }
        if (sum > 0) {
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }
        }

        // 多项式采样
        return multinomialSample(probs);
    }

    /**
     * Softmax
     */
    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float val : logits) {
            max = Math.max(max, val);
        }

        float[] probs = new float[logits.length];
        float sum = 0.0f;
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }

        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }

        return probs;
    }

    /**
     * Top-K 过滤
     */
    private float[] applyTopK(float[] probs, int topK) {
        int[] indices = argsort(probs);
        float[] filtered = new float[probs.length];
        
        for (int i = 0; i < topK && i < probs.length; i++) {
            int idx = indices[probs.length - 1 - i];
            filtered[idx] = probs[idx];
        }
        
        return filtered;
    }

    /**
     * Top-P 过滤
     */
    private float[] applyTopP(float[] probs, float topP) {
        int[] indices = argsort(probs);
        float[] filtered = new float[probs.length];
        
        float cumProb = 0.0f;
        for (int i = probs.length - 1; i >= 0; i--) {
            int idx = indices[i];
            cumProb += probs[idx];
            filtered[idx] = probs[idx];
            if (cumProb >= topP) {
                break;
            }
        }
        
        return filtered;
    }

    /**
     * 多项式采样
     */
    private int multinomialSample(float[] probs) {
        float rand = random.nextFloat();
        float cumProb = 0.0f;
        
        for (int i = 0; i < probs.length; i++) {
            cumProb += probs[i];
            if (rand < cumProb) {
                return i;
            }
        }
        
        return probs.length - 1;
    }

    /**
     * Argmax
     */
    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Argsort (返回排序后的索引)
     */
    private int[] argsort(float[] array) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(array[a], array[b]));
        
        int[] result = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = indices[i];
        }
        return result;
    }

    /**
     * 获取配置
     */
    public MiniMindConfig getConfig() {
        return config;
    }

    /**
     * 获取 MoE Block
     */
    public MiniMindMoEBlock getMoEBlock() {
        return moeBlock;
    }

    /**
     * 获取专家使用统计
     */
    public String getExpertUsageStats() {
        return moeBlock.getExpertUsageStats();
    }

    /**
     * 重置统计信息
     */
    public void resetStats() {
        moeBlock.resetStats();
    }

    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println("=" .repeat(60));
        System.out.println(moeBlock.getModelInfo());
        System.out.println("=" .repeat(60));
    }

    /**
     * 估算参数量
     */
    private long estimateParameters() {
        return config.estimateParameters();
    }
}
