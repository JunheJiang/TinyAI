package io.leavesfly.tinyai.rl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.DQNAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;
import io.leavesfly.tinyai.rl.Environment;

/**
 * DQN算法完整演示 - CartPole环境
 * 
 * <p>本演示展示如何使用DQN(Deep Q-Network)算法解决CartPole倒立摆问题:
 * <ul>
 *   <li>DQN核心组件: Q网络、目标网络、经验回放</li>
 *   <li>训练过程: 探索率衰减、批量学习、网络更新</li>
 *   <li>性能评估: 平均奖励、成功率、学习曲线</li>
 * </ul>
 * 
 * <p><b>运行方式:</b>
 * <pre>
 * mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.DQNCartPoleDemo" \
 *   -pl tinyai-deeplearning-rl
 * </pre>
 * 
 * @author TinyAI Team
 */
public class DQNCartPoleDemo {

    // 训练超参数
    private static final int MAX_EPISODES = 300;
    private static final int EVAL_INTERVAL = 50;
    private static final int EVAL_EPISODES = 10;
    
    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("        DQN算法完整演示 - CartPole        ");
        System.out.println("==========================================\n");

        demonstrateProblem();
        DQNAgent agent = setupDQNAgent();
        CartPoleEnvironment env = new CartPoleEnvironment(500);
        
        trainAgent(agent, env);
        evaluateAgent(agent, env);
        
        System.out.println("\n==========================================");
        System.out.println("            DQN训练完成!                  ");
        System.out.println("==========================================");
    }

    /**
     * 展示问题背景
     */
    private static void demonstrateProblem() {
        System.out.println("【问题背景: CartPole倒立摆】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("目标: 通过左右移动小车,保持杆子竖直平衡");
        System.out.println();
        System.out.println("           |");
        System.out.println("           |  ← 杆子");
        System.out.println("           |");
        System.out.println("      ┌────┴────┐");
        System.out.println("      │   小车   │ ← 可左右移动");
        System.out.println("      └─────────┘");
        System.out.println("  ═══════════════════");
        System.out.println();
        
        System.out.println("状态空间(4维连续):");
        System.out.println("  • 小车位置: [-2.4, 2.4]");
        System.out.println("  • 小车速度: [-∞, +∞]");
        System.out.println("  • 杆的角度: [-0.21, 0.21] 弧度");
        System.out.println("  • 杆的角速度: [-∞, +∞]");
        
        System.out.println("\n动作空间(2维离散):");
        System.out.println("  • 动作0: 向左推 ←");
        System.out.println("  • 动作1: 向右推 →");
        
        System.out.println("\n奖励设计:");
        System.out.println("  • 每步保持平衡: +1");
        System.out.println("  • 杆倒下或超界: 回合结束");
        System.out.println();
    }

    /**
     * 创建和配置DQN智能体
     */
    private static DQNAgent setupDQNAgent() {
        System.out.println("【DQN智能体配置】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        int stateDim = 4;
        int actionDim = 2;
        int[] hiddenSizes = {128, 128};
        float learningRate = 0.001f;
        float epsilon = 1.0f;  // 初始探索率
        float gamma = 0.99f;   // 折扣因子
        int batchSize = 64;
        int bufferSize = 10000;
        int targetUpdateFreq = 100;
        
        System.out.println("网络结构:");
        System.out.println("  输入层: " + stateDim + " (状态维度)");
        System.out.println("  隐藏层: " + hiddenSizes[0] + " → " + hiddenSizes[1]);
        System.out.println("  输出层: " + actionDim + " (Q值)");
        
        System.out.println("\n超参数:");
        System.out.println("  学习率: " + learningRate);
        System.out.println("  初始探索率: " + epsilon);
        System.out.println("  折扣因子: " + gamma);
        System.out.println("  批次大小: " + batchSize);
        System.out.println("  经验缓冲区: " + bufferSize);
        System.out.println("  目标网络更新频率: " + targetUpdateFreq + "步");
        
        DQNAgent agent = new DQNAgent(
            "CartPole-DQN",
            stateDim,
            actionDim,
            hiddenSizes,
            learningRate,
            epsilon,
            gamma,
            batchSize,
            bufferSize,
            targetUpdateFreq
        );
        
        System.out.println("\n✓ DQN智能体创建完成\n");
        return agent;
    }

    /**
     * 训练DQN智能体
     */
    private static void trainAgent(DQNAgent agent, CartPoleEnvironment env) {
        System.out.println("【开始训练】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("总回合数: " + MAX_EPISODES);
        System.out.println("评估间隔: 每" + EVAL_INTERVAL + "回合\n");
        
        System.out.println("回合 |  奖励  |   ε   | 缓冲区 | Loss");
        System.out.println("-----|--------|-------|--------|-------");
        
        float[] recentRewards = new float[10];
        int rewardIndex = 0;
        
        for (int episode = 0; episode < MAX_EPISODES; episode++) {
            Variable state = env.reset();
            float episodeReward = 0;
            int steps = 0;
            
            // 回合循环
            while (!env.isDone() && steps < 500) {
                // 选择动作
                Variable action = agent.selectAction(state);
                
                // 执行动作
                Environment.StepResult result = env.step(action);
                
                // 存储经验并学习
                Experience experience = new Experience(
                    state, action, result.getReward(),
                    result.getNextState(), result.isDone(), steps
                );
                agent.learn(experience);
                
                episodeReward += result.getReward();
                state = result.getNextState();
                steps++;
            }
            
            // 探索率衰减
            agent.decayEpsilon(0.995f);
            
            // 记录最近奖励
            recentRewards[rewardIndex] = episodeReward;
            rewardIndex = (rewardIndex + 1) % recentRewards.length;
            
            // 定期打印训练信息
            if ((episode + 1) % 10 == 0) {
                System.out.printf(" %3d | %6.1f | %.3f | %5.1f%% | %.4f\n",
                    episode + 1,
                    episodeReward,
                    agent.getCurrentEpsilon(),
                    agent.getBufferUsage() * 100,
                    agent.getAverageLoss()
                );
            }
            
            // 定期评估
            if ((episode + 1) % EVAL_INTERVAL == 0) {
                float avgReward = evaluatePerformance(agent, env, EVAL_EPISODES);
                System.out.println("─────|────────|───────|────────|───────");
                System.out.printf("评估 | %6.1f | (当前性能评估)\n", avgReward);
                System.out.println("─────|────────|───────|────────|───────");
                
                // 判断是否已经学会
                if (avgReward >= 195) {
                    System.out.println("\n✓ 智能体已学会控制倒立摆!(平均奖励≥195)");
                    System.out.println("  在第 " + (episode + 1) + " 回合达成目标");
                    break;
                }
            }
        }
        
        System.out.println();
    }

    /**
     * 评估性能
     */
    private static float evaluatePerformance(DQNAgent agent, CartPoleEnvironment env, int episodes) {
        agent.setTraining(false);
        float totalReward = 0;
        
        for (int episode = 0; episode < episodes; episode++) {
            Variable state = env.reset();
            float episodeReward = 0;
            int steps = 0;
            
            while (!env.isDone() && steps < 500) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = env.step(action);
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
            }
            
            totalReward += episodeReward;
        }
        
        agent.setTraining(true);
        return totalReward / episodes;
    }

    /**
     * 最终评估
     */
    private static void evaluateAgent(DQNAgent agent, CartPoleEnvironment env) {
        System.out.println("【最终评估】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("测试" + EVAL_EPISODES + "个回合,使用贪心策略(不探索)\n");
        
        agent.setTraining(false);
        float totalReward = 0;
        int successCount = 0;
        
        System.out.println("回合 | 步数 | 奖励 | 结果");
        System.out.println("-----|------|------|--------");
        
        for (int episode = 0; episode < EVAL_EPISODES; episode++) {
            Variable state = env.reset();
            float episodeReward = 0;
            int steps = 0;
            
            while (!env.isDone() && steps < 500) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = env.step(action);
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
            }
            
            totalReward += episodeReward;
            if (episodeReward >= 195) {
                successCount++;
            }
            
            System.out.printf(" %2d  | %3d  | %4.0f | %s\n",
                episode + 1,
                steps,
                episodeReward,
                episodeReward >= 195 ? "成功✓" : "失败✗"
            );
        }
        
        float avgReward = totalReward / EVAL_EPISODES;
        float successRate = (float) successCount / EVAL_EPISODES * 100;
        
        System.out.println("\n评估结果:");
        System.out.println("  平均奖励: " + String.format("%.2f", avgReward));
        System.out.println("  成功率: " + String.format("%.1f%% (%d/%d)", 
            successRate, successCount, EVAL_EPISODES));
        System.out.println("  最大可能奖励: 500");
        
        if (avgReward >= 195) {
            System.out.println("\n🎉 恭喜!DQN成功学会控制倒立摆!");
        } else if (avgReward >= 100) {
            System.out.println("\n👍 表现不错,但还有提升空间");
        } else {
            System.out.println("\n💪 继续训练可能会更好");
        }
        
        System.out.println("\n【DQN关键要点】");
        System.out.println("✓ 经验回放: 打破数据相关性,提高样本效率");
        System.out.println("✓ 目标网络: 稳定训练过程,防止震荡");
        System.out.println("✓ ε-贪心: 平衡探索与利用,逐步收敛");
        System.out.println("✓ 神经网络: 近似Q函数,处理高维状态");
    }
}
