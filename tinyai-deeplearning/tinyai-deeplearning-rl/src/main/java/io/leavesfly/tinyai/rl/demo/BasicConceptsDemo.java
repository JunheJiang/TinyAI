package io.leavesfly.tinyai.rl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.ReplayBuffer;
import io.leavesfly.tinyai.rl.agent.DQNAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;
import io.leavesfly.tinyai.rl.Environment;

/**
 * 强化学习核心概念演示
 * 
 * <p>本演示详细讲解强化学习的5个核心概念:
 * <ol>
 *   <li><b>状态(State)</b> - 环境的当前情况</li>
 *   <li><b>动作(Action)</b> - 智能体的决策选择</li>
 *   <li><b>奖励(Reward)</b> - 动作的即时反馈</li>
 *   <li><b>策略(Policy)</b> - 从状态到动作的映射</li>
 *   <li><b>价值函数(Value Function)</b> - 长期回报的估计</li>
 * </ol>
 * 
 * <p><b>运行方式:</b>
 * <pre>
 * mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.BasicConceptsDemo" \
 *   -pl tinyai-deeplearning-rl
 * </pre>
 * 
 * @author TinyAI Team
 */
public class BasicConceptsDemo {

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("       强化学习核心概念详解演示           ");
        System.out.println("==========================================\n");

        demonstrateState();
        demonstrateAction();
        demonstrateReward();
        demonstratePolicy();
        demonstrateValueFunction();
        demonstrateCompleteFlow();

        System.out.println("\n==========================================");
        System.out.println("         核心概念演示完成!                ");
        System.out.println("==========================================");
    }

    /**
     * 演示概念1: 状态(State)
     */
    private static void demonstrateState() {
        System.out.println("\n【概念1: 状态 (State)】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("状态是环境在某一时刻的完整描述。");
        System.out.println();

        // 创建CartPole环境
        CartPoleEnvironment env = new CartPoleEnvironment(500);
        Variable initialState = env.reset();

        System.out.println("示例: CartPole(倒立摆)环境的状态");
        System.out.println("状态维度: 4");
        float[] stateValues = initialState.getValue().getArray();
        System.out.println("  [0] 小车位置:   " + String.format("%.4f", stateValues[0]));
        System.out.println("  [1] 小车速度:   " + String.format("%.4f", stateValues[1]));
        System.out.println("  [2] 杆的角度:   " + String.format("%.4f", stateValues[2]));
        System.out.println("  [3] 杆的角速度: " + String.format("%.4f", stateValues[3]));
        
        System.out.println("\n💡 理解要点:");
        System.out.println("  • 状态包含了做出决策所需的所有信息");
        System.out.println("  • 不同环境有不同的状态表示方式");
        System.out.println("  • 状态可以是离散的(网格世界)或连续的(CartPole)");
    }

    /**
     * 演示概念2: 动作(Action)
     */
    private static void demonstrateAction() {
        System.out.println("\n【概念2: 动作 (Action)】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("动作是智能体可以执行的所有可能选择。");
        System.out.println();

        CartPoleEnvironment env = new CartPoleEnvironment(500);
        
        System.out.println("示例: CartPole环境的动作空间");
        System.out.println("动作类型: 离散动作");
        System.out.println("可选动作数: " + env.getActionDim());
        System.out.println("  动作0: 向左推小车 ←");
        System.out.println("  动作1: 向右推小车 →");
        
        System.out.println("\n动作选择演示:");
        Variable state = env.reset();
        for (int i = 0; i < 2; i++) {
            Variable action = new Variable(NdArray.of(i));
            System.out.println("  执行动作" + i + ": " + (i == 0 ? "←左推" : "→右推"));
        }

        System.out.println("\n💡 理解要点:");
        System.out.println("  • 动作空间定义了智能体能做什么");
        System.out.println("  • 离散动作: 有限个选择(如上下左右)");
        System.out.println("  • 连续动作: 无限个选择(如转向角度)");
    }

    /**
     * 演示概念3: 奖励(Reward)
     */
    private static void demonstrateReward() {
        System.out.println("\n【概念3: 奖励 (Reward)】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("奖励是环境对智能体动作的即时反馈信号。");
        System.out.println();

        CartPoleEnvironment env = new CartPoleEnvironment(500);
        Variable state = env.reset();

        System.out.println("示例: CartPole环境的奖励机制");
        System.out.println("奖励设计:");
        System.out.println("  • 杆保持平衡: +1");
        System.out.println("  • 杆倒下或超界: 0 (结束)");
        
        System.out.println("\n实际交互演示:");
        int step = 0;
        float totalReward = 0;
        
        while (!env.isDone() && step < 5) {
            Variable action = env.sampleAction();
            Environment.StepResult result = env.step(action);
            
            totalReward += result.getReward();
            
            System.out.printf("  步骤%d: 动作=%d, 奖励=%.1f, 累积奖励=%.1f\n",
                step + 1,
                (int) action.getValue().getNumber().floatValue(),
                result.getReward(),
                totalReward
            );
            
            step++;
        }

        System.out.println("\n💡 理解要点:");
        System.out.println("  • 奖励是学习的驱动力,告诉智能体什么是好的");
        System.out.println("  • 好的奖励设计至关重要");
        System.out.println("  • 智能体的目标是最大化累积奖励");
    }

    /**
     * 演示概念4: 策略(Policy)
     */
    private static void demonstratePolicy() {
        System.out.println("\n【概念4: 策略 (Policy)】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("策略是从状态到动作的映射函数 π(a|s)。");
        System.out.println();

        System.out.println("常见策略类型:");
        System.out.println("1. 确定性策略: 每个状态对应唯一动作");
        System.out.println("   π(s) = argmax Q(s, a)");
        
        System.out.println("\n2. 随机策略: 每个状态对应动作的概率分布");
        System.out.println("   π(a|s) = P(a|s)");
        
        System.out.println("\n3. ε-贪心策略: 平衡探索与利用");
        System.out.println("   • 以概率ε随机选择(探索)");
        System.out.println("   • 以概率(1-ε)选择最优(利用)");

        System.out.println("\n策略演示(ε=0.1):");
        float epsilon = 0.1f;
        for (int i = 0; i < 10; i++) {
            boolean explore = Math.random() < epsilon;
            System.out.printf("  选择%d: %s\n", i + 1, explore ? "探索(随机)" : "利用(最优)");
        }

        System.out.println("\n💡 理解要点:");
        System.out.println("  • 策略是智能体的决策规则");
        System.out.println("  • 探索(Exploration): 尝试新动作获取信息");
        System.out.println("  • 利用(Exploitation): 选择已知最优动作");
    }

    /**
     * 演示概念5: 价值函数(Value Function)
     */
    private static void demonstrateValueFunction() {
        System.out.println("\n【概念5: 价值函数 (Value Function)】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("价值函数估计从某个状态出发能获得的长期回报。");
        System.out.println();

        System.out.println("两种价值函数:");
        System.out.println("1. 状态价值函数 V(s)");
        System.out.println("   = 从状态s开始,遵循策略π的期望回报");
        
        System.out.println("\n2. 动作价值函数 Q(s,a)");
        System.out.println("   = 在状态s执行动作a后的期望回报");

        System.out.println("\nBellman方程:");
        System.out.println("  Q(s,a) = r + γ·max Q(s',a')");
        System.out.println("  其中:");
        System.out.println("    r   = 即时奖励");
        System.out.println("    γ   = 折扣因子(0-1)");
        System.out.println("    s'  = 下一状态");
        System.out.println("    a'  = 下一动作");

        System.out.println("\n示例: 简化的Q值表");
        System.out.println("  状态\\动作  |  左(0)  |  右(1)");
        System.out.println("  ----------|---------|--------");
        System.out.println("  状态1     |  2.5    |  3.8  ← 最优");
        System.out.println("  状态2     |  1.2  ← |  0.9");
        System.out.println("  状态3     |  4.1  ← |  2.3");

        System.out.println("\n💡 理解要点:");
        System.out.println("  • 价值函数评估状态/动作的好坏");
        System.out.println("  • DQN用神经网络近似Q函数");
        System.out.println("  • 学习过程就是不断更新价值估计");
    }

    /**
     * 演示完整的RL交互流程
     */
    private static void demonstrateCompleteFlow() {
        System.out.println("\n【完整RL交互流程】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("将所有概念整合,展示一个完整的学习循环\n");

        System.out.println("流程图:");
        System.out.println("  ┌─────────┐");
        System.out.println("  │  环 境  │");
        System.out.println("  └────┬────┘");
        System.out.println("       │ ① 观察状态 s_t");
        System.out.println("       ↓");
        System.out.println("  ┌─────────┐");
        System.out.println("  │ 智能体  │");
        System.out.println("  │ (策略π) │");
        System.out.println("  └────┬────┘");
        System.out.println("       │ ② 选择动作 a_t");
        System.out.println("       ↓");
        System.out.println("  ┌─────────┐");
        System.out.println("  │  环 境  │");
        System.out.println("  └────┬────┘");
        System.out.println("       │ ③ 返回 (s_{t+1}, r_t)");
        System.out.println("       ↓");
        System.out.println("  ┌─────────┐");
        System.out.println("  │ 智能体  │");
        System.out.println("  │ (学习)  │");
        System.out.println("  └────┬────┘");
        System.out.println("       │ ④ 更新策略/价值函数");
        System.out.println("       │");
        System.out.println("      循环");

        System.out.println("\n实际运行一个回合:");
        CartPoleEnvironment env = new CartPoleEnvironment(500);
        DQNAgent agent = new DQNAgent(
            "DemoAgent", 4, 2, new int[]{32, 32},
            0.001f, 0.1f, 0.99f, 32, 1000, 10
        );

        Variable state = env.reset();
        int step = 0;
        
        System.out.println("步骤 | 状态摘要 | 动作 | 奖励 | 新状态摘要");
        System.out.println("-----|---------|------|------|----------");
        
        while (!env.isDone() && step < 10) {
            Variable action = agent.selectAction(state);
            Environment.StepResult result = env.step(action);
            
            Experience experience = new Experience(
                state, action, result.getReward(),
                result.getNextState(), result.isDone(), step
            );
            agent.learn(experience);

            System.out.printf(" %2d  | [%.2f..] |  %d   | %.1f  | [%.2f..]\n",
                step + 1,
                state.getValue().get(0),
                (int) action.getValue().getNumber().floatValue(),
                result.getReward(),
                result.getNextState().getValue().get(0)
            );

            state = result.getNextState();
            step++;
        }

        System.out.println("\n💡 核心要点总结:");
        System.out.println("  • 智能体通过试错(Trial and Error)学习");
        System.out.println("  • 目标是最大化累积奖励");
        System.out.println("  • 需要平衡探索与利用");
        System.out.println("  • 学习过程是迭代优化策略/价值函数");
    }
}
