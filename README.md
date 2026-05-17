# AutoSolver: 让 AI Agent 自主求解配送分配问题

> 美团首届 AI Hackathon 2026 参赛作品 | 命题赛道：AutoSolver

## 项目简介

AutoSolver 是一个基于 AI Agent 的配送分配自主求解系统。系统核心是一个智能体（Agent），能够自主分析配送场景特征、选择最优求解算法、执行配送分配、评估方案质量并迭代优化，最终输出高质量的配送分配方案。

### 核心创新点

1. **AI Agent 自主决策**：Agent 根据场景特征（订单量、骑手数、紧急程度等）自主选择求解策略，无需人工干预
2. **多算法融合**：集成贪心算法、OR-Tools 约束规划求解器、遗传算法三种求解方法，Agent 自动选择和对比
3. **LLM 增强推理**：支持接入大语言模型，让 Agent 具备自然语言推理和工具调用能力
4. **动态模拟环境**：支持实时模拟配送过程，验证方案在动态场景下的鲁棒性
5. **可视化分析**：生成交互式地图和算法对比图表，直观展示分配方案

## 系统架构

```
┌─────────────────────────────────────────────┐
│              AutoSolver Agent                │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ 场景分析 │ │ 算法选择  │ │ 结果评估优化  │  │
│  └────┬────┘ └────┬─────┘ └──────┬───────┘  │
│       │           │              │           │
│  ┌────▼───────────▼──────────────▼───────┐  │
│  │           Tool Calling Layer          │  │
│  │  analyze_scenario | run_allocation    │  │
│  │  compare_results                      │  │
│  └────────────────┬─────────────────────┘  │
└───────────────────┼─────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
┌───▼───┐    ┌──────▼──────┐  ┌────▼────┐
│Greedy │    │  OR-Tools   │  │Genetic  │
│Allocator│   │   Solver    │  │Allocator│
└───────┘    └─────────────┘  └─────────┘
```

## 项目结构

```
meituan-AutoSolver/
├── autosolver/
│   ├── __init__.py              # 包入口
│   ├── models/                  # 数据模型
│   │   ├── order.py             # 订单模型
│   │   ├── rider.py             # 骑手模型
│   │   ├── restaurant.py        # 餐厅模型
│   │   ├── delivery_point.py    # 配送点模型
│   │   └── scenario.py          # 场景模型
│   ├── algorithms/              # 求解算法
│   │   ├── greedy.py            # 贪心算法
│   │   ├── or_tools_solver.py   # OR-Tools 求解器
│   │   ├── genetic.py           # 遗传算法
│   │   └── registry.py          # 算法注册中心
│   ├── agent/                   # AI Agent
│   │   ├── agent.py             # 核心Agent实现
│   │   └── tools.py             # Agent工具集
│   ├── visualization/           # 可视化
│   │   └── visualizer.py        # 地图与图表生成
│   └── simulation/              # 模拟环境
│       └── simulation.py        # 动态配送模拟
├── main.py                      # 命令行入口
├── app.py                       # Streamlit Web应用
├── pyproject.toml               # 项目配置
├── .env.example                 # 环境变量示例
└── .gitignore
```

## 快速开始

### 安装依赖

```bash
pip install -e .
```

### 命令行运行

```bash
python main.py
```

### Web 应用

```bash
streamlit run app.py
```

### 代码使用

```python
from autosolver import DeliveryScenario, AutoSolverAgent

# 创建配送场景
scenario = DeliveryScenario.create_scenario(
    scenario_id="demo",
    name="演示场景",
    description="50订单15骑手",
    area_bounds=(39.85, 116.25, 40.05, 116.55),  # 北京区域
    num_restaurants=10,
    num_orders=50,
    num_riders=15,
)

# Agent 自主求解
agent = AutoSolverAgent(verbose=True)
result = agent.solve(scenario)

# 查看结果
print(f"最优算法: {result.best_algorithm}")
print(f"准时率: {result.best_result['on_time_rate']:.1%}")
```

### 使用 LLM Agent

```python
agent = AutoSolverAgent(api_key="your-key", model_name="gpt-4o")
result = agent.solve_with_llm(scenario)
```

## 算法说明

| 算法 | 特点 | 适用场景 | 时间复杂度 |
|------|------|---------|-----------|
| Greedy | 基于距离+优先级的贪心分配 | 实时响应、小规模 | O(n·m) |
| OR-Tools | 约束规划精确求解 | 中小规模、质量优先 | 指数级(有界) |
| Genetic | 进化元启发式全局搜索 | 大规模、复杂约束 | O(g·p·n·m) |

## Agent 决策策略

1. **场景分析**：计算订单/骑手比、紧急订单比例、餐厅负载分布
2. **算法选择**：
   - 订单/骑手比 ≤ 2 且订单 ≤ 50 → OR-Tools（精确解）
   - 订单/骑手比 > 4 或订单 > 100 → 遗传算法（全局搜索）
   - 至少运行 2 种算法进行对比
3. **结果评估**：综合准时率、配送时间、负载均衡、求解速度
4. **迭代优化**：根据评估结果调整参数或切换算法

## 评估指标

- **准时率**：承诺时间内送达比例（目标 > 90%）
- **平均配送时间**：下单到送达平均时间（目标 < 30min）
- **负载均衡度**：骑手间工作量标准差（目标 < 1.5）
- **总配送距离**：所有骑手总行驶距离

## 技术栈

- Python 3.10+
- OpenAI API（LLM Agent 模式）
- Google OR-Tools（约束规划求解）
- NumPy / SciPy（数值计算）
- Streamlit（Web 界面）
- Folium / Leaflet（地图可视化）
- Matplotlib（图表生成）

## License

MIT
