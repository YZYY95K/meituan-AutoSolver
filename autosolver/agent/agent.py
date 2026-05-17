from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openai import OpenAI

from ..models.scenario import DeliveryScenario
from ..algorithms.registry import AlgorithmRegistry
from .tools import AllocationTool, ScenarioAnalysisTool, ComparisonTool


SYSTEM_PROMPT = """你是 AutoSolver，一个专业的配送分配问题 AI Agent。你的任务是自主分析配送场景、选择最优算法、求解配送分配方案，并持续优化结果。

## 你的核心能力

1. **场景分析**：分析订单分布、骑手位置、时间约束等，评估问题规模和难度
2. **算法选择**：根据场景特征自主选择最合适的求解算法
3. **方案求解**：调用算法工具执行配送分配，获取分配方案
4. **结果评估**：评估方案质量，识别改进空间
5. **迭代优化**：尝试不同算法或参数，比较结果，选择最优方案

## 可用算法

- **greedy**：贪心算法 - 基于距离和优先级的快速分配，适合实时场景，求解速度快
- **or_tools**：OR-Tools求解器 - 基于约束规划的精确求解，适合中小规模问题，解质量高
- **genetic**：遗传算法 - 基于进化的元启发式求解，适合大规模复杂问题，全局搜索能力强

## 决策策略

1. 先分析场景：订单数量、骑手数量、订单/骑手比、紧急订单比例
2. 根据场景选择算法：
   - 订单/骑手比 ≤ 2 且订单 ≤ 30：优先使用 or_tools（精确解）
   - 订单/骑手比 > 4 或订单 > 100：优先使用 genetic（全局搜索）
   - 需要快速响应时：使用 greedy（实时性）
3. 至少运行2种算法进行对比
4. 选择综合指标最优的方案

## 评估指标

- **准时率**：在承诺时间内送达的订单比例（目标 > 90%）
- **平均配送时间**：从下单到送达的平均时间（目标 < 30分钟）
- **负载均衡度**：骑手间工作量的均衡程度（目标偏差 < 1.5）
- **总配送距离**：所有骑手的总行驶距离（越短越好）

## 输出格式

最终方案需要包含：
1. 场景分析摘要
2. 算法选择理由
3. 最优分配方案详情
4. 关键指标评估
5. 优化建议
"""


@dataclass
class AgentStep:
    step_type: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: Any | None = None


@dataclass
class AgentResult:
    scenario_analysis: dict
    algorithm_choices: list[str]
    algorithm_results: dict[str, dict]
    best_algorithm: str
    best_result: dict
    final_assignments: dict[str, list[str]]
    optimization_suggestions: list[str]
    steps: list[AgentStep] = field(default_factory=list)
    total_time_seconds: float = 0.0


class AutoSolverAgent:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = "gpt-4o",
        verbose: bool = True,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.verbose = verbose

        self.allocation_tool = AllocationTool()
        self.analysis_tool = ScenarioAnalysisTool()
        self.comparison_tool = ComparisonTool()

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_scenario",
                    "description": "分析配送场景的特征、难度和关键指标",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scenario_id": {
                                "type": "string",
                                "description": "场景ID",
                            }
                        },
                        "required": ["scenario_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_allocation",
                    "description": "使用指定算法运行配送分配求解",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "algorithm": {
                                "type": "string",
                                "enum": ["greedy", "or_tools", "genetic"],
                                "description": "求解算法名称",
                            },
                            "scenario_id": {
                                "type": "string",
                                "description": "场景ID",
                            },
                        },
                        "required": ["algorithm", "scenario_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_results",
                    "description": "比较不同算法的求解结果",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "待比较的算法结果列表",
                            }
                        },
                        "required": ["results"],
                    },
                },
            },
        ]

        self._scenarios: dict[str, DeliveryScenario] = {}

    def register_scenario(self, scenario: DeliveryScenario) -> None:
        self._scenarios[scenario.scenario_id] = scenario

    def solve(self, scenario: DeliveryScenario) -> AgentResult:
        import time

        total_start = time.time()
        steps: list[AgentStep] = []

        self.register_scenario(scenario)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"AutoSolver Agent 开始求解: {scenario.name}")
            print(f"场景: {scenario.num_orders}个订单, {scenario.num_riders}个骑手")
            print(f"难度: {scenario.difficulty_level}")
            print(f"{'='*60}\n")

        step = AgentStep(
            step_type="thinking",
            content="开始分析配送场景特征...",
        )
        steps.append(step)

        analysis = self.analysis_tool.analyze(scenario)
        steps.append(AgentStep(
            step_type="tool_call",
            content="场景分析完成",
            tool_name="analyze_scenario",
            tool_input={"scenario_id": scenario.scenario_id},
            tool_output=analysis,
        ))

        if self.verbose:
            print(f"📊 场景分析:")
            print(f"   订单/骑手比: {analysis['order_rider_ratio']:.1f}")
            print(f"   难度等级: {analysis['difficulty_level']}")
            print(f"   紧急订单: {analysis['urgent_orders']}个")
            print()

        algorithm_choices = self._select_algorithms(analysis)
        steps.append(AgentStep(
            step_type="decision",
            content=f"根据场景分析，选择算法: {', '.join(algorithm_choices)}",
        ))

        if self.verbose:
            print(f"🤖 选择算法: {', '.join(algorithm_choices)}")
            print()

        algorithm_results: dict[str, dict] = {}
        for algo_name in algorithm_choices:
            if self.verbose:
                print(f"⚙️  运行 {algo_name} 算法...")

            result = self.allocation_tool.run(algo_name, scenario)
            algorithm_results[algo_name] = result

            steps.append(AgentStep(
                step_type="tool_call",
                content=f"{algo_name} 算法求解完成",
                tool_name="run_allocation",
                tool_input={"algorithm": algo_name, "scenario_id": scenario.scenario_id},
                tool_output=result,
            ))

            if self.verbose:
                print(f"   准时率: {result['on_time_rate']:.1%}")
                print(f"   平均配送时间: {result['avg_delivery_time']:.1f}min")
                print(f"   负载偏差: {result['max_load_imbalance']:.2f}")
                print(f"   求解耗时: {result['solve_time_seconds']:.3f}s")
                print()

        if len(algorithm_results) > 1:
            comparison = self.comparison_tool.compare(algorithm_results)
            steps.append(AgentStep(
                step_type="tool_call",
                content="算法对比完成",
                tool_name="compare_results",
                tool_output=comparison,
            ))

            best_algorithm = comparison["best_algorithm"]
            if self.verbose:
                print(f"🏆 最优算法: {best_algorithm}")
                print(f"   理由: {comparison['reason']}")
                print()
        else:
            best_algorithm = algorithm_choices[0]

        best_result = algorithm_results[best_algorithm]
        suggestions = self._generate_suggestions(analysis, best_result)

        total_time = time.time() - total_start

        if self.verbose:
            print(f"{'='*60}")
            print(f"✅ 求解完成! 总耗时: {total_time:.2f}s")
            print(f"最优方案: {best_algorithm}")
            print(f"{'='*60}\n")

        return AgentResult(
            scenario_analysis=analysis,
            algorithm_choices=algorithm_choices,
            algorithm_results=algorithm_results,
            best_algorithm=best_algorithm,
            best_result=best_result,
            final_assignments=best_result["assignments"],
            optimization_suggestions=suggestions,
            steps=steps,
            total_time_seconds=total_time,
        )

    def solve_with_llm(self, scenario: DeliveryScenario) -> AgentResult:
        import time

        total_start = time.time()
        self.register_scenario(scenario)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请自主求解以下配送分配问题：\n"
                    f"场景ID: {scenario.scenario_id}\n"
                    f"场景名称: {scenario.name}\n"
                    f"订单数量: {scenario.num_orders}\n"
                    f"骑手数量: {scenario.num_riders}\n"
                    f"餐厅数量: {scenario.num_restaurants}\n"
                    f"订单/骑手比: {scenario.order_rider_ratio:.1f}\n"
                    f"难度等级: {scenario.difficulty_level}\n\n"
                    f"请先分析场景，然后选择合适的算法求解，最后比较结果并给出最优方案。"
                ),
            },
        ]

        max_iterations = 10
        all_results: dict[str, dict] = {}

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    if func_name == "analyze_scenario":
                        sid = func_args["scenario_id"]
                        result = self.analysis_tool.analyze(self._scenarios[sid])
                    elif func_name == "run_allocation":
                        algo = func_args["algorithm"]
                        sid = func_args["scenario_id"]
                        result = self.allocation_tool.run(algo, self._scenarios[sid])
                        all_results[algo] = result
                    elif func_name == "compare_results":
                        result = self.comparison_tool.compare(all_results)
                    else:
                        result = {"error": f"Unknown tool: {func_name}"}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
            else:
                messages.append(msg)
                break

        analysis = self.analysis_tool.analyze(scenario)

        if all_results:
            if len(all_results) > 1:
                comparison = self.comparison_tool.compare(all_results)
                best_algorithm = comparison["best_algorithm"]
            else:
                best_algorithm = list(all_results.keys())[0]
            best_result = all_results[best_algorithm]
        else:
            best_algorithm = "greedy"
            result = self.allocation_tool.run("greedy", scenario)
            best_result = result
            all_results["greedy"] = result

        suggestions = self._generate_suggestions(analysis, best_result)

        return AgentResult(
            scenario_analysis=analysis,
            algorithm_choices=list(all_results.keys()),
            algorithm_results=all_results,
            best_algorithm=best_algorithm,
            best_result=best_result,
            final_assignments=best_result["assignments"],
            optimization_suggestions=suggestions,
            total_time_seconds=time.time() - total_start,
        )

    def _select_algorithms(self, analysis: dict) -> list[str]:
        ratio = analysis["order_rider_ratio"]
        num_orders = analysis["num_orders"]
        difficulty = analysis["difficulty_level"]
        urgent = analysis["urgent_orders"]

        choices = []

        if difficulty in ("easy", "medium") and num_orders <= 50:
            choices.append("or_tools")
        elif difficulty in ("hard", "extreme") or num_orders > 100:
            choices.append("genetic")
        else:
            choices.append("or_tools")
            choices.append("genetic")

        choices.append("greedy")

        if urgent > num_orders * 0.3:
            if "greedy" not in choices:
                choices.append("greedy")

        return list(dict.fromkeys(choices))

    def _generate_suggestions(self, analysis: dict, best_result: dict) -> list[str]:
        suggestions = []

        if best_result["on_time_rate"] < 0.9:
            suggestions.append("准时率低于90%，建议增加骑手数量或调整配送区域划分")

        if best_result["max_load_imbalance"] > 2.0:
            suggestions.append("骑手负载不均衡，建议优化分配策略或调整骑手工作区域")

        if analysis["urgent_orders"] > analysis["num_orders"] * 0.3:
            suggestions.append("紧急订单比例较高，建议启用优先调度策略")

        if analysis["order_rider_ratio"] > 5:
            suggestions.append("订单/骑手比过高，建议增加运力或实施动态定价调节需求")

        if best_result["avg_delivery_time"] > 35:
            suggestions.append("平均配送时间较长，建议优化路径规划或增加中转站点")

        if not suggestions:
            suggestions.append("当前方案各项指标良好，建议持续监控并动态调整")

        return suggestions
