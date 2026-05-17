from __future__ import annotations
from typing import Any
from ..models.scenario import DeliveryScenario
from ..models.order import OrderPriority
from ..algorithms.registry import AlgorithmRegistry
from ..algorithms.greedy import AllocationResult


class ScenarioAnalysisTool:
    name = "analyze_scenario"

    def analyze(self, scenario: DeliveryScenario) -> dict[str, Any]:
        urgent_count = sum(1 for o in scenario.orders if o.is_urgent)
        priority_dist = {}
        for p in OrderPriority:
            count = sum(1 for o in scenario.orders if o.priority == p)
            priority_dist[p.value] = count

        avg_prep_time = (
            sum(o.prep_time_minutes for o in scenario.orders) / max(1, len(scenario.orders))
        )

        restaurant_loads: dict[str, int] = {}
        for o in scenario.orders:
            restaurant_loads[o.restaurant_id] = restaurant_loads.get(o.restaurant_id, 0) + 1

        max_restaurant_load = max(restaurant_loads.values()) if restaurant_loads else 0
        hot_restaurants = sum(1 for v in restaurant_loads.values() if v > len(scenario.orders) / max(1, len(scenario.restaurant_loads)) * 2)

        available_riders = sum(1 for r in scenario.riders if r.is_available)
        avg_capacity = sum(r.max_orders for r in scenario.riders) / max(1, len(scenario.riders))

        return {
            "scenario_id": scenario.scenario_id,
            "num_orders": scenario.num_orders,
            "num_riders": scenario.num_riders,
            "num_restaurants": scenario.num_restaurants,
            "order_rider_ratio": scenario.order_rider_ratio,
            "difficulty_level": scenario.difficulty_level,
            "urgent_orders": urgent_count,
            "urgent_ratio": urgent_count / max(1, scenario.num_orders),
            "priority_distribution": priority_dist,
            "avg_prep_time": avg_prep_time,
            "max_restaurant_load": max_restaurant_load,
            "hot_restaurants": hot_restaurants,
            "available_riders": available_riders,
            "avg_rider_capacity": avg_capacity,
            "total_capacity": sum(r.max_orders for r in scenario.riders),
        }


class AllocationTool:
    name = "run_allocation"

    def run(self, algorithm_name: str, scenario: DeliveryScenario) -> dict[str, Any]:
        algorithm = AlgorithmRegistry.get(algorithm_name)
        result: AllocationResult = algorithm.solve(scenario.orders, scenario.riders)
        return {
            "algorithm": result.algorithm_name,
            "assignments": result.assignments,
            "total_distance": result.total_distance,
            "on_time_rate": result.on_time_rate,
            "avg_delivery_time": result.avg_delivery_time,
            "max_load_imbalance": result.max_load_imbalance,
            "num_assigned": result.num_assigned,
            "solve_time_seconds": result.solve_time_seconds,
            "metadata": result.metadata,
        }


class ComparisonTool:
    name = "compare_results"

    def compare(self, results: dict[str, dict]) -> dict[str, Any]:
        if not results:
            return {"best_algorithm": "none", "reason": "No results to compare"}

        scored: dict[str, float] = {}
        for name, result in results.items():
            score = 0.0
            score += result["on_time_rate"] * 40
            score += max(0, 1 - result["avg_delivery_time"] / 60) * 30
            score += max(0, 1 - result["max_load_imbalance"] / 5) * 20
            score += max(0, 1 - result["solve_time_seconds"] / 30) * 10
            scored[name] = score

        best = max(scored, key=scored.get)

        comparison = {}
        for name, result in results.items():
            comparison[name] = {
                "on_time_rate": result["on_time_rate"],
                "avg_delivery_time": result["avg_delivery_time"],
                "max_load_imbalance": result["max_load_imbalance"],
                "solve_time_seconds": result["solve_time_seconds"],
                "composite_score": scored[name],
            }

        best_result = results[best]
        reason_parts = []
        if best_result["on_time_rate"] == max(r["on_time_rate"] for r in results.values()):
            reason_parts.append("准时率最高")
        if best_result["avg_delivery_time"] == min(r["avg_delivery_time"] for r in results.values()):
            reason_parts.append("平均配送时间最短")
        if best_result["max_load_imbalance"] == min(r["max_load_imbalance"] for r in results.values()):
            reason_parts.append("负载最均衡")

        return {
            "best_algorithm": best,
            "best_score": scored[best],
            "reason": "、".join(reason_parts) if reason_parts else "综合评分最高",
            "comparison": comparison,
        }
