from autosolver.models.scenario import DeliveryScenario
from autosolver.agent.agent import AutoSolverAgent
from autosolver.visualization.visualizer import DeliveryVisualizer
from autosolver.algorithms.registry import AlgorithmRegistry


def main():
    print("=" * 60)
    print("  AutoSolver - AI Agent 自主求解配送分配问题")
    print("  Meituan AI Hackathon 2026")
    print("=" * 60)
    print()

    beijing_bounds = (39.85, 116.25, 40.05, 116.55)

    scenarios = [
        ("easy", "简单场景", "低峰期少量订单", 5, 15, 8),
        ("medium", "中等场景", "常规时段中等订单量", 10, 50, 15),
        ("hard", "困难场景", "高峰期大量订单", 15, 100, 20),
    ]

    for scenario_id, name, desc, num_rest, num_orders, num_riders in scenarios:
        print(f"\n{'─' * 50}")
        print(f"📋 场景: {name} - {desc}")
        print(f"   餐厅: {num_rest} | 订单: {num_orders} | 骑手: {num_riders}")
        print(f"{'─' * 50}")

        scenario = DeliveryScenario.create_scenario(
            scenario_id=scenario_id,
            name=name,
            description=desc,
            area_bounds=beijing_bounds,
            num_restaurants=num_rest,
            num_orders=num_orders,
            num_riders=num_riders,
            seed=42,
        )

        agent = AutoSolverAgent(verbose=True)
        result = agent.solve(scenario)

        print(f"\n📊 最终结果:")
        print(f"   最优算法: {result.best_algorithm}")
        print(f"   准时率: {result.best_result['on_time_rate']:.1%}")
        print(f"   平均配送时间: {result.best_result['avg_delivery_time']:.1f}min")
        print(f"   负载偏差: {result.best_result['max_load_imbalance']:.2f}")

        if result.optimization_suggestions:
            print(f"\n💡 优化建议:")
            for s in result.optimization_suggestions:
                print(f"   • {s}")

        try:
            visualizer = DeliveryVisualizer()
            from autosolver.algorithms.greedy import AllocationResult
            algo = AlgorithmRegistry.get(result.best_algorithm)
            alloc_result = algo.solve(scenario.orders, scenario.riders)
            map_path = visualizer.generate_map_html(
                scenario, alloc_result, f"delivery_map_{scenario_id}.html"
            )
            print(f"\n🗺️  地图已生成: {map_path}")
        except Exception as e:
            print(f"\n⚠️  地图生成失败: {e}")

    print(f"\n{'=' * 60}")
    print("  ✅ 所有场景求解完成!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
