import streamlit as st
from autosolver.models.scenario import DeliveryScenario
from autosolver.agent.agent import AutoSolverAgent
from autosolver.algorithms.registry import AlgorithmRegistry
from autosolver.visualization.visualizer import DeliveryVisualizer
from autosolver.simulation.simulation import DeliverySimulation, SimulationConfig


st.set_page_config(page_title="AutoSolver", page_icon="🛵", layout="wide")

st.title("🛵 AutoSolver - AI Agent 自主求解配送分配问题")
st.caption("Meituan AI Hackathon 2026 | 让 AI Agent 自主求解配送分配问题")

with st.sidebar:
    st.header("⚙️ 场景配置")
    num_restaurants = st.slider("餐厅数量", 3, 30, 10)
    num_orders = st.slider("订单数量", 10, 200, 50)
    num_riders = st.slider("骑手数量", 5, 50, 15)
    seed = st.number_input("随机种子", value=42)

    st.divider()
    st.header("🧪 模拟配置")
    enable_simulation = st.checkbox("启用动态模拟", value=False)
    sim_time = st.slider("模拟时长(分钟)", 30, 360, 120)
    order_rate = st.slider("订单生成速率", 0.5, 10.0, 2.0, 0.5)

    st.divider()
    st.header("🤖 Agent 配置")
    use_llm = st.checkbox("使用 LLM Agent", value=False)
    api_key = st.text_input("API Key", type="password")

beijing_bounds = (39.85, 116.25, 40.05, 116.55)

if st.button("🚀 开始求解", type="primary", use_container_width=True):
    with st.spinner("正在生成场景..."):
        scenario = DeliveryScenario.create_scenario(
            scenario_id="interactive",
            name="交互式场景",
            description=f"用户自定义: {num_orders}单/{num_riders}骑手",
            area_bounds=beijing_bounds,
            num_restaurants=num_restaurants,
            num_orders=num_orders,
            num_riders=num_riders,
            seed=seed,
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("订单数", scenario.num_orders)
    col2.metric("骑手数", scenario.num_riders)
    col3.metric("餐厅数", scenario.num_restaurants)
    col4.metric("难度", scenario.difficulty_level)

    with st.spinner("AutoSolver Agent 正在自主求解..."):
        agent = AutoSolverAgent(verbose=False)
        if use_llm and api_key:
            result = agent.solve_with_llm(scenario)
        else:
            result = agent.solve(scenario)

    st.success(f"求解完成! 最优算法: **{result.best_algorithm}**")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("准时率", f"{result.best_result['on_time_rate']:.1%}")
    mc2.metric("平均配送时间", f"{result.best_result['avg_delivery_time']:.1f}min")
    mc3.metric("负载偏差", f"{result.best_result['max_load_imbalance']:.2f}")
    mc4.metric("求解耗时", f"{result.best_result['solve_time_seconds']:.3f}s")

    st.subheader("📊 算法对比")
    if len(result.algorithm_results) > 1:
        import pandas as pd
        comparison_data = []
        for algo_name, algo_result in result.algorithm_results.items():
            comparison_data.append({
                "算法": algo_name,
                "准时率": f"{algo_result['on_time_rate']:.1%}",
                "平均配送时间(min)": f"{algo_result['avg_delivery_time']:.1f}",
                "负载偏差": f"{algo_result['max_load_imbalance']:.2f}",
                "求解时间(s)": f"{algo_result['solve_time_seconds']:.3f}",
            })
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    st.subheader("📋 分配方案")
    assignments = result.final_assignments
    for rider_id, order_ids in assignments.items():
        if order_ids:
            with st.expander(f"🛵 {rider_id} - {len(order_ids)}单"):
                st.write(f"订单: {', '.join(order_ids)}")

    if result.optimization_suggestions:
        st.subheader("💡 优化建议")
        for s in result.optimization_suggestions:
            st.info(f"• {s}")

    if enable_simulation:
        st.subheader("🧪 动态模拟")
        with st.spinner("正在运行动态模拟..."):
            sim_config = SimulationConfig(
                total_time_minutes=sim_time,
                order_generation_rate=order_rate,
                dynamic_orders=True,
                seed=seed,
            )
            sim = DeliverySimulation(scenario, sim_config, agent)
            sim_metrics = sim.run()

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("生成订单", sim_metrics.total_orders_generated)
        sc2.metric("已送达", sim_metrics.total_orders_delivered)
        sc3.metric("超时订单", sim_metrics.total_orders_overdue)
        sc4.metric("骑手利用率", f"{sim_metrics.rider_utilization:.1%}")

st.markdown("---")
st.caption("AutoSolver v1.0 | Meituan AI Hackathon 2026")
