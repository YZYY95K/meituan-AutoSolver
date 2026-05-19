"""服务器端测试脚本"""
from solver import solve, _parse_input, _evaluate
import time

with open("large_seed301.txt", "r") as f:
    input_text = f.read()

candidates = _parse_input(input_text)

# AutoSolver
start = time.time()
result = solve(input_text)
elapsed = time.time() - start
metrics = _evaluate(result, candidates)

print("=== AutoSolver ===")
print("分配数:", metrics["num_assigned"])
print("覆盖任务数:", metrics["num_tasks_covered"])
print("总 Score:", round(metrics["total_score"], 3))
print("平均 Score:", round(metrics["avg_score"], 3))
print("平均 Willingness:", round(metrics["avg_willingness"], 4))
print("耗时:", round(elapsed, 3), "s")

# Baseline
def baseline_solve(input_text):
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    candidates_list = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        tid_str, cid, score_s, will_s = parts[:4]
        try:
            score = float(score_s)
            willingness = float(will_s)
        except ValueError:
            continue
        candidates_list.append((score, tid_str.strip(), cid.strip(), willingness))
    candidates_list.sort(key=lambda x: x[0])
    assigned_couriers = set()
    assigned_tasks = set()
    result_list = []
    for score, tid_str, cid, will in candidates_list:
        task_ids = [t.strip() for t in tid_str.split(",")]
        if cid in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in task_ids):
            continue
        assigned_couriers.add(cid)
        for t in task_ids:
            assigned_tasks.add(t)
        result_list.append((tid_str, [cid]))
    return result_list

start = time.time()
baseline_result = baseline_solve(input_text)
baseline_elapsed = time.time() - start
baseline_metrics = _evaluate(baseline_result, candidates)

print()
print("=== 贪心基线 ===")
print("分配数:", baseline_metrics["num_assigned"])
print("覆盖任务数:", baseline_metrics["num_tasks_covered"])
print("总 Score:", round(baseline_metrics["total_score"], 3))
print("平均 Score:", round(baseline_metrics["avg_score"], 3))
print("耗时:", round(baseline_elapsed, 3), "s")

print()
score_diff = baseline_metrics["total_score"] - metrics["total_score"]
task_diff = metrics["num_tasks_covered"] - baseline_metrics["num_tasks_covered"]
print("=== 对比 ===")
print("Score 改善:", round(score_diff, 3))
print("任务覆盖改善:", task_diff)
