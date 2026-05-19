"""详细测试：输出每个策略的结果"""
from solver import solve, _parse_input, _evaluate
from collections import defaultdict
import time

with open("large_seed301.txt", "r") as f:
    input_text = f.read()

candidates = _parse_input(input_text)

# 测试每个策略
from solver import (
    _greedy_assign, _solve_double_task_priority, _solve_task_by_task,
    _solve_double_task_combination, _solve_task_pair_search,
    _solve_reverse_greedy, _local_search, _select_best
)

task_to_cands = defaultdict(list)
courier_to_cands = defaultdict(list)
for c in candidates:
    task_to_cands[c["task_key"]].append(c)
    courier_to_cands[c["courier_id"]].append(c)

all_tasks = set()
for c in candidates:
    for t in c["task_ids"]:
        all_tasks.add(t)
n_tasks = len(all_tasks)

cand_map = {}
for c in candidates:
    key = (c["task_key"], c["courier_id"])
    cand_map[key] = c

strategies = [
    ("greedy_score", _greedy_assign(sorted(candidates, key=lambda c: c["score"]))),
    ("double_priority", _solve_double_task_priority(candidates, n_tasks)),
    ("task_by_task", _solve_task_by_task(candidates, all_tasks)),
    ("weighted", _greedy_assign(sorted(candidates, key=lambda c: c["score"] * (1.0 - c["willingness"] * 0.5)))),
    ("double_combo", _solve_double_task_combination(candidates, all_tasks, n_tasks)),
    ("task_pair_search", _solve_task_pair_search(candidates, all_tasks, cand_map)),
    ("reverse_greedy", _solve_reverse_greedy(candidates, all_tasks, cand_map)),
]

print(f"{'Strategy':<25} {'Assigned':>8} {'Tasks':>6} {'Total Score':>12} {'Avg Score':>10}")
print("-" * 65)

for name, result in strategies:
    m = _evaluate(result, candidates)
    print(f"{name:<25} {m['num_assigned']:>8} {m['num_tasks_covered']:>6} {m['total_score']:>12.3f} {m['avg_score']:>10.3f}")

# 局部搜索后
print("\n--- After Local Search ---")
print(f"{'Strategy':<25} {'Assigned':>8} {'Tasks':>6} {'Total Score':>12} {'Avg Score':>10}")
print("-" * 65)

for name, result in strategies:
    start = time.time()
    refined = _local_search(result, candidates, task_to_cands, courier_to_cands, cand_map, all_tasks)
    elapsed = time.time() - start
    m = _evaluate(refined, candidates)
    print(f"{name + '_ref':<25} {m['num_assigned']:>8} {m['num_tasks_covered']:>6} {m['total_score']:>12.3f} {m['avg_score']:>10.3f}  ({elapsed:.2f}s)")

# 最终solve
start = time.time()
final = solve(input_text)
elapsed = time.time() - start
fm = _evaluate(final, candidates)
print(f"\n=== Final Result ===")
print(f"Total Score: {fm['total_score']:.3f}")
print(f"Tasks Covered: {fm['num_tasks_covered']}")
print(f"Assigned: {fm['num_assigned']}")
print(f"Time: {elapsed:.3f}s")
