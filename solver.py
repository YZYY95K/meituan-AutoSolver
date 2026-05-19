"""
AutoSolver: AI Agent 自主求解配送分配问题
==========================================
美团 AI Hackathon 2026 | 命题赛道四

核心思路：
- 40个任务、80个骑手，每个骑手只能分配一次
- task_id_list 可含1-2个任务，双任务key可节省骑手
- 目标：最小化 total_score，覆盖所有40个任务
- 策略：多策略并行 + 局部搜索优化
"""

from collections import defaultdict
import itertools


def solve(input_text: str) -> list:
    """AutoSolver 主入口"""
    candidates = _parse_input(input_text)
    if not candidates:
        return []

    # 构建索引
    task_to_cands = defaultdict(list)
    courier_to_cands = defaultdict(list)
    task_id_to_cands = defaultdict(list)  # 按单个task_id索引
    for c in candidates:
        task_to_cands[c["task_key"]].append(c)
        courier_to_cands[c["courier_id"]].append(c)
        for t in c["task_ids"]:
            task_id_to_cands[t].append(c)

    all_tasks = set()
    for c in candidates:
        for t in c["task_ids"]:
            all_tasks.add(t)
    n_tasks = len(all_tasks)

    # 构建候选映射
    cand_map = {}
    for c in candidates:
        key = (c["task_key"], c["courier_id"])
        cand_map[key] = c

    # 运行多种策略
    results = []

    # 策略1: 基线贪心（score升序）
    r1 = _greedy_assign(sorted(candidates, key=lambda c: c["score"]))
    results.append(("greedy_score", r1))

    # 策略2: 双任务优先贪心
    r2 = _solve_double_task_priority(candidates, n_tasks)
    results.append(("double_priority", r2))

    # 策略3: 任务逐个最优分配
    r3 = _solve_task_by_task(candidates, all_tasks)
    results.append(("task_by_task", r3))

    # 策略4: 加权贪心
    r4 = _greedy_assign(sorted(candidates, key=lambda c: c["score"] * (1.0 - c["willingness"] * 0.5)))
    results.append(("weighted", r4))

    # 策略5: 双任务key组合优化（枚举双任务key数量）
    r5 = _solve_double_task_combination(candidates, all_tasks, n_tasks)
    results.append(("double_combo", r5))

    # 策略6: 任务对最优搜索（对每对任务找最优双任务key）
    r6 = _solve_task_pair_search(candidates, all_tasks, cand_map)
    results.append(("task_pair_search", r6))

    # 策略7: 反向贪心（先选最贵的任务，用双任务key覆盖）
    r7 = _solve_reverse_greedy(candidates, all_tasks, cand_map)
    results.append(("reverse_greedy", r7))

    # 对最优策略做轻量局部搜索（只做替换，不做双任务key合并）
    best_initial = _select_best(results, candidates, n_tasks)
    refined = _light_local_search(best_initial, candidates, task_to_cands, courier_to_cands, cand_map, all_tasks)
    refined_results = [("best_refined", refined)]

    all_results = results + refined_results

    # 选择最优方案
    best_result = _select_best(all_results, candidates, n_tasks)

    return best_result


def _parse_input(input_text: str) -> list[dict]:
    """解析输入数据"""
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    candidates = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue

        task_ids = tuple(t.strip() for t in task_id_list_str.split(","))
        task_key = task_id_list_str.strip()

        candidates.append({
            "task_key": task_key,
            "task_ids": task_ids,
            "courier_id": courier_id.strip(),
            "score": score,
            "willingness": willingness,
        })

    return candidates


def _greedy_assign(sorted_candidates: list[dict]) -> list:
    """通用贪心分配"""
    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for c in sorted_candidates:
        if c["courier_id"] in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in c["task_ids"]):
            continue

        assigned_couriers.add(c["courier_id"])
        for t in c["task_ids"]:
            assigned_tasks.add(t)
        result.append((c["task_key"], [c["courier_id"]]))

    return result


def _solve_double_task_priority(candidates: list[dict], n_tasks: int) -> list:
    """双任务key优先策略"""
    doubles = [c for c in candidates if len(c["task_ids"]) == 2]
    singles = [c for c in candidates if len(c["task_ids"]) == 1]

    doubles_sorted = sorted(doubles, key=lambda c: c["score"] / len(c["task_ids"]))
    singles_sorted = sorted(singles, key=lambda c: c["score"])

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for c in doubles_sorted:
        if c["courier_id"] in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in c["task_ids"]):
            continue
        assigned_couriers.add(c["courier_id"])
        for t in c["task_ids"]:
            assigned_tasks.add(t)
        result.append((c["task_key"], [c["courier_id"]]))

    for c in singles_sorted:
        if c["courier_id"] in assigned_couriers:
            continue
        if c["task_ids"][0] in assigned_tasks:
            continue
        assigned_couriers.add(c["courier_id"])
        assigned_tasks.add(c["task_ids"][0])
        result.append((c["task_key"], [c["courier_id"]]))

    return result


def _solve_task_by_task(candidates: list[dict], all_tasks: set) -> list:
    """逐任务最优分配"""
    task_best = defaultdict(list)
    for c in candidates:
        for t in c["task_ids"]:
            task_best[t].append(c)

    for t in all_tasks:
        task_best[t].sort(key=lambda c: c["score"])

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    task_order = sorted(all_tasks, key=lambda t: task_best[t][0]["score"] if task_best[t] else 999)

    for task in task_order:
        if task in assigned_tasks:
            continue

        best_c = None
        best_eff = float("inf")

        for c in task_best[task]:
            if c["courier_id"] in assigned_couriers:
                continue
            other_covered = all(t in assigned_tasks for t in c["task_ids"] if t != task)
            if other_covered and len(c["task_ids"]) > 1:
                eff = c["score"]
            elif any(t in assigned_tasks for t in c["task_ids"] if t != task):
                continue
            else:
                eff = c["score"] / len(c["task_ids"])

            if eff < best_eff:
                best_eff = eff
                best_c = c

        if best_c is not None:
            assigned_couriers.add(best_c["courier_id"])
            for t in best_c["task_ids"]:
                assigned_tasks.add(t)
            result.append((best_c["task_key"], [best_c["courier_id"]]))

    return result


def _solve_double_task_combination(candidates: list[dict], all_tasks: set, n_tasks: int) -> list:
    """双任务key组合优化"""
    doubles = [c for c in candidates if len(c["task_ids"]) == 2]
    singles = [c for c in candidates if len(c["task_ids"]) == 1]

    doubles_sorted = sorted(doubles, key=lambda c: c["score"] / 2)
    singles_sorted = sorted(singles, key=lambda c: c["score"])

    best_result = None
    best_total = float("inf")

    for n_doubles_target in range(0, min(21, n_tasks // 2 + 1)):
        assigned_couriers = set()
        assigned_tasks = set()
        result = []
        n_doubles_used = 0

        for c in doubles_sorted:
            if n_doubles_used >= n_doubles_target:
                break
            if c["courier_id"] in assigned_couriers:
                continue
            if any(t in assigned_tasks for t in c["task_ids"]):
                continue
            assigned_couriers.add(c["courier_id"])
            for t in c["task_ids"]:
                assigned_tasks.add(t)
            result.append((c["task_key"], [c["courier_id"]]))
            n_doubles_used += 1

        for c in singles_sorted:
            if c["courier_id"] in assigned_couriers:
                continue
            if c["task_ids"][0] in assigned_tasks:
                continue
            assigned_couriers.add(c["courier_id"])
            assigned_tasks.add(c["task_ids"][0])
            result.append((c["task_key"], [c["courier_id"]]))

        total = sum(_get_score(candidates, tk, cid) for tk, cids in result for cid in cids)
        if total < best_total and len(assigned_tasks) == n_tasks:
            best_total = total
            best_result = list(result)

    return best_result if best_result else _greedy_assign(sorted(candidates, key=lambda c: c["score"]))


def _solve_task_pair_search(candidates: list[dict], all_tasks: set, cand_map: dict) -> list:
    """任务对最优搜索：对每对任务找最优双任务key，贪心选择"""
    # 找出所有任务对的最优双任务key
    doubles = [c for c in candidates if len(c["task_ids"]) == 2]
    singles = [c for c in candidates if len(c["task_ids"]) == 1]

    # 对每对任务，找score最低的双任务key
    pair_best = {}
    for c in doubles:
        pair = tuple(sorted(c["task_ids"]))
        if pair not in pair_best or c["score"] < pair_best[pair]["score"]:
            pair_best[pair] = c

    # 对每个任务，找score最低的单任务key
    single_best = {}
    for c in singles:
        t = c["task_ids"][0]
        if t not in single_best or c["score"] < single_best[t]["score"]:
            single_best[t] = c

    # 计算每对任务的"节省量"：两个单任务最优之和 - 双任务最优
    pair_savings = []
    for pair, double_c in pair_best.items():
        t1, t2 = pair
        s1 = single_best.get(t1)
        s2 = single_best.get(t2)
        if s1 and s2:
            saving = (s1["score"] + s2["score"]) - double_c["score"]
            pair_savings.append((saving, pair, double_c))

    # 按节省量降序排列
    pair_savings.sort(key=lambda x: -x[0])

    # 贪心选择：选节省量最大的任务对
    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for saving, pair, double_c in pair_savings:
        if saving <= 0:
            break
        if double_c["courier_id"] in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in double_c["task_ids"]):
            continue
        assigned_couriers.add(double_c["courier_id"])
        for t in double_c["task_ids"]:
            assigned_tasks.add(t)
        result.append((double_c["task_key"], [double_c["courier_id"]]))

    # 用单任务key覆盖剩余
    singles_sorted = sorted(singles, key=lambda c: c["score"])
    for c in singles_sorted:
        if c["courier_id"] in assigned_couriers:
            continue
        if c["task_ids"][0] in assigned_tasks:
            continue
        assigned_couriers.add(c["courier_id"])
        assigned_tasks.add(c["task_ids"][0])
        result.append((c["task_key"], [c["courier_id"]]))

    return result


def _solve_reverse_greedy(candidates: list[dict], all_tasks: set, cand_map: dict) -> list:
    """反向贪心：先处理最难分配（最优score最高）的任务"""
    # 对每个任务找最优score
    task_best_score = {}
    task_best_cands = defaultdict(list)
    for c in candidates:
        for t in c["task_ids"]:
            task_best_cands[t].append(c)

    for t in all_tasks:
        if task_best_cands[t]:
            task_best_score[t] = min(c["score"] for c in task_best_cands[t])

    # 按最优score降序排列（最难的任务先处理）
    task_order = sorted(all_tasks, key=lambda t: -task_best_score.get(t, 999))

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for task in task_order:
        if task in assigned_tasks:
            continue

        # 找该任务的所有可用候选，优先双任务key
        available = []
        for c in task_best_cands[task]:
            if c["courier_id"] in assigned_couriers:
                continue
            if any(t in assigned_tasks for t in c["task_ids"] if t != task):
                continue
            # 计算有效成本
            new_tasks = [t for t in c["task_ids"] if t not in assigned_tasks]
            eff = c["score"] / len(new_tasks) if new_tasks else float("inf")
            available.append((eff, c))

        if available:
            available.sort(key=lambda x: x[0])
            best_c = available[0][1]
            assigned_couriers.add(best_c["courier_id"])
            for t in best_c["task_ids"]:
                assigned_tasks.add(t)
            result.append((best_c["task_key"], [best_c["courier_id"]]))

    return result


def _light_local_search(result: list, candidates: list[dict],
                        task_to_cands: dict, courier_to_cands: dict,
                        cand_map: dict, all_tasks: set) -> list:
    """轻量局部搜索：只做同task_key的骑手替换和交换，不改变覆盖"""
    if not result:
        return result

    courier_to_taskkey = {}
    for task_key, courier_ids in result:
        cid = courier_ids[0]
        courier_to_taskkey[cid] = task_key

    for _ in range(3):
        improved = False

        # 替换：找更低score的空闲骑手
        for cid, task_key in list(courier_to_taskkey.items()):
            current_c = cand_map.get((task_key, cid))
            if current_c is None:
                continue
            for c in task_to_cands.get(task_key, []):
                other_cid = c["courier_id"]
                if other_cid == cid or other_cid in courier_to_taskkey:
                    continue
                if c["score"] < current_c["score"]:
                    courier_to_taskkey.pop(cid)
                    courier_to_taskkey[other_cid] = task_key
                    improved = True
                    break

        # 交换：两个骑手互换任务
        courier_list = list(courier_to_taskkey.keys())
        for i in range(len(courier_list)):
            cid1 = courier_list[i]
            if cid1 not in courier_to_taskkey:
                continue
            tk1 = courier_to_taskkey[cid1]
            c1 = cand_map.get((tk1, cid1))
            if c1 is None:
                continue
            for j in range(i + 1, len(courier_list)):
                cid2 = courier_list[j]
                if cid2 not in courier_to_taskkey:
                    continue
                tk2 = courier_to_taskkey[cid2]
                c2 = cand_map.get((tk2, cid2))
                if c2 is None:
                    continue

                c1_swap = cand_map.get((tk1, cid2))
                c2_swap = cand_map.get((tk2, cid1))
                if c1_swap and c2_swap:
                    if c1_swap["score"] + c2_swap["score"] < c1["score"] + c2["score"]:
                        courier_to_taskkey[cid1] = tk2
                        courier_to_taskkey[cid2] = tk1
                        improved = True

        if not improved:
            break

    refined_result = []
    for cid, task_key in courier_to_taskkey.items():
        refined_result.append((task_key, [cid]))

    return refined_result


def _local_search(result: list, candidates: list[dict],
                  task_to_cands: dict, courier_to_cands: dict,
                  cand_map: dict, all_tasks: set) -> list:
    """局部搜索：尝试替换分配以降低总score，保持任务覆盖不减少"""
    if not result:
        return result

    # 构建当前分配状态
    courier_to_taskkey = {}
    for task_key, courier_ids in result:
        cid = courier_ids[0]
        courier_to_taskkey[cid] = task_key

    # 计算当前覆盖的任务
    def get_covered_tasks():
        covered = set()
        for cid, tk in courier_to_taskkey.items():
            c = cand_map.get((tk, cid))
            if c:
                for t in c["task_ids"]:
                    covered.add(t)
        return covered

    def calc_total():
        total = 0.0
        for cid, tk in courier_to_taskkey.items():
            c = cand_map.get((tk, cid))
            if c:
                total += c["score"]
        return total

    initial_covered = get_covered_tasks()

    # 多轮迭代
    for _ in range(5):
        improved = False

        # 尝试1：替换单个骑手（找更低score的骑手，同task_key）
        for cid, task_key in list(courier_to_taskkey.items()):
            current_c = cand_map.get((task_key, cid))
            if current_c is None:
                continue
            current_score = current_c["score"]

            for c in task_to_cands.get(task_key, []):
                other_cid = c["courier_id"]
                if other_cid == cid or other_cid in courier_to_taskkey:
                    continue
                if c["score"] < current_score:
                    courier_to_taskkey.pop(cid, None)
                    courier_to_taskkey[other_cid] = task_key
                    improved = True
                    break

        # 尝试2：交换两个骑手的任务分配
        courier_list = list(courier_to_taskkey.keys())
        n_couriers = len(courier_list)
        for i in range(n_couriers):
            cid1 = courier_list[i]
            if cid1 not in courier_to_taskkey:
                continue
            tk1 = courier_to_taskkey[cid1]
            c1 = cand_map.get((tk1, cid1))
            if c1 is None:
                continue
            for j in range(i + 1, min(i + 10, n_couriers)):
                cid2 = courier_list[j]
                if cid2 not in courier_to_taskkey:
                    continue
                tk2 = courier_to_taskkey[cid2]
                c2 = cand_map.get((tk2, cid2))
                if c2 is None:
                    continue

                current_total = c1["score"] + c2["score"]
                c1_swap = cand_map.get((tk1, cid2))
                c2_swap = cand_map.get((tk2, cid1))

                if c1_swap and c2_swap:
                    swap_total = c1_swap["score"] + c2_swap["score"]
                    if swap_total < current_total:
                        courier_to_taskkey[cid1] = tk2
                        courier_to_taskkey[cid2] = tk1
                        improved = True

        # 尝试3：用双任务key替换两个单任务key（节省骑手，然后用释放的骑手覆盖更多任务或降低score）
        singles_in_result = [(cid, tk) for cid, tk in courier_to_taskkey.items()
                            if len(cand_map.get((tk, cid), {}).get("task_ids", [])) == 1]

        # 预构建：每个任务对应的最优双任务key
        task_to_best_doubles = defaultdict(list)
        for dc_list in task_to_cands.values():
            for double_c in dc_list:
                if len(double_c["task_ids"]) != 2:
                    continue
                for t in double_c["task_ids"]:
                    task_to_best_doubles[t].append(double_c)

        for t in task_to_best_doubles:
            task_to_best_doubles[t].sort(key=lambda c: c["score"])

        for cid, task_key in singles_in_result[:15]:
            if cid not in courier_to_taskkey:
                continue
            c = cand_map.get((task_key, cid))
            if c is None or len(c["task_ids"]) != 1:
                continue

            task = c["task_ids"][0]
            best_double = None
            best_saving = 0
            best_other_cid = None

            for double_c in task_to_best_doubles.get(task, [])[:5]:
                if double_c["courier_id"] in courier_to_taskkey:
                    continue

                other_task = [t for t in double_c["task_ids"] if t != task][0]

                # 找另一个任务的当前骑手
                other_cid = None
                for ocid, otk in courier_to_taskkey.items():
                    oc = cand_map.get((otk, ocid))
                    if oc and other_task in oc["task_ids"]:
                        other_cid = ocid
                        break

                if other_cid is None or other_cid == cid:
                    continue

                other_c = cand_map.get((courier_to_taskkey[other_cid], other_cid))
                if other_c is None:
                    continue

                saving = (c["score"] + other_c["score"]) - double_c["score"]
                if saving > best_saving:
                    best_saving = saving
                    best_double = double_c
                    best_other_cid = other_cid

            if best_double and best_saving > 0:
                # 替换两个单任务为一个双任务，释放一个骑手
                courier_to_taskkey.pop(cid, None)
                courier_to_taskkey.pop(best_other_cid, None)
                courier_to_taskkey[best_double["courier_id"]] = best_double["task_key"]

                # 用释放的骑手覆盖新任务或降低现有任务的score
                freed_couriers = [cid, best_other_cid]
                current_covered = get_covered_tasks()
                uncovered = all_tasks - current_covered

                for freed_cid in freed_couriers:
                    if uncovered:
                        # 尝试用释放的骑手覆盖未覆盖的任务
                        best_new = None
                        for c in courier_to_cands.get(freed_cid, []):
                            if any(t in uncovered for t in c["task_ids"]):
                                if best_new is None or c["score"] < best_new["score"]:
                                    best_new = c
                        if best_new:
                            courier_to_taskkey[freed_cid] = best_new["task_key"]
                            for t in best_new["task_ids"]:
                                if t in uncovered:
                                    uncovered.discard(t)
                            continue

                    # 没有未覆盖任务，尝试替换现有分配中score更高的
                    current_total = calc_total()
                    best_replace = None
                    best_improvement = 0
                    for c in courier_to_cands.get(freed_cid, []):
                        # 找该候选能替换的现有分配
                        for exist_cid, exist_tk in list(courier_to_taskkey.items()):
                            if exist_cid == freed_cid:
                                continue
                            exist_c = cand_map.get((exist_tk, exist_cid))
                            if exist_c is None:
                                continue
                            # 检查freed_cid能否接手exist_cid的任务
                            swap_c = cand_map.get((exist_tk, freed_cid))
                            if swap_c and swap_c["score"] < exist_c["score"]:
                                improvement = exist_c["score"] - swap_c["score"]
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_replace = (exist_cid, exist_tk)

                    if best_replace:
                        old_cid, old_tk = best_replace
                        courier_to_taskkey[freed_cid] = old_tk
                        courier_to_taskkey.pop(old_cid, None)

                improved = True

        # 尝试4：用两个单任务key替换一个双任务key（如果更优且不减少覆盖）
        doubles_in_result = [(cid, tk) for cid, tk in courier_to_taskkey.items()
                            if len(cand_map.get((tk, cid), {}).get("task_ids", [])) == 2]

        for cid, task_key in doubles_in_result:
            if cid not in courier_to_taskkey:
                continue
            c = cand_map.get((task_key, cid))
            if c is None or len(c["task_ids"]) != 2:
                continue

            t1, t2 = c["task_ids"]

            best_s1 = None
            best_s2 = None
            for sc in task_to_cands.get(t1, []):
                if len(sc["task_ids"]) != 1:
                    continue
                if sc["courier_id"] in courier_to_taskkey and sc["courier_id"] != cid:
                    continue
                if sc["task_ids"][0] == t1:
                    if best_s1 is None or sc["score"] < best_s1["score"]:
                        best_s1 = sc
            for sc in task_to_cands.get(t2, []):
                if len(sc["task_ids"]) != 1:
                    continue
                if sc["courier_id"] in courier_to_taskkey and sc["courier_id"] != cid:
                    continue
                if sc["task_ids"][0] == t2:
                    if best_s2 is None or sc["score"] < best_s2["score"]:
                        best_s2 = sc

            if best_s1 and best_s2 and best_s1["courier_id"] != best_s2["courier_id"]:
                new_total = best_s1["score"] + best_s2["score"]
                if new_total < c["score"]:
                    courier_to_taskkey.pop(cid, None)
                    courier_to_taskkey[best_s1["courier_id"]] = best_s1["task_key"]
                    courier_to_taskkey[best_s2["courier_id"]] = best_s2["task_key"]
                    improved = True

        # 验证覆盖不减少
        current_covered = get_covered_tasks()
        if len(current_covered) < len(initial_covered):
            # 回滚到初始状态
            break

        if not improved:
            break

    # 重建结果
    refined_result = []
    for cid, task_key in courier_to_taskkey.items():
        refined_result.append((task_key, [cid]))

    return refined_result


def _get_score(candidates: list[dict], task_key: str, courier_id: str) -> float:
    """获取指定候选的score"""
    for c in candidates:
        if c["task_key"] == task_key and c["courier_id"] == courier_id:
            return c["score"]
    return float("inf")


def _evaluate(result: list, candidates: list[dict]) -> dict:
    """评估分配方案"""
    if not result:
        return {"total_score": float("inf"), "num_assigned": 0, "num_tasks_covered": 0,
                "avg_score": float("inf"), "avg_willingness": 0}

    cand_map = {}
    for c in candidates:
        key = (c["task_key"], c["courier_id"])
        cand_map[key] = c

    total_score = 0.0
    total_willingness = 0.0
    num_assigned = 0
    assigned_tasks = set()

    for task_key, courier_ids in result:
        cid = courier_ids[0]
        key = (task_key, cid)
        if key in cand_map:
            c = cand_map[key]
            total_score += c["score"]
            total_willingness += c["willingness"]
            num_assigned += 1
            for tid in c["task_ids"]:
                assigned_tasks.add(tid)

    return {
        "total_score": total_score,
        "num_assigned": num_assigned,
        "num_tasks_covered": len(assigned_tasks),
        "avg_score": total_score / max(1, num_assigned),
        "avg_willingness": total_willingness / max(1, num_assigned),
    }


def _select_best(all_results: list, candidates: list[dict], n_tasks: int) -> list:
    """选择最优方案"""
    best_result = None
    best_metric = float("inf")

    for name, result in all_results:
        metrics = _evaluate(result, candidates)
        if metrics["num_tasks_covered"] == 0:
            continue

        penalty = (n_tasks - metrics["num_tasks_covered"]) * 100
        metric = metrics["total_score"] + penalty

        if metric < best_metric:
            best_metric = metric
            best_result = result

    if best_result is None:
        sorted_cands = sorted(candidates, key=lambda c: c["score"])
        best_result = _greedy_assign(sorted_cands)

    return best_result
