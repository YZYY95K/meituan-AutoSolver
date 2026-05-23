"""
AutoSolver v10: 基于v9safe的优化版
==========================================
优化点：
1. 骑手排序优化：输出前按willingness降序排列（最先接起者获得订单）
2. 期望score模型修正：顺序接单模型替代加权平均
3. 统一评估函数：用_m5_total替代_expected_cost
4. 减少时间缓冲：7.5s -> 9.2s
5. 启用random策略
6. repair后添加backup
7. penalty细粒度扫描
"""

from collections import defaultdict


def solve(input_text: str) -> list:
    """AutoSolver 主入口（带异常兜底）"""
    try:
        result = _solve_main(input_text)
        if not result:
            return _baseline_greedy(input_text)
        return result
    except Exception as exc:
        import sys, traceback
        print(f"[solver] main failed: {exc!r}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        try:
            return _baseline_greedy(input_text)
        except Exception as exc2:
            print(f"[solver] baseline also failed: {exc2!r}", file=sys.stderr)
            return []


def _baseline_greedy(input_text: str) -> list:
    """与 example_solution.txt 等价的最简贪心兜底"""
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    rows = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        tid_str, cid, score_s, _ = parts[:4]
        try:
            score = float(score_s)
        except ValueError:
            continue
        rows.append((score, tid_str.strip(), cid.strip()))
    rows.sort(key=lambda x: x[0])
    used_c, used_t = set(), set()
    out = []
    for _, tid_str, cid in rows:
        tids = [t.strip() for t in tid_str.split(",")]
        if cid in used_c or any(t in used_t for t in tids):
            continue
        used_c.add(cid)
        for t in tids:
            used_t.add(t)
        out.append((tid_str, [cid]))
    return out


# ============================================================
# 核心：顺序接单模型下的 slot_cost
# ============================================================
def _slot_cost_ordered(tk, cids, cand_map, penalty):
    """顺序接单模型：骑手按列表顺序尝试接单，第一个接起的获得订单。
    E[cost] = sum_k [P(k接单) * score_k] + P(全部拒单) * penalty * n
    P(k接单) = w_k * prod(1-w_j, j<k)
    """
    if not cids:
        return penalty * 100
    first_c = cand_map.get((tk, cids[0]))
    if not first_c:
        return penalty * 100
    n = len(first_c["task_ids"])

    p_all_rej = 1.0
    expected_score = 0.0
    for cid in cids:
        c = cand_map.get((tk, cid))
        if c:
            w = c["willingness"]
            p_k_accept = w * p_all_rej  # 前面都拒，k接
            expected_score += p_k_accept * c["score"]
            p_all_rej *= (1 - w)

    p_any_accept = 1 - p_all_rej
    return p_any_accept * expected_score + p_all_rej * penalty * n


def _slot_cost_weighted(tk, cids, cand_map, penalty):
    """加权平均模型（v9safe原始版本，保留对比）"""
    if not cids:
        return penalty * 100
    first_c = cand_map.get((tk, cids[0]))
    if not first_c:
        return penalty * 100
    n = len(first_c["task_ids"])
    p_rej = 1.0
    sum_w = 0.0
    sum_ws = 0.0
    for cid in cids:
        c = cand_map.get((tk, cid))
        if c:
            p_rej *= (1 - c["willingness"])
            sum_w += c["willingness"]
            sum_ws += c["willingness"] * c["score"]
    avg = sum_ws / sum_w if sum_w > 0 else 0.0
    return (1 - p_rej) * avg + p_rej * penalty * n


# 使用顺序接单模型作为默认
_slot_cost = _slot_cost_ordered


def _m5_total(assignments, cand_map, all_tasks, penalty):
    """整解 M5 总成本（= 各分派 slot_cost 之和 + 未覆盖任务 penalty）。"""
    total = 0.0
    covered = set()
    for tk, cids in assignments:
        if not cids:
            continue
        first_c = cand_map.get((tk, cids[0]))
        if not first_c:
            continue
        total += _slot_cost(tk, cids, cand_map, penalty)
        for t in first_c["task_ids"]:
            covered.add(t)
    total += (len(all_tasks) - len(covered)) * penalty
    return total


def _solve_main(input_text: str) -> list:
    """主算法：willingness 加权 + 多骑手 backup，扫描多个 penalty 值 + ALNS。"""
    import time
    start_ts = time.time()
    deadline = start_ts + 9.2  # v10: 减少缓冲到0.8s

    candidates = _parse_input(input_text)
    if not candidates:
        return []

    all_tasks = set()
    all_couriers = set()
    taskkey_to_cands = defaultdict(list)
    cand_map = {}
    for c in candidates:
        all_couriers.add(c["courier_id"])
        for t in c["task_ids"]:
            all_tasks.add(t)
        taskkey_to_cands[c["task_key"]].append(c)
        cand_map[(c["task_key"], c["courier_id"])] = c

    # v10: 细粒度penalty扫描 + 4种策略
    best_assignments = None
    best_metric = float("inf")
    best_covered = 0

    strategies = ("global", "hard_first", "doubles_first", "random")
    penalties = (60, 80, 100, 120, 150, 200, 300)

    for primary_strategy in strategies:
        for penalty in penalties:
            if primary_strategy == "global":
                primary = _pick_primary(candidates, all_tasks, penalty)
            elif primary_strategy == "hard_first":
                primary = _pick_primary_hard_first(candidates, all_tasks, penalty)
            elif primary_strategy == "doubles_first":
                primary = _pick_primary_doubles_first(candidates, all_tasks, penalty)
            else:
                primary = _pick_primary_random(candidates, all_tasks, penalty)
            primary = _rescue_coverage(primary, candidates, all_tasks, cand_map)

            covered = set()
            for tk, cid in primary:
                c = cand_map.get((tk, cid))
                if c:
                    for t in c["task_ids"]:
                        covered.add(t)

            assignments = _add_backups(primary, all_couriers, taskkey_to_cands, cand_map, penalty)
            assignments = _local_swap(assignments, taskkey_to_cands, cand_map, penalty)

            # v10: 用_m5_total统一评估（含未覆盖penalty）
            metric = _m5_total(assignments, cand_map, all_tasks, penalty)
            if metric < best_metric:
                best_metric = metric
                best_assignments = assignments
                best_covered = len(covered)

    # 兜底：用 baseline 作为额外候选
    baseline_out = _baseline_greedy(input_text)
    baseline_covered = set()
    for tk, cids in baseline_out:
        for cid in cids:
            c = cand_map.get((tk, cid))
            if c:
                for t in c["task_ids"]:
                    baseline_covered.add(t)

    if not best_assignments or len(baseline_covered) > best_covered:
        primary_from_baseline = [(tk, cids[0]) for tk, cids in baseline_out if cids]
        assignments = _add_backups(primary_from_baseline, all_couriers, taskkey_to_cands, cand_map, 100)
        assignments = _local_swap(assignments, taskkey_to_cands, cand_map, 100)
        best_assignments = assignments

    if not best_assignments:
        return []

    # inter_swap
    best_assignments = _inter_swap(
        best_assignments, all_couriers, taskkey_to_cands, cand_map, 100)

    # ALNS
    if time.time() < deadline - 0.3:
        best_assignments = _alns(
            best_assignments, candidates, all_tasks, all_couriers,
            cand_map, taskkey_to_cands, 100, deadline)

    # v10核心优化：输出前对每个task_key的骑手按willingness降序排列
    result = []
    for tk, cids in best_assignments:
        # 按willingness降序排列：高w骑手优先接单，降低期望score
        sorted_cids = sorted(cids, key=lambda cid: -cand_map.get((tk, cid), {}).get("willingness", 0))
        result.append((tk, sorted_cids))

    return result


def _pick_primary_hard_first(candidates, all_tasks, penalty):
    """难任务优先：候选最少的任务先选骑手，避免被全局贪心挤掉。"""
    task_to_cands = {}
    for c in candidates:
        for t in c["task_ids"]:
            task_to_cands.setdefault(t, []).append(c)

    task_order = sorted(all_tasks, key=lambda t: len(task_to_cands.get(t, [])))

    used_c = set()
    used_t = set()
    out = []
    for task in task_order:
        if task in used_t:
            continue
        cands_for_task = []
        for c in task_to_cands.get(task, []):
            if c["courier_id"] in used_c:
                continue
            if any(t in used_t for t in c["task_ids"] if t != task):
                continue
            n = len(c["task_ids"])
            w = c["willingness"]
            cost = (w * c["score"] + (1 - w) * penalty * n) / n
            cands_for_task.append((cost, c))

        if not cands_for_task:
            continue
        cands_for_task.sort(key=lambda x: x[0])
        _, best = cands_for_task[0]
        out.append((best["task_key"], best["courier_id"]))
        used_c.add(best["courier_id"])
        for t in best["task_ids"]:
            used_t.add(t)

    return out


def _pick_primary_doubles_first(candidates, all_tasks, penalty):
    """双任务候选优先"""
    enriched = []
    for c in candidates:
        n = len(c["task_ids"])
        w = c["willingness"]
        per_task_cost = (w * c["score"] + (1 - w) * penalty * n) / n
        enriched.append(((0 if n >= 2 else 1, per_task_cost), c))
    enriched.sort(key=lambda x: x[0])

    used_c = set()
    used_t = set()
    out = []
    n_total = len(all_tasks)
    for _, c in enriched:
        if c["courier_id"] in used_c:
            continue
        if any(t in used_t for t in c["task_ids"]):
            continue
        out.append((c["task_key"], c["courier_id"]))
        used_c.add(c["courier_id"])
        for t in c["task_ids"]:
            used_t.add(t)
        if len(used_t) >= n_total:
            break
    return out


def _pick_primary_random(candidates, all_tasks, penalty):
    """全局贪心 + 固定种子随机扰动"""
    import random
    rng = random.Random(42 + int(penalty))
    enriched = []
    for c in candidates:
        n = len(c["task_ids"])
        w = c["willingness"]
        per_task_cost = (w * c["score"] + (1 - w) * penalty * n) / n
        enriched.append((per_task_cost + rng.random() * 0.5, c))
    enriched.sort(key=lambda x: x[0])

    used_c = set()
    used_t = set()
    out = []
    n_total = len(all_tasks)
    for _, c in enriched:
        if c["courier_id"] in used_c:
            continue
        if any(t in used_t for t in c["task_ids"]):
            continue
        out.append((c["task_key"], c["courier_id"]))
        used_c.add(c["courier_id"])
        for t in c["task_ids"]:
            used_t.add(t)
        if len(used_t) >= n_total:
            break
    return out


def _rescue_coverage(primary, candidates, all_tasks, cand_map):
    """补全未覆盖任务"""
    covered = set()
    used_couriers = set()
    for tk, cid in primary:
        used_couriers.add(cid)
        c = cand_map.get((tk, cid))
        if c:
            for t in c["task_ids"]:
                covered.add(t)

    uncovered = all_tasks - covered
    if not uncovered:
        return primary

    task_to_cands = {}
    for c in candidates:
        for t in c["task_ids"]:
            task_to_cands.setdefault(t, []).append(c)

    uncov_order = sorted(uncovered, key=lambda t: len(task_to_cands.get(t, [])))

    extended = list(primary)
    for task in uncov_order:
        if task in covered:
            continue
        candidates_for_task = []
        for c in task_to_cands.get(task, []):
            if c["courier_id"] in used_couriers:
                continue
            other_tasks = [t for t in c["task_ids"] if t != task]
            if any(t in covered for t in other_tasks):
                continue
            candidates_for_task.append(c)

        if not candidates_for_task:
            for c in task_to_cands.get(task, []):
                if c["courier_id"] in used_couriers:
                    continue
                candidates_for_task.append(c)

        if not candidates_for_task:
            continue

        best = min(candidates_for_task, key=lambda c: c["score"])
        extended.append((best["task_key"], best["courier_id"]))
        used_couriers.add(best["courier_id"])
        for t in best["task_ids"]:
            covered.add(t)

    return extended


def _pick_primary(candidates, all_tasks, penalty):
    """贪心：按每任务期望代价升序选不冲突的"""
    enriched = []
    for c in candidates:
        n = len(c["task_ids"])
        w = c["willingness"]
        per_task_cost = (w * c["score"] + (1 - w) * penalty * n) / n
        enriched.append((per_task_cost, c))
    enriched.sort(key=lambda x: x[0])

    used_c = set()
    used_t = set()
    out = []
    n_total = len(all_tasks)
    for _, c in enriched:
        if c["courier_id"] in used_c:
            continue
        if any(t in used_t for t in c["task_ids"]):
            continue
        out.append((c["task_key"], c["courier_id"]))
        used_c.add(c["courier_id"])
        for t in c["task_ids"]:
            used_t.add(t)
        if len(used_t) >= n_total:
            break
    return out


def _add_backups(primary, all_couriers, taskkey_to_cands, cand_map, penalty):
    """对每个 primary 分派，迭代添加 backup 骑手，直到没有正收益。"""
    assignments = []
    tk_index = {}
    used_couriers = set()
    for tk, cid in primary:
        tk_index[tk] = len(assignments)
        assignments.append((tk, [cid]))
        used_couriers.add(cid)

    while True:
        best_gain = 1e-6
        best_choice = None

        for idx, (tk, cids) in enumerate(assignments):
            # v10: 用顺序接单模型计算当前成本
            current_cost = _slot_cost(tk, cids, cand_map, penalty)
            first_c = cand_map.get((tk, cids[0]))
            if not first_c:
                continue
            n_tasks = len(first_c["task_ids"])

            p_rej = 1.0
            for cid in cids:
                cc = cand_map.get((tk, cid))
                if cc:
                    p_rej *= (1 - cc["willingness"])
            if p_rej < 0.03:
                continue

            for c in taskkey_to_cands[tk]:
                cid = c["courier_id"]
                if cid in used_couriers:
                    continue
                # v10: 用完整的slot_cost计算收益，而非近似
                new_cids = cids + [cid]
                new_cost = _slot_cost(tk, new_cids, cand_map, penalty)
                gain = current_cost - new_cost
                if gain > best_gain:
                    best_gain = gain
                    best_choice = (idx, cid)

        if best_choice is None:
            break
        idx, cid = best_choice
        tk, cids = assignments[idx]
        cids.append(cid)
        used_couriers.add(cid)

    return assignments


def _local_swap(assignments, taskkey_to_cands, cand_map, penalty):
    """局部搜索：替换骑手 + 删除负收益备份"""
    used_couriers = set()
    for tk, cids in assignments:
        for cid in cids:
            used_couriers.add(cid)

    improved = True
    rounds = 0
    while improved and rounds < 5:
        improved = False
        rounds += 1

        for i, (tk, cids) in enumerate(assignments):
            current_cost = _slot_cost(tk, cids, cand_map, penalty)
            for pos in range(len(cids)):
                old_cid = cids[pos]
                for c in taskkey_to_cands[tk]:
                    new_cid = c["courier_id"]
                    if new_cid == old_cid or new_cid in used_couriers:
                        continue
                    cids[pos] = new_cid
                    new_cost = _slot_cost(tk, cids, cand_map, penalty)
                    if new_cost < current_cost - 1e-6:
                        used_couriers.discard(old_cid)
                        used_couriers.add(new_cid)
                        current_cost = new_cost
                        improved = True
                        break
                    else:
                        cids[pos] = old_cid

        for i, (tk, cids) in enumerate(assignments):
            if len(cids) <= 1:
                continue
            current_cost = _slot_cost(tk, cids, cand_map, penalty)
            for pos in range(len(cids) - 1, 0, -1):
                removed_cid = cids.pop(pos)
                new_cost = _slot_cost(tk, cids, cand_map, penalty)
                if new_cost < current_cost - 1e-6:
                    used_couriers.discard(removed_cid)
                    current_cost = new_cost
                    improved = True
                else:
                    cids.insert(pos, removed_cid)

    return assignments


def _inter_swap(assignments, all_couriers, taskkey_to_cands, cand_map, penalty):
    """单调局部搜索：SWAP / MOVE / ADD / REMOVE"""
    used = set()
    for tk, cids in assignments:
        for cid in cids:
            used.add(cid)
    free = set(all_couriers) - used

    n_asgn = len(assignments)
    improved = True
    rounds = 0
    while improved and rounds < 8:
        improved = False
        rounds += 1

        for i in range(n_asgn):
            tk_i, cids_i = assignments[i]
            for j in range(n_asgn):
                if i == j:
                    continue
                tk_j, cids_j = assignments[j]
                ci_before = _slot_cost(tk_i, cids_i, cand_map, penalty)
                cj_before = _slot_cost(tk_j, cids_j, cand_map, penalty)
                base = ci_before + cj_before
                done = False
                for px in range(len(cids_i)):
                    X = cids_i[px]
                    if (tk_j, X) not in cand_map:
                        continue
                    if X in cids_j:
                        continue
                    for py in range(len(cids_j)):
                        Y = cids_j[py]
                        if X == Y:
                            continue
                        if (tk_i, Y) not in cand_map:
                            continue
                        if Y in cids_i:
                            continue
                        cids_i[px], cids_j[py] = Y, X
                        new_cost = _slot_cost(tk_i, cids_i, cand_map, penalty) + _slot_cost(tk_j, cids_j, cand_map, penalty)
                        if new_cost < base - 1e-6:
                            improved = True
                            done = True
                            break
                        cids_i[px], cids_j[py] = X, Y
                    if done:
                        break
                if done:
                    continue
                if len(cids_i) <= 1:
                    continue
                for px in range(len(cids_i)):
                    X = cids_i[px]
                    if (tk_j, X) not in cand_map:
                        continue
                    if X in cids_j:
                        continue
                    removed = cids_i.pop(px)
                    cids_j.append(X)
                    new_cost = _slot_cost(tk_i, cids_i, cand_map, penalty) + _slot_cost(tk_j, cids_j, cand_map, penalty)
                    if new_cost < base - 1e-6:
                        improved = True
                        break
                    cids_j.pop()
                    cids_i.insert(px, removed)

        for j in range(n_asgn):
            tk_j, cids_j = assignments[j]
            cj_before = _slot_cost(tk_j, cids_j, cand_map, penalty)
            best_gain = 0.0
            best_X = None
            for X in list(free):
                if (tk_j, X) not in cand_map:
                    continue
                if X in cids_j:
                    continue
                cids_j.append(X)
                new_cost = _slot_cost(tk_j, cids_j, cand_map, penalty)
                cids_j.pop()
                gain = cj_before - new_cost
                if gain > best_gain + 1e-6:
                    best_gain = gain
                    best_X = X
            if best_X is not None:
                cids_j.append(best_X)
                free.discard(best_X)
                improved = True

        for j in range(n_asgn):
            tk_j, cids_j = assignments[j]
            if len(cids_j) <= 1:
                continue
            cj_before = _slot_cost(tk_j, cids_j, cand_map, penalty)
            for pos in range(len(cids_j) - 1, 0, -1):
                removed = cids_j.pop(pos)
                new_cost = _slot_cost(tk_j, cids_j, cand_map, penalty)
                if new_cost < cj_before - 1e-6:
                    free.add(removed)
                    cj_before = new_cost
                    improved = True
                else:
                    cids_j.insert(pos, removed)

    return assignments


def _destroy_random(assignments, k, rng):
    n = len(assignments)
    if n == 0:
        return []
    k = min(k, n)
    indices = rng.sample(range(n), k)
    removed = []
    for i in sorted(indices, reverse=True):
        removed.append(assignments.pop(i))
    return removed


def _destroy_worst(assignments, k, cand_map, penalty):
    n = len(assignments)
    if n == 0:
        return []
    k = min(k, n)
    costs = [(_slot_cost(tk, cids, cand_map, penalty), i)
             for i, (tk, cids) in enumerate(assignments)]
    costs.sort(reverse=True)
    indices = [costs[i][1] for i in range(k)]
    removed = []
    for i in sorted(indices, reverse=True):
        removed.append(assignments.pop(i))
    return removed


def _repair_greedy(assignments, candidates, all_tasks, cand_map, penalty):
    """对每个未覆盖任务，贪心选边际 M5 成本最低的合法候选作为 primary 加入。"""
    covered = set()
    used = set()
    for tk, cids in assignments:
        for cid in cids:
            used.add(cid)
            c = cand_map.get((tk, cid))
            if c:
                for t in c["task_ids"]:
                    covered.add(t)
    uncovered = all_tasks - covered

    while uncovered:
        best_c = None
        best_cost = float("inf")
        for c in candidates:
            if c["courier_id"] in used:
                continue
            tids = c["task_ids"]
            if not any(t in uncovered for t in tids):
                continue
            if any(t in covered for t in tids):
                continue
            n = len(tids)
            w = c["willingness"]
            mc = w * c["score"] + (1 - w) * penalty * n
            if mc < best_cost:
                best_cost = mc
                best_c = c
        if best_c is None:
            break
        assignments.append((best_c["task_key"], [best_c["courier_id"]]))
        used.add(best_c["courier_id"])
        for t in best_c["task_ids"]:
            covered.add(t)
            uncovered.discard(t)


def _alns(initial_assignments, candidates, all_tasks, all_couriers,
          cand_map, taskkey_to_cands, penalty, deadline):
    """ALNS 元启发式（destroy + repair + 模拟退火接受准则）"""
    import math
    import random
    import time

    rng = random.Random(20260520)

    current = [(tk, list(cids)) for tk, cids in initial_assignments]
    current_cost = _m5_total(current, cand_map, all_tasks, penalty)
    best = [(tk, list(cids)) for tk, cids in current]
    best_cost = current_cost

    initial_covered = set()
    for tk, cids in current:
        if cids:
            c = cand_map.get((tk, cids[0]))
            if c:
                for t in c["task_ids"]:
                    initial_covered.add(t)
    initial_cov_count = len(initial_covered)

    def _cov_count(asgn):
        cov = set()
        for tk, cids in asgn:
            if cids:
                c = cand_map.get((tk, cids[0]))
                if c:
                    for t in c["task_ids"]:
                        cov.add(t)
        return len(cov)

    T = max(5.0, best_cost * 0.025)
    T_min = 0.5
    cooling = 0.997
    n_assignments = len(current)
    k_min = max(3, n_assignments // 8)
    k_max = max(k_min + 1, n_assignments // 3)

    iter_count = 0
    while time.time() < deadline - 0.2:
        iter_count += 1
        cand = [(tk, list(cids)) for tk, cids in current]
        k = rng.randint(k_min, k_max)
        if rng.random() < 0.5:
            _destroy_random(cand, k, rng)
        else:
            _destroy_worst(cand, k, cand_map, penalty)
        _repair_greedy(cand, candidates, all_tasks, cand_map, penalty)
        # v10: repair后添加backup
        used_in_cand = set()
        for tk, cids in cand:
            for cid in cids:
                used_in_cand.add(cid)
        free_in_cand = set(all_couriers) - used_in_cand
        cand = _add_backups_for_repair(cand, free_in_cand, taskkey_to_cands, cand_map, penalty)
        cand = _inter_swap(cand, all_couriers, taskkey_to_cands, cand_map, penalty)
        cand_cost = _m5_total(cand, cand_map, all_tasks, penalty)
        cand_cov = _cov_count(cand)
        delta = cand_cost - current_cost
        if cand_cov < initial_cov_count:
            continue
        accept = False
        if delta < 0:
            accept = True
        elif T > 0 and rng.random() < math.exp(-delta / T):
            accept = True
        if accept:
            current = cand
            current_cost = cand_cost
            if cand_cost < best_cost - 1e-6:
                best = [(tk, list(cids)) for tk, cids in cand]
                best_cost = cand_cost
        T *= cooling
        if T < T_min:
            T = max(2.0, best_cost * 0.015)

    return best


def _add_backups_for_repair(assignments, free_couriers, taskkey_to_cands, cand_map, penalty):
    """repair后为新增分派添加backup（轻量版，只添加一次）"""
    used_couriers = set()
    for tk, cids in assignments:
        for cid in cids:
            used_couriers.add(cid)

    for idx, (tk, cids) in enumerate(assignments):
        if len(cids) > 1:
            continue  # 已有backup
        current_cost = _slot_cost(tk, cids, cand_map, penalty)
        best_gain = 1e-6
        best_cid = None
        for c in taskkey_to_cands[tk]:
            cid = c["courier_id"]
            if cid in used_couriers:
                continue
            new_cids = cids + [cid]
            new_cost = _slot_cost(tk, new_cids, cand_map, penalty)
            gain = current_cost - new_cost
            if gain > best_gain:
                best_gain = gain
                best_cid = cid
        if best_cid is not None:
            cids.append(best_cid)
            used_couriers.add(best_cid)

    return assignments


def _expected_cost(assignments, cand_map, penalty):
    """估算总期望代价（兼容旧接口）"""
    total = 0.0
    for tk, cids in assignments:
        if not cids:
            continue
        total += _slot_cost(tk, cids, cand_map, penalty)
    return total


def _parse_input(input_text: str) -> list:
    """解析输入数据"""
    import re
    text = input_text.lstrip("\ufeff")
    lines = text.strip().splitlines()
    if not lines:
        return []
    first = lines[0].lstrip("\ufeff").strip()
    start = 1 if first.startswith("task_id_list") else 0

    sample = lines[start] if len(lines) > start else first
    if "\t" in sample:
        splitter = lambda s: s.split("\t")
    elif re.search(r"\s{2,}", sample):
        splitter = lambda s: re.split(r"\s{2,}", s)
    elif "," in sample and sample.count(",") >= 3:
        def splitter(s):
            return [p.strip() for p in s.rsplit(",", 3)]
    else:
        splitter = lambda s: s.split()

    candidates = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = splitter(line)
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id = parts[0], parts[1]
        score_str, willingness_str = parts[-2], parts[-1]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue

        task_ids = tuple(t.strip() for t in task_id_list_str.replace(" ", ",").split(",") if t.strip())
        if not task_ids:
            continue
        task_key = task_id_list_str.strip()

        candidates.append({
            "task_key": task_key,
            "task_ids": task_ids,
            "courier_id": courier_id.strip(),
            "score": score,
            "willingness": willingness,
        })

    return candidates
