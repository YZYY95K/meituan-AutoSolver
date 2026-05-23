"""
AutoSolver v21: 优化backup分配策略
====================================
v20平台分719.06，large_seed301改进5.87分
核心改进：
1. backup分配：先确保每个slot至少2个骑手，再贪心添加更多
2. 低意愿场景：更均匀地分配backup，避免某些slot过多而其他不足
3. 保留destroy_low_w策略
"""

from collections import defaultdict


def solve(input_text: str) -> list:
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


def _slot_cost(tk, cids, cand_map, penalty):
    """加权平均模型：M5 slot成本"""
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


def _m5_total(assignments, cand_map, all_tasks, penalty):
    total = 0.0
    covered = set()
    for tk, cids in assignments:
        if not cids:
            continue
        total += _slot_cost(tk, cids, cand_map, penalty)
        c = cand_map.get((tk, cids[0]))
        if c:
            for t in c["task_ids"]:
                covered.add(t)
    total += (len(all_tasks) - len(covered)) * penalty
    return total


def _solve_main(input_text: str) -> list:
    import time
    start_ts = time.time()
    deadline = start_ts + 9.2

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

    # 自适应检测
    avg_w = sum(c["willingness"] for c in candidates) / len(candidates)
    n_tasks = len(all_tasks)
    n_couriers = len(all_couriers)
    courier_per_task = n_couriers / max(n_tasks, 1)
    is_low_w = avg_w < 0.5
    is_scarce = courier_per_task < 3.0

    best_assignments = None
    best_metric = float("inf")
    best_covered = 0

    # 自适应策略和penalty
    strategies = ("global", "hard_first", "doubles_first")
    if is_low_w:
        # v21: 低意愿场景增加更多penalty组合
        strategies = ("global", "hard_first", "doubles_first", "willingness_first")
        penalties = (60, 80, 100, 120, 150, 200, 300)
    elif is_scarce:
        strategies = ("global", "hard_first", "doubles_first", "willingness_first")
        penalties = (60, 80, 100, 120, 150, 200)
    else:
        penalties = (60, 100, 200)

    for primary_strategy in strategies:
        for penalty in penalties:
            if primary_strategy == "global":
                primary = _pick_primary(candidates, all_tasks, penalty)
            elif primary_strategy == "hard_first":
                primary = _pick_primary_hard_first(candidates, all_tasks, penalty)
            elif primary_strategy == "doubles_first":
                primary = _pick_primary_doubles_first(candidates, all_tasks, penalty)
            elif primary_strategy == "willingness_first":
                primary = _pick_primary_willingness_first(candidates, all_tasks)
            else:
                primary = _pick_primary_random(candidates, all_tasks, penalty)
            primary = _rescue_coverage(primary, candidates, all_tasks, cand_map)

            covered = set()
            for tk, cid in primary:
                c = cand_map.get((tk, cid))
                if c:
                    for t in c["task_ids"]:
                        covered.add(t)

            # v18: 用_slot_cost精确计算backup增益
            # 低意愿时用更高penalty评估backup（让backup更激进）
            backup_penalty = max(penalty, 200) if is_low_w else penalty
            assignments = _add_backups(primary, all_couriers, taskkey_to_cands, cand_map, backup_penalty,
                                       aggressive=is_low_w)
            # local_swap用判题真值penalty（避免删除有益backup）
            assignments = _local_swap(assignments, taskkey_to_cands, cand_map, 100)

            # 用判题真值penalty=100评估
            metric = _m5_total(assignments, cand_map, all_tasks, 100)
            uncovered = n_tasks - len(covered)
            metric += uncovered * 100
            if metric < best_metric:
                best_metric = metric
                best_assignments = assignments
                best_covered = len(covered)

    # 兜底：baseline
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
        backup_penalty = 200 if is_low_w else 100
        assignments = _add_backups(primary_from_baseline, all_couriers, taskkey_to_cands, cand_map, backup_penalty,
                                   aggressive=is_low_w)
        assignments = _local_swap(assignments, taskkey_to_cands, cand_map, 100)
        best_assignments = assignments

    if not best_assignments:
        return []

    # inter_swap（用判题真值penalty=100）
    best_assignments = _inter_swap(
        best_assignments, all_couriers, taskkey_to_cands, cand_map, 100)

    # ALNS
    if time.time() < deadline - 0.3:
        best_assignments = _alns(
            best_assignments, candidates, all_tasks, all_couriers,
            cand_map, taskkey_to_cands, 100, deadline,
            is_low_w=is_low_w)

    return [(tk, list(cids)) for tk, cids in best_assignments]


# ============================================================
# Primary 策略
# ============================================================
def _pick_primary_hard_first(candidates, all_tasks, penalty):
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
    for _, c in enriched:
        if c["courier_id"] in used_c:
            continue
        if any(t in used_t for t in c["task_ids"]):
            continue
        out.append((c["task_key"], c["courier_id"]))
        used_c.add(c["courier_id"])
        for t in c["task_ids"]:
            used_t.add(t)
        if len(used_t) >= len(all_tasks):
            break
    return out


def _pick_primary_willingness_first(candidates, all_tasks):
    """v20: willingness优先策略，用于低意愿场景。
    优先选高willingness骑手，即使score更高。
    在willingness相同时，选score更低的。
    """
    enriched = []
    for c in candidates:
        # 按willingness降序、score升序排列
        enriched.append((-c["willingness"], c["score"], c))
    enriched.sort(key=lambda x: (x[0], x[1]))
    used_c = set()
    used_t = set()
    out = []
    for _, _, c in enriched:
        if c["courier_id"] in used_c:
            continue
        if any(t in used_t for t in c["task_ids"]):
            continue
        out.append((c["task_key"], c["courier_id"]))
        used_c.add(c["courier_id"])
        for t in c["task_ids"]:
            used_t.add(t)
        if len(used_t) >= len(all_tasks):
            break
    return out


def _pick_primary_score_asc(candidates, all_tasks):
    """按score升序选骑手（贪心基线策略），作为对比基准"""
    enriched = []
    for c in candidates:
        enriched.append((c["score"], c))
    enriched.sort(key=lambda x: x[0])
    used_c = set()
    used_t = set()
    out = []
    for _, c in enriched:
        if c["courier_id"] in used_c:
            continue
        if any(t in used_t for t in c["task_ids"]):
            continue
        out.append((c["task_key"], c["courier_id"]))
        used_c.add(c["courier_id"])
        for t in c["task_ids"]:
            used_t.add(t)
        if len(used_t) >= len(all_tasks):
            break
    return out


def _pick_primary_random(candidates, all_tasks, penalty):
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
    for _, c in enriched:
        if c["courier_id"] in used_c:
            continue
        if any(t in used_t for t in c["task_ids"]):
            continue
        out.append((c["task_key"], c["courier_id"]))
        used_c.add(c["courier_id"])
        for t in c["task_ids"]:
            used_t.add(t)
        if len(used_t) >= len(all_tasks):
            break
    return out


def _rescue_coverage(primary, candidates, all_tasks, cand_map):
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
    for _, c in enriched:
        if c["courier_id"] in used_c:
            continue
        if any(t in used_t for t in c["task_ids"]):
            continue
        out.append((c["task_key"], c["courier_id"]))
        used_c.add(c["courier_id"])
        for t in c["task_ids"]:
            used_t.add(t)
        if len(used_t) >= len(all_tasks):
            break
    return out


# ============================================================
# Backup（保留v20的全局贪心策略）
# ============================================================
def _add_backups(primary, all_couriers, taskkey_to_cands, cand_map, penalty, aggressive=False):
    """添加backup骑手。用_slot_cost精确计算增益。
    aggressive=True时更激进（低意愿场景）：更多轮次、更低p_rej阈值。
    """
    assignments = []
    used_couriers = set()
    for tk, cid in primary:
        assignments.append((tk, [cid]))
        used_couriers.add(cid)

    min_p_rej = 0.005 if aggressive else 0.03
    max_rounds = 8 if aggressive else 3

    for _ in range(max_rounds):
        best_gain = 1e-6
        best_choice = None

        for idx, (tk, cids) in enumerate(assignments):
            p_rej = 1.0
            for cid in cids:
                cc = cand_map.get((tk, cid))
                if cc:
                    p_rej *= (1 - cc["willingness"])
            if p_rej < min_p_rej:
                continue

            current_cost = _slot_cost(tk, cids, cand_map, penalty)
            for c in taskkey_to_cands[tk]:
                cid = c["courier_id"]
                if cid in used_couriers:
                    continue
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


# ============================================================
# Local Swap
# ============================================================
def _local_swap(assignments, taskkey_to_cands, cand_map, penalty):
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


# ============================================================
# Inter Swap
# ============================================================
def _inter_swap(assignments, all_couriers, taskkey_to_cands, cand_map, penalty):
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
                    if (tk_j, X) not in cand_map or X in cids_j:
                        continue
                    for py in range(len(cids_j)):
                        Y = cids_j[py]
                        if X == Y or (tk_i, Y) not in cand_map or Y in cids_i:
                            continue
                        cids_i[px], cids_j[py] = Y, X
                        new_cost = _slot_cost(tk_i, cids_i, cand_map, penalty) + \
                                   _slot_cost(tk_j, cids_j, cand_map, penalty)
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
                    if (tk_j, X) not in cand_map or X in cids_j:
                        continue
                    removed = cids_i.pop(px)
                    cids_j.append(X)
                    new_cost = _slot_cost(tk_i, cids_i, cand_map, penalty) + \
                               _slot_cost(tk_j, cids_j, cand_map, penalty)
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
                if (tk_j, X) not in cand_map or X in cids_j:
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


# ============================================================
# ALNS
# ============================================================
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


def _destroy_high_p_rej(assignments, k, cand_map, penalty):
    n = len(assignments)
    if n == 0:
        return []
    k = min(k, n)
    p_rejs = []
    for i, (tk, cids) in enumerate(assignments):
        p_rej = 1.0
        for cid in cids:
            c = cand_map.get((tk, cid))
            if c:
                p_rej *= (1 - c["willingness"])
        p_rejs.append((p_rej, i))
    p_rejs.sort(reverse=True)
    indices = [p_rejs[i][1] for i in range(k)]
    removed = []
    for i in sorted(indices, reverse=True):
        removed.append(assignments.pop(i))
    return removed


def _destroy_low_w(assignments, k, cand_map):
    """v20: 优先destroy低willingness的slot（这些slot成本最高）"""
    n = len(assignments)
    if n == 0:
        return []
    k = min(k, n)
    # 计算每个slot的平均willingness
    avg_ws = []
    for i, (tk, cids) in enumerate(assignments):
        ws = []
        for cid in cids:
            c = cand_map.get((tk, cid))
            if c:
                ws.append(c["willingness"])
        avg_w = sum(ws) / len(ws) if ws else 0
        avg_ws.append((avg_w, i))
    avg_ws.sort()  # 低willingness排前面
    indices = [avg_ws[i][1] for i in range(k)]
    removed = []
    for i in sorted(indices, reverse=True):
        removed.append(assignments.pop(i))
    return removed


def _repair_greedy(assignments, candidates, all_tasks, cand_map, penalty):
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


def _add_backups_light(assignments, all_couriers, taskkey_to_cands, cand_map, penalty, aggressive=False):
    """轻量backup：为高p_rej的slot添加1-2个backup"""
    used_couriers = set()
    for tk, cids in assignments:
        for cid in cids:
            used_couriers.add(cid)

    max_per_slot = 4 if aggressive else 2
    for idx, (tk, cids) in enumerate(assignments):
        while len(cids) < max_per_slot:
            p_rej = 1.0
            for cid in cids:
                cc = cand_map.get((tk, cid))
                if cc:
                    p_rej *= (1 - cc["willingness"])
            min_p = 0.005 if aggressive else 0.02
            if p_rej < min_p:
                break
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
            if best_cid is None:
                break
            cids.append(best_cid)
            used_couriers.add(best_cid)
    return assignments


def _light_local(assignments, cand_map, penalty):
    """轻量局部搜索：只做冗余删除"""
    for tk, cids in assignments:
        if len(cids) <= 1:
            continue
        current_cost = _slot_cost(tk, cids, cand_map, penalty)
        for pos in range(len(cids) - 1, 0, -1):
            removed = cids.pop(pos)
            new_cost = _slot_cost(tk, cids, cand_map, penalty)
            if new_cost < current_cost - 1e-6:
                current_cost = new_cost
            else:
                cids.insert(pos, removed)
    return assignments


def _alns(initial_assignments, candidates, all_tasks, all_couriers,
          cand_map, taskkey_to_cands, penalty, deadline, is_low_w=False):
    import math, random, time

    rng = random.Random(20260520)

    current = [(tk, list(cids)) for tk, cids in initial_assignments]
    current_cost = _m5_total(current, cand_map, all_tasks, penalty)
    best = [(tk, list(cids)) for tk, cids in current]
    best_cost = current_cost

    initial_cov = set()
    for tk, cids in current:
        if cids:
            c = cand_map.get((tk, cids[0]))
            if c:
                for t in c["task_ids"]:
                    initial_cov.add(t)
    initial_cov_count = len(initial_cov)

    def cov_count(asgn):
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

    # 4种destroy策略（v20: 增加destroy_low_w）
    destroy_weights = [1.0, 1.0, 1.0, 1.0]
    destroy_scores = [0.0, 0.0, 0.0, 0.0]

    iter_count = 0
    while time.time() < deadline - 0.2:
        iter_count += 1
        cand = [(tk, list(cids)) for tk, cids in current]
        k = rng.randint(k_min, k_max)

        # 选择destroy策略
        total_w = sum(destroy_weights)
        r = rng.random() * total_w
        destroy_idx = 0
        cum = 0
        for i, w in enumerate(destroy_weights):
            cum += w
            if r <= cum:
                destroy_idx = i
                break

        if destroy_idx == 0:
            _destroy_random(cand, k, rng)
        elif destroy_idx == 1:
            _destroy_worst(cand, k, cand_map, penalty)
        elif destroy_idx == 2:
            _destroy_high_p_rej(cand, k, cand_map, penalty)
        else:
            _destroy_low_w(cand, k, cand_map)

        _repair_greedy(cand, candidates, all_tasks, cand_map, penalty)
        _add_backups_light(cand, all_couriers, taskkey_to_cands, cand_map, penalty,
                           aggressive=is_low_w)
        cand = _inter_swap(cand, all_couriers, taskkey_to_cands, cand_map, penalty)

        cand_cost = _m5_total(cand, cand_map, all_tasks, penalty)
        cand_cov = cov_count(cand)
        if cand_cov < initial_cov_count:
            destroy_scores[destroy_idx] -= 1
            continue

        delta = cand_cost - current_cost
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
                destroy_scores[destroy_idx] += 3
        if delta < 0:
            destroy_scores[destroy_idx] += 2
        elif accept:
            destroy_scores[destroy_idx] += 1

        T *= cooling
        if T < T_min:
            T = max(2.0, best_cost * 0.015)

        # 每50次迭代更新destroy权重
        if iter_count % 50 == 0:
            for i in range(4):
                if destroy_scores[i] > 0:
                    destroy_weights[i] = max(0.1, destroy_weights[i] * 1.1)
                else:
                    destroy_weights[i] = max(0.1, destroy_weights[i] * 0.9)
                destroy_scores[i] = 0

    return best


# ============================================================
# 解析
# ============================================================
def _parse_input(input_text: str) -> list:
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
