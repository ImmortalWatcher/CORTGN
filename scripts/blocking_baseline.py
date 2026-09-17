#!/usr/bin/env python3
"""
weibo22 动态阻断：非学习基线（实现轨迹上的子树切除）。

协议（离线、可复现）
----------------
时刻 0 = 根帖时间 t_min。在观察截止 t0 之后，只能对「已经出现且不是根帖」的节点干预（根帖视为已发出；若允许删源，所有策略都会退化成 block_rate=1）。
阻断节点 v：其在 t0 之后的全部转发后代视为不再发生（真实树上的反事实切除）。
不引入独立级联的随机传播；树没有多父节点，这是该数据上的标准近似。

观察窗口
  time:1h / 6h / 24h  — 突发早期（主设定）
  prefix:20 / 50      — 按到达顺序的前 N 帖（不泄漏最终规模）

预算 k ∈ {1, 5, 10}，候选集 = 观察窗口内节点（不含根）。
级联纳入条件：未来节点数 ≥ 20，且观察集非空；实际选取 min(k, |候选|)。

策略
  random          观察集均匀抽样（5 个种子取平均）
  earliest        最早出现的 k 个转发节点（不含根）
  followers       followers_count 最大的 k 个
  outdegree_obs   观察窗口内出度（已发生的直接转发）最大的 k 个
  greedy_oracle   贪心：反复选「未覆盖未来后代」最多的观察节点（事后上界，不是可部署策略）

主指标 block_rate = |被切除的未来节点| / |全部未来节点|
谣言事件（events.txt label=1）单独汇总，非谣言作对照。
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "weibo22" / "posts" / "Weibo"
EVENTS = ROOT / "weibo22" / "events.txt"
OUT_JSON = ROOT / "scripts" / "blocking_baseline_results.json"
OUT_CSV = ROOT / "scripts" / "blocking_baseline_results.csv"

SEED = 3074
K_LIST = (1, 5, 10)
MIN_FUTURE = 20
RANDOM_REPS = 5

TIME_WINDOWS_H = (("time:1h", 1.0), ("time:6h", 6.0), ("time:24h", 24.0))
PREFIX_WINDOWS = (("prefix:20", 20), ("prefix:50", 50))
STRATEGIES = ("random", "earliest", "followers", "outdegree_obs", "greedy_oracle")


def load_labels() -> dict[str, int]:
    raw = EVENTS.read_text(encoding="utf-8", errors="replace")
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"eid:(\d+)\s+label:(-?\d+)", raw)}


def parse_cascade(path: Path) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        return None
    nodes: dict[str, dict] = {}
    children: dict[str, list[str]] = defaultdict(list)
    t_min = None
    root_id = None
    for item in data:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("id") or item.get("mid") or "")
        if not nid:
            continue
        try:
            t = int(item["t"])
        except (KeyError, TypeError, ValueError):
            continue
        parent = item.get("parent")
        parent_s = str(parent) if parent not in (None, "", "null") else None
        followers = item.get("followers_count") or 0
        try:
            followers = int(followers)
        except (TypeError, ValueError):
            followers = 0
        nodes[nid] = {"t": t, "parent": parent_s, "followers": followers}
        if parent_s:
            children[parent_s].append(nid)
        if t_min is None or t < t_min:
            t_min = t
            root_id = nid
        if parent_s is None:
            root_id = nid
    if not nodes or t_min is None or root_id is None:
        return None
    return {"id": path.stem, "root": root_id, "t_min": t_min, "nodes": nodes, "children": children}


def observed_future(cas: dict, window: str):
    nodes = cas["nodes"]
    t_min = cas["t_min"]
    items = sorted(nodes.items(), key=lambda kv: (kv[1]["t"], kv[0]))
    if window.startswith("time:"):
        hours = float(window.split(":", 1)[1].replace("h", ""))
        cut = t_min + hours * 3600.0
        obs = [nid for nid, inf in items if inf["t"] <= cut]
        fut = [nid for nid, inf in items if inf["t"] > cut]
    else:
        n = int(window.split(":", 1)[1])
        obs = [nid for nid, _ in items[:n]]
        fut = [nid for nid, _ in items[n:]]
    return obs, fut


def outdegree_obs(cas: dict, obs_set: set[str]) -> dict[str, int]:
    deg = {nid: 0 for nid in obs_set}
    for nid in obs_set:
        p = cas["nodes"][nid]["parent"]
        if p in obs_set:
            deg[p] = deg.get(p, 0) + 1
    return deg


def future_in_subtree(cas: dict, future_set: set[str]) -> dict[str, int]:
    """每个节点子树内的未来节点数（含自身若在未来）。一次后序遍历。"""
    children = cas["children"]
    root = cas["root"]
    count: dict[str, int] = {}
    stack: list[tuple[str, bool]] = [(root, False)]
    seen = {root}
    while stack:
        v, done = stack.pop()
        if done:
            s = 1 if v in future_set else 0
            for c in children.get(v, ()):
                s += count.get(c, 0)
            count[v] = s
            continue
        stack.append((v, True))
        for c in children.get(v, ()):
            if c not in seen:
                seen.add(c)
                stack.append((c, False))
    return count


def mark_subtree(start: str, children: dict[str, list[str]], acc: set[str]) -> None:
    stack = [start]
    acc.add(start)
    while stack:
        v = stack.pop()
        for c in children.get(v, ()):
            if c not in acc:
                acc.add(c)
                stack.append(c)


def greedy_oracle(cas: dict, obs: list[str], fut: list[str], k: int) -> list[str]:
    future_set = set(fut)
    gain = future_in_subtree(cas, future_set)
    parent = {nid: inf["parent"] for nid, inf in cas["nodes"].items()}
    children = cas["children"]
    remaining = set(obs)
    chosen: list[str] = []
    for _ in range(min(k, len(obs))):
        best = None
        best_g = -1
        for n in remaining:
            g = gain.get(n, 0)
            if g > best_g or (g == best_g and (best is None or n < best)):
                best_g = g
                best = n
        if best is None or best_g <= 0:
            break
        chosen.append(best)
        g = best_g
        sub: set[str] = set()
        mark_subtree(best, children, sub)
        remaining -= sub
        p = parent.get(best)
        while p:
            gain[p] = max(0, gain.get(p, 0) - g)
            p = parent.get(p)
    return chosen


def pick(strategy: str, cas: dict, obs: list[str], fut: list[str], k: int, rng: random.Random) -> list[str]:
    k = min(k, len(obs))
    if k <= 0:
        return []
    obs_set = set(obs)
    if strategy == "random":
        return rng.sample(obs, k)
    if strategy == "earliest":
        return sorted(obs, key=lambda n: (cas["nodes"][n]["t"], n))[:k]
    if strategy == "followers":
        return sorted(obs, key=lambda n: (-cas["nodes"][n]["followers"], n))[:k]
    if strategy == "outdegree_obs":
        deg = outdegree_obs(cas, obs_set)
        return sorted(obs, key=lambda n: (-deg.get(n, 0), n))[:k]
    if strategy == "greedy_oracle":
        return greedy_oracle(cas, obs, fut, k)
    raise ValueError(strategy)


def covered_future(cas: dict, selected: list[str], fut: list[str]) -> int:
    future_set = set(fut)
    covered: set[str] = set()
    for v in selected:
        mark_subtree(v, cas["children"], covered)
    return sum(1 for n in covered if n in future_set)


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    m = len(ys) // 2
    if len(ys) % 2:
        return float(ys[m])
    return (ys[m - 1] + ys[m]) / 2.0


def main() -> None:
    labels = load_labels()
    files = sorted(POSTS_DIR.glob("*.json"))
    windows = [w for w, _ in TIME_WINDOWS_H] + [w for w, _ in PREFIX_WINDOWS]
    # results[window][k][strategy] -> list of block_rate, split by rumor
    store = {
        w: {
            k: {s: {"all": [], "rumor": [], "nonrumor": []} for s in STRATEGIES}
            for k in K_LIST
        }
        for w in windows
    }
    eligible = {w: {"all": 0, "rumor": 0, "nonrumor": 0, "obs": [], "fut": []} for w in windows}

    for i, fp in enumerate(files):
        cas = parse_cascade(fp)
        if cas is None:
            continue
        lab = labels.get(cas["id"], labels.get(fp.stem))
        is_rumor = lab == 1
        split = "rumor" if is_rumor else "nonrumor" if lab == 0 else None

        for wname, _ in TIME_WINDOWS_H + tuple((p, None) for p, _ in PREFIX_WINDOWS):
            obs_all, fut = observed_future(cas, wname)
            obs = [n for n in obs_all if n != cas["root"]]
            if len(fut) < MIN_FUTURE or not obs:
                continue
            eligible[wname]["all"] += 1
            eligible[wname]["obs"].append(len(obs))
            eligible[wname]["fut"].append(len(fut))
            if split:
                eligible[wname][split] += 1

            for k in K_LIST:
                for strat in STRATEGIES:
                    if strat == "random":
                        rates = []
                        for r in range(RANDOM_REPS):
                            rng = random.Random(SEED + r * 17 + k)
                            sel = pick(strat, cas, obs, fut, k, rng)
                            rates.append(covered_future(cas, sel, fut) / len(fut))
                        rate = mean(rates)
                    else:
                        sel = pick(strat, cas, obs, fut, k, random.Random(SEED))
                        rate = covered_future(cas, sel, fut) / len(fut)
                    store[wname][k][strat]["all"].append(rate)
                    if split:
                        store[wname][k][strat][split].append(rate)

        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    summary = {"protocol": {
        "min_future": MIN_FUTURE,
        "k": list(K_LIST),
        "windows": windows,
        "strategies": list(STRATEGIES),
        "seed": SEED,
        "metric": "block_rate = covered_future_nodes / future_nodes",
        "action_space": "observed nodes excluding the cascade root",
    }, "windows": {}}
    csv_rows = ["window,k,strategy,split,n,mean_block_rate,median_block_rate"]

    for w in windows:
        el = eligible[w]
        summary["windows"][w] = {
            "n_eligible": el["all"],
            "n_rumor": el["rumor"],
            "n_nonrumor": el["nonrumor"],
            "mean_observed": mean(el["obs"]),
            "mean_future": mean(el["fut"]),
            "median_observed": median(el["obs"]),
            "median_future": median(el["fut"]),
            "k": {},
        }
        for k in K_LIST:
            summary["windows"][w]["k"][str(k)] = {}
            for strat in STRATEGIES:
                summary["windows"][w]["k"][str(k)][strat] = {}
                for split in ("all", "rumor", "nonrumor"):
                    xs = store[w][k][strat][split]
                    cell = {
                        "n": len(xs),
                        "mean": round(mean(xs), 4) if xs else None,
                        "median": round(median(xs), 4) if xs else None,
                    }
                    summary["windows"][w]["k"][str(k)][strat][split] = cell
                    if cell["mean"] is not None:
                        csv_rows.append(
                            f"{w},{k},{strat},{split},{cell['n']},{cell['mean']},{cell['median']}"
                        )

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_CSV.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    print("Wrote", OUT_JSON)
    print("Wrote", OUT_CSV)
    # compact print rumor, k=5
    print("\n=== rumor only, k=5, mean block_rate ===", flush=True)
    for w in windows:
        print(w, {s: summary["windows"][w]["k"]["5"][s]["rumor"] for s in STRATEGIES})


if __name__ == "__main__":
    main()
