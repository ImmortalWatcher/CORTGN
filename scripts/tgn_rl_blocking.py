#!/usr/bin/env python3
"""
甲：TGN 记忆 + REINFORCE 序贯阻断。

一条转发到来时，用 GRU 更新这条边两端的记忆，再对「当前还能选的人」
加上「这一步谁也不拦」一起做 softmax。名额用完或窗口走完即停。
上游被选中后，其观察窗口内的下游不再进入动作集合。

奖励在树结束时是真实 block_rate。训练另加潜力塑形：每拦一个人，
把新盖住的未来节点比例乘上系数 alpha。alpha 从第 1 个 epoch 的 1
线性降到最后一轮的 0。因为被选中的点在观察树上互不为祖先，这些边际
加总等于 block_rate，所以训练回报 = block_rate * (1 + alpha)。
塑形只进训练奖励。验证和测试的 alpha 固定为 0。

不读取 scripts/tgn_blocking.pt，也不改一次性回归那条训练。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking_baseline import (  # noqa: E402
    MIN_FUTURE,
    POSTS_DIR,
    SEED,
    covered_future,
    load_labels,
    observed_future,
    parse_cascade,
    pick,
)
from tgn_blocking import node_depth, stratified_split  # noqa: E402

OUT_JSON = Path(__file__).resolve().parent / "tgn_rl_blocking_results.json"
CKPT = Path(__file__).resolve().parent / "tgn_rl_blocking.pt"
TRAIN_LOG = Path(__file__).resolve().parent / "tgn_rl_blocking_train.log"
MEM_FEAT_DIM = 3
POLICY_FEAT_DIM = 8
BPTT_CHUNK = 1536
ORACLE_1H_K5 = 0.1601
OUTDEG_1H_K5 = 0.1119
SUCCESS_1H_K5 = 0.122


def log(msg: str) -> None:
    print(msg, flush=True)
    with TRAIN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


class TGNPolicy(nn.Module):
    """与 MiniTGN 相同的边记忆，打分头改成序贯策略。"""

    def __init__(self, mem_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.mem_dim = mem_dim
        self.hidden = hidden
        self.feat_enc = nn.Sequential(nn.Linear(MEM_FEAT_DIM, mem_dim), nn.ReLU())
        self.msg = nn.Linear(mem_dim * 2 + 1, mem_dim)
        self.gru_src = nn.GRUCell(mem_dim, mem_dim)
        self.gru_dst = nn.GRUCell(mem_dim, mem_dim)
        self.scorer = nn.Sequential(
            nn.Linear(mem_dim + POLICY_FEAT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.skip_mem = nn.Parameter(torch.zeros(mem_dim))
        nn.init.normal_(self.skip_mem, std=0.02)

    def edge_update(self, hs: torch.Tensor, hd: torch.Tensor, log_dt: torch.Tensor):
        msg = torch.tanh(self.msg(torch.cat([hs, hd, log_dt], dim=0)))
        ns = self.gru_src(msg.unsqueeze(0), hs.unsqueeze(0)).squeeze(0)
        nd = self.gru_dst(msg.unsqueeze(0), hd.unsqueeze(0)).squeeze(0)
        return ns, nd

    def score(self, mem: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        return self.scorer(torch.cat([mem, extra], dim=-1)).squeeze(-1)


@dataclass
class Episode:
    cid: str
    label: int
    cas: dict
    fut: list[str]
    nids: list[str]
    parent: list[int]
    obs_children: list[list[int]]
    arrivals: list[int]
    edge_src: list[int]
    log_dt: list[float]
    feat: torch.Tensor
    hours: list[float]
    depth: list[float]
    log_followers: list[float]


@dataclass
class Trajectory:
    log_probs: list[torch.Tensor] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)
    block_rate: float = 0.0
    selected: list[str] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    choices: list[int] = field(default_factory=list)


def reward_to_go(step_rewards: list[float], terminal: float) -> list[float]:
    goto = [0.0] * len(step_rewards)
    acc = terminal
    for t in range(len(step_rewards) - 1, -1, -1):
        acc += step_rewards[t]
        goto[t] = acc
    return goto


def mark_descendants(start: int, children: list[list[int]], acc: set[int]) -> None:
    stack = list(children[start])
    while stack:
        v = stack.pop()
        if v in acc:
            continue
        acc.add(v)
        stack.extend(children[v])


def ancestor_selected(idx: int, parent: list[int], selected: set[int]) -> bool:
    seen: set[int] = set()
    p = parent[idx]
    while p >= 0 and p not in seen:
        if p in selected:
            return True
        seen.add(p)
        p = parent[p]
    return False


def newly_covered(cas: dict, start: str, fut_set: set[str], covered: set[str]) -> int:
    stack = [start]
    seen: set[str] = set()
    got = 0
    children = cas["children"]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        if v in fut_set and v not in covered:
            covered.add(v)
            got += 1
        stack.extend(children.get(v, ()))
    return got


def load_cascades() -> dict[str, dict]:
    out = {}
    files = sorted(POSTS_DIR.glob("*.json"))
    for i, fp in enumerate(files):
        cas = parse_cascade(fp)
        if cas is not None:
            out[cas["id"]] = cas
        if (i + 1) % 500 == 0:
            log(f"  load {i + 1}/{len(files)}")
    return out


def build_episode(cas: dict, label: int, window: str) -> Episode | None:
    obs, fut = observed_future(cas, window)
    root = cas["root"]
    if len(fut) < MIN_FUTURE or not any(n != root for n in obs):
        return None
    obs_set = set(obs)
    order = sorted(obs, key=lambda n: (cas["nodes"][n]["t"], n))
    index = {nid: i for i, nid in enumerate(order)}
    n = len(order)
    parent = [-1] * n
    obs_children: list[list[int]] = [[] for _ in range(n)]
    depth_cache: dict[str, int] = {}
    hours = [0.0] * n
    depth = [0.0] * n
    log_followers = [0.0] * n
    feat = torch.zeros(n, MEM_FEAT_DIM, dtype=torch.float32)
    t0 = cas["t_min"]
    for nid, i in index.items():
        inf = cas["nodes"][nid]
        p = inf["parent"]
        if p in index:
            parent[i] = index[p]
            obs_children[index[p]].append(i)
        h = max(0.0, (inf["t"] - t0) / 3600.0)
        d = float(node_depth(cas, nid, depth_cache))
        lf = math.log1p(inf["followers"])
        hours[i] = h
        depth[i] = d
        log_followers[i] = lf
        feat[i, 0] = lf
        feat[i, 1] = math.log1p(h)
        feat[i, 2] = d / 16.0
    arrivals = [index[nid] for nid in order if nid != root]
    edge_src = []
    log_dt = []
    for dst in arrivals:
        src = parent[dst]
        edge_src.append(src)
        if src >= 0:
            dt = max(0, cas["nodes"][order[dst]]["t"] - cas["nodes"][order[src]]["t"])
            log_dt.append(math.log1p(dt))
        else:
            log_dt.append(0.0)
    return Episode(
        cid=cas["id"],
        label=label,
        cas=cas,
        fut=fut,
        nids=order,
        parent=parent,
        obs_children=obs_children,
        arrivals=arrivals,
        edge_src=edge_src,
        log_dt=log_dt,
        feat=feat,
        hours=hours,
        depth=depth,
        log_followers=log_followers,
    )


def build_episodes(cas_map: dict[str, dict], ids: list[str], labels: dict[str, int], window: str) -> list[Episode]:
    out = []
    for n_done, cid in enumerate(ids, start=1):
        ep = build_episode(cas_map[cid], labels.get(cid, -1), window)
        if ep is not None:
            out.append(ep)
        if n_done % 500 == 0:
            log(f"  episodes {n_done}/{len(ids)} kept {len(out)}")
    return out


def run_episode(
    model: TGNPolicy,
    ep: Episode,
    *,
    k: int,
    alpha: float,
    train: bool,
    use_outdegree: bool,
    ancestor_feature: bool,
    skip_blocked: bool,
    device: torch.device,
    capture: list[torch.Tensor] | None = None,
    check_mask: bool = False,
    force_skip: bool = False,
    allow_skip: bool = True,
    force_fill: bool = False,
    choice_out: list[int] | None = None,
    fixed_choices: list[int] | None = None,
    advantages: list[float] | None = None,
    loss_scale: float = 1.0,
    bptt_chunk: int = 0,
    build_graph: bool | None = None,
) -> Trajectory:
    if build_graph is None:
        build_graph = train
    grad_ctx = torch.enable_grad() if build_graph else torch.no_grad()
    traj = Trajectory()
    with grad_ctx:
        feat = ep.feat.to(device)
        init = model.feat_enc(feat)
        mems = [init[i] for i in range(init.shape[0])]
        log_dt_all = torch.tensor(ep.log_dt, dtype=torch.float32, device=device)
        blocked: set[int] = set()
        selected: set[int] = set()
        covered: set[str] = set()
        fut_set = set(ep.fut) if alpha > 0.0 else set()
        deg = [0] * len(ep.nids)
        dead = [False] * len(ep.nids)
        pool: list[int] = []
        remaining = k
        nfut = len(ep.fut)
        pending_lp: list[torch.Tensor] = []
        pending_adv: list[float] = []
        made = 0

        def flush() -> None:
            if not pending_lp:
                return
            loss = torch.stack([-lp * a for lp, a in zip(pending_lp, pending_adv)]).sum() / loss_scale
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite policy gradient")
            loss.backward()
            pending_lp.clear()
            pending_adv.clear()
            for i in range(len(mems)):
                mems[i] = mems[i].detach()

        for step_i, dst in enumerate(ep.arrivals):
            if remaining <= 0:
                break
            src = ep.edge_src[step_i]
            if src >= 0:
                deg[src] += 1
                ns, nd = model.edge_update(mems[src], mems[dst], log_dt_all[step_i : step_i + 1])
                mems[src] = ns
                mems[dst] = nd
            if dst in blocked:
                dead[dst] = True
            elif not dead[dst]:
                pool.append(dst)
            legal = [i for i in pool if not dead[i]]
            pool = legal
            if not legal:
                continue
            if check_mask and skip_blocked:
                hit = blocked.intersection(legal)
                if hit:
                    raise RuntimeError(f"blocked descendants are still selectable: {sorted(hit)[:5]}")
            rows = []
            t_now = ep.hours[dst]
            n_sel = float(len(selected))
            for i in legal:
                anc = 1.0 if ancestor_selected(i, ep.parent, selected) else 0.0
                rows.append(
                    (
                        ep.log_followers[i],
                        ep.hours[i],
                        ep.depth[i],
                        math.log1p(deg[i]) if use_outdegree else 0.0,
                        anc if ancestor_feature else 0.0,
                        float(remaining),
                        n_sel,
                        math.exp(-(t_now - ep.hours[i])),
                    )
                )
            mem_rows = torch.stack([mems[i] for i in legal], dim=0)
            extra = torch.tensor(rows, dtype=torch.float32, device=device)
            skip_extra = torch.tensor(
                [[0.0, t_now, 0.0, 0.0, 0.0, float(remaining), n_sel, 0.0]],
                dtype=torch.float32,
                device=device,
            )
            logits = model.score(mem_rows, extra)
            if allow_skip:
                skip_logit = model.score(model.skip_mem.unsqueeze(0), skip_extra)
                all_logits = torch.cat([logits, skip_logit.view(1)], dim=0)
            else:
                all_logits = logits
            if capture is not None:
                capture.append(all_logits.detach().cpu())
            dist = torch.distributions.Categorical(logits=all_logits)
            if fixed_choices is not None:
                if made >= len(fixed_choices):
                    raise RuntimeError("replay produced a decision that was not recorded")
                choice_i = fixed_choices[made]
                choice_t = torch.tensor(choice_i, device=all_logits.device)
                lp = dist.log_prob(choice_t)
                made += 1
                if advantages is not None:
                    pending_lp.append(lp)
                    pending_adv.append(advantages[made - 1])
                    if bptt_chunk > 0 and len(pending_lp) >= bptt_chunk:
                        flush()
                else:
                    traj.log_probs.append(lp)
            elif force_skip:
                choice_i = len(legal)
                if train and build_graph:
                    traj.log_probs.append(dist.log_prob(torch.tensor(choice_i, device=all_logits.device)))
            elif train:
                choice = dist.sample()
                lp = dist.log_prob(choice)
                if build_graph:
                    traj.log_probs.append(lp)
                choice_i = int(choice.item())
            else:
                choice_i = int(torch.argmax(all_logits).item())
            if choice_i < 0 or choice_i > len(legal):
                raise RuntimeError(f"choice {choice_i} outside 0..{len(legal)}")
            if choice_out is not None:
                choice_out.append(choice_i)
            if choice_i == len(legal):
                traj.step_rewards.append(0.0)
                continue
            node_i = legal[choice_i]
            selected.add(node_i)
            dead[node_i] = True
            traj.selected.append(ep.nids[node_i])
            remaining -= 1
            if skip_blocked:
                mark_descendants(node_i, ep.obs_children, blocked)
                for d in blocked:
                    dead[d] = True
            if alpha > 0.0:
                got = newly_covered(ep.cas, ep.nids[node_i], fut_set, covered)
                traj.step_rewards.append(alpha * got / nfut)
            else:
                traj.step_rewards.append(0.0)
        if force_fill and remaining > 0 and ep.arrivals:
            t_end = ep.hours[ep.arrivals[-1]]
            while remaining > 0:
                legal = [i for i in pool if not dead[i]]
                pool = legal
                if not legal:
                    break
                rows = []
                n_sel = float(len(selected))
                for i in legal:
                    anc = 1.0 if ancestor_selected(i, ep.parent, selected) else 0.0
                    rows.append(
                        (
                            ep.log_followers[i],
                            ep.hours[i],
                            ep.depth[i],
                            math.log1p(deg[i]) if use_outdegree else 0.0,
                            anc if ancestor_feature else 0.0,
                            float(remaining),
                            n_sel,
                            math.exp(-(t_end - ep.hours[i])),
                        )
                    )
                mem_rows = torch.stack([mems[i] for i in legal], dim=0)
                extra = torch.tensor(rows, dtype=torch.float32, device=device)
                all_logits = model.score(mem_rows, extra)
                dist = torch.distributions.Categorical(logits=all_logits)
                if fixed_choices is not None:
                    if made >= len(fixed_choices):
                        raise RuntimeError("replay produced a decision that was not recorded")
                    choice_i = fixed_choices[made]
                    lp = dist.log_prob(torch.tensor(choice_i, device=all_logits.device))
                    made += 1
                    if advantages is not None:
                        pending_lp.append(lp)
                        pending_adv.append(advantages[made - 1])
                        if bptt_chunk > 0 and len(pending_lp) >= bptt_chunk:
                            flush()
                    else:
                        traj.log_probs.append(lp)
                elif train:
                    choice = dist.sample()
                    lp = dist.log_prob(choice)
                    if build_graph:
                        traj.log_probs.append(lp)
                    choice_i = int(choice.item())
                else:
                    choice_i = int(torch.argmax(all_logits).item())
                if choice_out is not None:
                    choice_out.append(choice_i)
                if choice_i < 0 or choice_i >= len(legal):
                    raise RuntimeError(f"fill choice {choice_i} outside 0..{len(legal) - 1}")
                node_i = legal[choice_i]
                selected.add(node_i)
                dead[node_i] = True
                traj.selected.append(ep.nids[node_i])
                remaining -= 1
                if skip_blocked:
                    mark_descendants(node_i, ep.obs_children, blocked)
                    for d in blocked:
                        dead[d] = True
                if alpha > 0.0:
                    got = newly_covered(ep.cas, ep.nids[node_i], fut_set, covered)
                    traj.step_rewards.append(alpha * got / nfut)
                else:
                    traj.step_rewards.append(0.0)
        if fixed_choices is not None and made != len(fixed_choices):
            raise RuntimeError(f"replay decisions {made} != recorded {len(fixed_choices)}")
        flush()
        traj.block_rate = covered_future(ep.cas, traj.selected, ep.fut) / nfut
    return traj


def reinforce_loss(batch: list[Trajectory]) -> torch.Tensor:
    ep_returns = []
    ready: list[Trajectory] = []
    for traj in batch:
        if not traj.log_probs:
            continue
        traj.returns = reward_to_go(traj.step_rewards, traj.block_rate)
        if len(traj.returns) != len(traj.log_probs):
            raise RuntimeError("return length does not match decisions")
        ep_returns.append(traj.returns[0])
        ready.append(traj)
    if not ready:
        raise RuntimeError("batch has no decisions")
    baseline = sum(ep_returns) / len(ep_returns)
    terms = []
    for traj in ready:
        adv = [g - baseline for g in traj.returns]
        terms.append(torch.stack([-lp * a for lp, a in zip(traj.log_probs, adv)]).mean())
    return torch.stack(terms).mean()


def rule_block_rate(episodes: list[Episode], strategy: str, k: int) -> float:
    if not episodes:
        return float("nan")
    rng = random.Random(SEED)
    total = 0.0
    for ep in episodes:
        obs = [ep.nids[i] for i in ep.arrivals]
        sel = pick(strategy, ep.cas, obs, ep.fut, k, rng)
        total += covered_future(ep.cas, sel, ep.fut) / len(ep.fut)
    return total / len(episodes)


@torch.no_grad()
def policy_block_rate(
    model: TGNPolicy,
    episodes: list[Episode],
    *,
    k: int,
    use_outdegree: bool,
    ancestor_feature: bool,
    skip_blocked: bool,
    allow_skip: bool,
    force_fill: bool,
    device: torch.device,
) -> float:
    if not episodes:
        return float("nan")
    was = model.training
    model.eval()
    total = 0.0
    for ep in episodes:
        traj = run_episode(
            model,
            ep,
            k=k,
            alpha=0.0,
            train=False,
            use_outdegree=use_outdegree,
            ancestor_feature=ancestor_feature,
            skip_blocked=skip_blocked,
            allow_skip=allow_skip,
            force_fill=force_fill,
            device=device,
        )
        total += traj.block_rate
    if was:
        model.train()
    return total / len(episodes)


def alpha_for_epoch(epoch: int, epochs: int, shaping: bool) -> float:
    if not shaping or epochs <= 1:
        return 0.0
    return max(0.0, (epochs - epoch) / (epochs - 1))


def clone_cas(cas: dict) -> dict:
    return {
        "id": cas["id"],
        "root": cas["root"],
        "t_min": cas["t_min"],
        "nodes": {k: dict(v) for k, v in cas["nodes"].items()},
        "children": {k: list(v) for k, v in cas["children"].items()},
    }


def shuffle_future(cas: dict, window: str, rng: random.Random) -> dict:
    obs, fut = observed_future(cas, window)
    obs_set = set(obs)
    copied = clone_cas(cas)
    pool = list(fut) + [cas["root"]]
    for nid in fut:
        choice = rng.choice(pool)
        if choice == nid:
            choice = cas["root"]
        copied["nodes"][nid]["parent"] = choice
    children: dict[str, list[str]] = {}
    for nid, inf in copied["nodes"].items():
        p = inf["parent"]
        if p:
            children.setdefault(p, []).append(nid)
    copied["children"] = children
    # 观察窗口内的父子边应保持原样；这里只是让断言读起来明确。
    for nid in obs:
        if copied["nodes"][nid]["parent"] != cas["nodes"][nid]["parent"]:
            raise RuntimeError("shuffle touched an observation-window parent")
        if nid in obs_set and nid in fut:
            raise RuntimeError("node is both observed and future")
    return copied


def self_check() -> None:
    children = [[1, 3], [2], [], []]
    blocked: set[int] = set()
    mark_descendants(1, children, blocked)
    if blocked != {2}:
        raise RuntimeError(f"descendant mask expected {{2}}, got {blocked}")
    legal = [i for i in (1, 2, 3) if i not in {1} and i not in blocked]
    if legal != [3]:
        raise RuntimeError(f"legal set expected [3], got {legal}")
    if reward_to_go([0.5], 0.5) != [1.0]:
        raise RuntimeError("reward-to-go terminal+shaping failed")
    if reward_to_go([0.0, 0.0], 0.25) != [0.25, 0.25]:
        raise RuntimeError("reward-to-go skip path failed")
    log("self-check ok")


def leakage_check(model: TGNPolicy, ep: Episode, window: str) -> None:
    obs, fut = observed_future(ep.cas, window)
    if set(ep.nids) != set(obs):
        raise RuntimeError("episode nodes are not exactly the observation window")
    if set(ep.nids) & set(fut):
        raise RuntimeError("future node entered the episode state")
    was = model.training
    model.eval()
    device = torch.device("cpu")
    cpu_model = TGNPolicy(model.mem_dim, model.hidden)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.to(device).eval()
    kwargs = dict(
        k=5,
        alpha=0.0,
        train=False,
        use_outdegree=True,
        ancestor_feature=True,
        skip_blocked=True,
        device=device,
        check_mask=True,
    )
    first: list[torch.Tensor] = []
    run_episode(cpu_model, ep, capture=first, **kwargs)
    shuffled = shuffle_future(ep.cas, window, random.Random(0))
    ep2 = build_episode(shuffled, ep.label, window)
    if ep2 is None:
        raise RuntimeError("shuffled cascade became unevaluable")
    second: list[torch.Tensor] = []
    run_episode(cpu_model, ep2, capture=second, **kwargs)
    if len(first) != len(second):
        raise RuntimeError(f"decision count changed after shuffling the future: {len(first)} vs {len(second)}")
    for i, (a, b) in enumerate(zip(first, second)):
        if a.shape != b.shape or not torch.allclose(a, b, atol=1e-5, rtol=1e-4):
            raise RuntimeError(f"action logits changed at step {i} after shuffling the future")
    if was:
        model.train()
    log(f"leakage-check ok on {ep.cid} ({len(first)} decisions)")


def check_published_baselines(test_rumor: list[Episode], window: str, k: int) -> dict[str, float]:
    outdeg = rule_block_rate(test_rumor, "outdegree_obs", k)
    oracle = rule_block_rate(test_rumor, "greedy_oracle", k)
    log(
        f"baseline replay {window} k={k} n={len(test_rumor)} "
        f"outdegree {outdeg:.4%} oracle {oracle:.4%}"
    )
    if window == "time:1h" and k == 5:
        if len(test_rumor) != 311 or abs(outdeg - OUTDEG_1H_K5) > 5e-5 or abs(oracle - ORACLE_1H_K5) > 5e-5:
            raise RuntimeError(
                "test-set replay does not match the published 311 trees, "
                f"outdegree {OUTDEG_1H_K5:.2%}, oracle {ORACLE_1H_K5:.2%}"
            )
        log("baseline replay matches the published test table")
    return {"outdegree_obs": outdeg, "greedy_oracle": oracle}


def save_ckpt(
    model: TGNPolicy,
    epoch: int,
    val_rate: float,
    args: argparse.Namespace,
    *,
    optimizer: Adam | None = None,
    stall: int = 0,
    history: list | None = None,
    next_epoch: int | None = None,
) -> None:
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": epoch,
        "best_epoch": epoch,
        "val_block_rate": val_rate,
        "args": vars(args),
        "stall": stall,
        "history": history or [],
        "next_epoch": epoch + 1 if next_epoch is None else next_epoch,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, CKPT)


def episode_kwargs(args: argparse.Namespace, device: torch.device) -> dict:
    return {
        "k": args.k,
        "use_outdegree": args.use_outdegree,
        "ancestor_feature": args.ancestor_feature,
        "skip_blocked": args.skip_blocked_descendants,
        "allow_skip": args.allow_skip,
        "force_fill": args.force_fill,
        "device": device,
    }


def train_epoch(
    model: TGNPolicy,
    opt: Adam,
    episodes: list[Episode],
    *,
    args: argparse.Namespace,
    alpha: float,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    order = list(episodes)
    random.Random(args.seed + epoch).shuffle(order)
    rates: list[float] = []
    returns: list[float] = []
    decisions: list[int] = []
    common = episode_kwargs(args, device)
    t0 = time.perf_counter()
    batch: list[tuple[Episode, Trajectory]] = []

    def apply_batch(group: list[tuple[Episode, Trajectory]]) -> None:
        ready = []
        for ep, traj in group:
            if not traj.choices:
                continue
            traj.returns = reward_to_go(traj.step_rewards, traj.block_rate)
            if len(traj.returns) != len(traj.choices):
                raise RuntimeError("return length does not match decisions")
            ready.append((ep, traj))
        if not ready:
            return
        baseline = sum(traj.returns[0] for _, traj in ready) / len(ready)
        returns.extend(traj.returns[0] for _, traj in ready)
        opt.zero_grad(set_to_none=True)
        try:
            for ep, traj in ready:
                adv = [g - baseline for g in traj.returns]
                run_episode(
                    model,
                    ep,
                    alpha=0.0,
                    train=True,
                    fixed_choices=traj.choices,
                    advantages=adv,
                    loss_scale=float(len(ready) * len(adv)),
                    bptt_chunk=BPTT_CHUNK,
                    build_graph=True,
                    **common,
                )
        except FloatingPointError:
            log("  non-finite loss, batch skipped")
            opt.zero_grad(set_to_none=True)
            return
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    for n_done, ep in enumerate(order, start=1):
        choices: list[int] = []
        traj = run_episode(
            model,
            ep,
            alpha=alpha,
            train=True,
            choice_out=choices,
            build_graph=False,
            **common,
        )
        traj.choices = choices
        batch.append((ep, traj))
        rates.append(traj.block_rate)
        decisions.append(len(choices))
        if len(batch) >= args.batch_size:
            apply_batch(batch)
            batch = []
        if n_done % 50 == 0 or n_done == len(order):
            elapsed = time.perf_counter() - t0
            log(
                f"  train {n_done}/{len(order)} "
                f"block {sum(rates) / len(rates):.4%} "
                f"decisions {sum(decisions) / len(decisions):.1f} "
                f"{elapsed / n_done:.2f}s/tree"
            )
    if batch:
        apply_batch(batch)
    return {
        "train_block_rate": sum(rates) / len(rates) if rates else float("nan"),
        "train_return": sum(returns) / len(returns) if returns else float("nan"),
        "mean_decisions": sum(decisions) / len(decisions) if decisions else float("nan"),
        "seconds": time.perf_counter() - t0,
    }


def selection_summary(model: TGNPolicy, episodes: list[Episode], args: argparse.Namespace, device: torch.device) -> dict:
    """数现有策略在贪心决定下实际拦了几个人。不改权重。"""
    if not episodes:
        return {"n": 0}
    was = model.training
    model.eval()
    counts: list[int] = []
    rates: list[float] = []
    for ep in episodes:
        traj = run_episode(
            model,
            ep,
            k=args.k,
            alpha=0.0,
            train=False,
            use_outdegree=args.use_outdegree,
            ancestor_feature=args.ancestor_feature,
            skip_blocked=args.skip_blocked_descendants,
            allow_skip=args.allow_skip,
            force_fill=args.force_fill,
            device=device,
        )
        counts.append(len(traj.selected))
        rates.append(traj.block_rate)
    if was:
        model.train()
    n = len(counts)
    hist = {str(i): sum(c == i for c in counts) for i in range(args.k + 1)}
    extra = sum(c > args.k for c in counts)
    if extra:
        hist[f">{args.k}"] = extra
    return {
        "n": n,
        "mean_selected": sum(counts) / n,
        "mean_block_rate": sum(rates) / n,
        "frac_full_budget": sum(c == args.k for c in counts) / n,
        "frac_none": sum(c == 0 for c in counts) / n,
        "hist": hist,
    }


def evaluate_split(model, episodes, args, device) -> dict[str, float]:
    t0 = time.perf_counter()
    model_rate = policy_block_rate(
        model,
        episodes,
        k=args.k,
        use_outdegree=args.use_outdegree,
        ancestor_feature=args.ancestor_feature,
        skip_blocked=args.skip_blocked_descendants,
        allow_skip=args.allow_skip,
        force_fill=args.force_fill,
        device=device,
    )
    return {
        "model": model_rate,
        "outdegree_obs": rule_block_rate(episodes, "outdegree_obs", args.k),
        "greedy_oracle": rule_block_rate(episodes, "greedy_oracle", args.k),
        "seconds": time.perf_counter() - t0,
        "n": float(len(episodes)),
    }


def _time_full_pass(model, opt, ep, args, device) -> str:
    """Always skip, so the whole window is differentiated. This is the slow path."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    opt.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    try:
        traj = run_episode(
            model,
            ep,
            k=args.k,
            alpha=0.0,
            train=True,
            use_outdegree=args.use_outdegree,
            ancestor_feature=args.ancestor_feature,
            skip_blocked=args.skip_blocked_descendants,
            device=device,
            force_skip=True,
        )
        # Two copies make a non-zero baseline without a second full window.
        twin = Trajectory(log_probs=list(traj.log_probs), step_rewards=list(traj.step_rewards), block_rate=traj.block_rate + 0.01)
        loss = reinforce_loss([traj, twin])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sec = time.perf_counter() - t0
        mem = ""
        if device.type == "cuda":
            mem = f" peak_gb {torch.cuda.max_memory_allocated(device) / 1024**3:.2f}"
        return f"obs {len(ep.arrivals)} decisions {len(traj.log_probs)} {sec:.1f}s{mem}"
    except RuntimeError as exc:
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return f"obs {len(ep.arrivals)} FAILED {exc}"


def smoke_time(model, episodes, args, device) -> None:
    if not episodes:
        raise RuntimeError("no training episodes to time")
    ranked = sorted(episodes, key=lambda ep: len(ep.arrivals))
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model.train()
    picks = {
        "p50": ranked[len(ranked) // 2],
        "p90": ranked[min(len(ranked) - 1, int(0.90 * (len(ranked) - 1)))],
        "p99": ranked[min(len(ranked) - 1, int(0.99 * (len(ranked) - 1)))],
        "max": ranked[-1],
    }
    seconds = {}
    for name, ep in picks.items():
        msg = _time_full_pass(model, opt, ep, args, device)
        log(f"smoke full-window {name} {ep.cid} {msg}")
        seconds[name] = msg


def write_results(payload: dict) -> None:
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {OUT_JSON.name}")


def apply_tag(tag: str) -> None:
    global OUT_JSON, CKPT, TRAIN_LOG
    if not tag:
        return
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    base = Path(__file__).resolve().parent
    OUT_JSON = base / f"tgn_rl_{safe}_results.json"
    CKPT = base / f"tgn_rl_{safe}.pt"
    TRAIN_LOG = base / f"tgn_rl_{safe}_train.log"


def main() -> None:
    parser = argparse.ArgumentParser(description="甲：TGN 记忆 + REINFORCE 序贯阻断")
    parser.add_argument("--window", default="time:1h")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mem-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shaping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ancestor-feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-blocked-descendants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-outdegree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-skip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-fill", action="store_true", help="窗口走完后把没花掉的名额按模型分数补满")
    parser.add_argument("--tag", default="", help="非空时结果、权重和日志改写到 tgn_rl_<tag>.*，避免覆盖上一轮")
    parser.add_argument("--count-selected", action="store_true", help="只统计已有权重实际拦了几个人")
    parser.add_argument("--smoke", type=int, default=0, help="非 0 时只做自检、基线回放和耗时估计")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="从 tgn_rl_blocking.pt 的 next_epoch 接着训")
    parser.add_argument("--ckpt", default="")
    args = parser.parse_args()
    apply_tag(args.tag)
    if args.batch_size < 2:
        raise SystemExit("batch size must be at least 2 so the return baseline is not identically zero")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device} window {args.window} k {args.k} seed {args.seed} allow_skip {args.allow_skip} force_fill {args.force_fill} tag {args.tag or '-'}")
    self_check()

    log("Loading cascades ...")
    labels = load_labels()
    cas_map = load_cascades()
    ids = [i for i in cas_map if i in labels]
    split_rng = random.Random(args.seed)
    train_ids, val_ids, test_ids = stratified_split(ids, labels, split_rng)
    log(f"split train/val/test {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    log("Building episodes ...")
    train_eps = build_episodes(cas_map, train_ids, labels, args.window)
    val_eps = [ep for ep in build_episodes(cas_map, val_ids, labels, args.window) if ep.label == 1]
    test_eps = [ep for ep in build_episodes(cas_map, test_ids, labels, args.window) if ep.label == 1]
    sizes = sorted(len(ep.arrivals) for ep in train_eps)
    if sizes:
        def _at(p: float) -> int:
            return sizes[min(len(sizes) - 1, int(p * (len(sizes) - 1)))]

        log(
            f"usable train {len(train_eps)} val_rumor {len(val_eps)} test_rumor {len(test_eps)} "
            f"train_obs_p50 {_at(0.50)} p90 {_at(0.90)} p99 {_at(0.99)} max {sizes[-1]} "
            f"over_1000 {sum(s > 1000 for s in sizes)}"
        )
    if not train_eps:
        raise RuntimeError("no usable training episodes")
    published = check_published_baselines(test_eps, args.window, args.k)
    model = TGNPolicy(args.mem_dim, args.hidden)
    leakage_ep = min(train_eps, key=lambda ep: len(ep.arrivals))
    leakage_check(model, leakage_ep, args.window)
    if args.smoke:
        model.to(device)
        smoke_time(model, train_eps, args, device)
        log("smoke done, training not started")
        return

    model.to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        log(f"loaded {args.ckpt}")
    elif (args.eval_only or args.count_selected) and CKPT.exists():
        ckpt = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        log(f"loaded {CKPT.name}")

    if args.count_selected:
        val_stats = selection_summary(model, val_eps, args, device)
        test_stats = selection_summary(model, test_eps, args, device)
        payload = {
            "checkpoint": str(CKPT),
            "k": args.k,
            "window": args.window,
            "val_rumor": val_stats,
            "test_rumor": test_stats,
        }
        if args.tag:
            out = CKPT.with_name(CKPT.stem + "_selection_counts.json")
        else:
            out = Path(__file__).resolve().parent / "tgn_rl_selection_counts.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(
            f"val selected mean {val_stats['mean_selected']:.2f} "
            f"full {val_stats['frac_full_budget']:.1%} none {val_stats['frac_none']:.1%} "
            f"block {val_stats['mean_block_rate']:.4%}"
        )
        log(
            f"test selected mean {test_stats['mean_selected']:.2f} "
            f"full {test_stats['frac_full_budget']:.1%} none {test_stats['frac_none']:.1%} "
            f"block {test_stats['mean_block_rate']:.4%}"
        )
        log(f"wrote {out.name}")
        return

    if args.eval_only:
        test_metrics = evaluate_split(model, test_eps, args, device)
        void = test_metrics["model"] > ORACLE_1H_K5 + 1e-4 or test_metrics["model"] > test_metrics["greedy_oracle"] + 1e-4
        payload = {
            "protocol": protocol_dict(args, train_ids, val_ids, test_ids, train_eps, val_eps, test_eps),
            "published_test_baseline": published,
            "test_rumor": {**test_metrics, "leakage_void": void},
        }
        write_results(payload)
        return

    log("Scoring validation baselines once ...")
    val_outdeg = rule_block_rate(val_eps, "outdegree_obs", args.k)
    val_oracle = rule_block_rate(val_eps, "greedy_oracle", args.k)
    log(f"val rumor n={len(val_eps)} outdegree {val_outdeg:.4%} oracle {val_oracle:.4%}")
    log(f"BPTT chunk {BPTT_CHUNK}: shorter trees get a full gradient, longer trees detach between chunks")
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best = -1.0
    stall = 0
    best_epoch = 0
    history: list = []
    start_epoch = 1
    if args.resume or args.ckpt:
        resume_path = Path(args.ckpt) if args.ckpt else CKPT
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        log(f"loaded {resume_path}")
        if args.resume:
            best = float(resume_ckpt.get("val_block_rate", -1.0))
            best_epoch = int(resume_ckpt.get("best_epoch", resume_ckpt.get("epoch", 0)))
            stall = int(resume_ckpt.get("stall", 0))
            history = list(resume_ckpt.get("history", []))
            start_epoch = int(resume_ckpt.get("next_epoch", best_epoch + 1))
            if "optimizer" in resume_ckpt:
                opt.load_state_dict(resume_ckpt["optimizer"])
                log("restored optimizer")
            else:
                log("optimizer starts fresh")
            log(
                f"resume at epoch {start_epoch}/{args.epochs} "
                f"best epoch {best_epoch} val {best:.4%} stall {stall}/{args.patience}"
            )
    for epoch in range(start_epoch, args.epochs + 1):
        alpha = alpha_for_epoch(epoch, args.epochs, args.shaping)
        log(f"epoch {epoch} alpha {alpha:.4f}")
        train_stats = train_epoch(model, opt, train_eps, args=args, alpha=alpha, epoch=epoch, device=device)
        t_val = time.perf_counter()
        val_rate = policy_block_rate(
            model,
            val_eps,
            k=args.k,
            use_outdegree=args.use_outdegree,
            ancestor_feature=args.ancestor_feature,
            skip_blocked=args.skip_blocked_descendants,
            allow_skip=args.allow_skip,
            force_fill=args.force_fill,
            device=device,
        )
        val_seconds = time.perf_counter() - t_val
        row = {
            "epoch": epoch,
            "alpha": alpha,
            **train_stats,
            "val_block_rate": val_rate,
            "val_outdegree": val_outdeg,
            "val_oracle": val_oracle,
            "val_seconds": val_seconds,
        }
        history.append(row)
        log(
            f"epoch {epoch} done train_block {train_stats['train_block_rate']:.4%} "
            f"val {val_rate:.4%} outdegree {val_outdeg:.4%} oracle {val_oracle:.4%}"
        )
        if val_rate > best:
            best = val_rate
            best_epoch = epoch
            stall = 0
            save_ckpt(model, epoch, best, args, optimizer=opt, stall=0, history=history, next_epoch=epoch + 1)
            log(f"  saved {CKPT.name} val {best:.4%}")
        else:
            stall += 1
            if CKPT.exists():
                kept = torch.load(CKPT, map_location="cpu", weights_only=False)
                kept["stall"] = stall
                kept["history"] = history
                kept["next_epoch"] = epoch + 1
                kept["best_epoch"] = best_epoch
                kept["val_block_rate"] = best
                torch.save(kept, CKPT)
            log(f"  no improvement {stall}/{args.patience}")
            if stall >= args.patience:
                log("early stop")
                break

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate_split(model, test_eps, args, device)
    void = test_metrics["model"] > ORACLE_1H_K5 + 1e-4 or test_metrics["model"] > test_metrics["greedy_oracle"] + 1e-4
    payload = {
        "protocol": protocol_dict(args, train_ids, val_ids, test_ids, train_eps, val_eps, test_eps),
        "published_test_baseline": published,
        "best_epoch": best_epoch,
        "best_val_block_rate": best,
        "history": history,
        "test_rumor": {
            "n": len(test_eps),
            "model": test_metrics["model"],
            "outdegree_obs": test_metrics["outdegree_obs"],
            "greedy_oracle": test_metrics["greedy_oracle"],
            "leakage_void": void,
            "meets_success_line": (not void) and test_metrics["model"] >= SUCCESS_1H_K5,
        },
    }
    write_results(payload)
    log(
        f"test rumor n={len(test_eps)} model {test_metrics['model']:.4%} "
        f"outdegree {test_metrics['outdegree_obs']:.4%} oracle {test_metrics['greedy_oracle']:.4%} "
        f"void {void}"
    )


def protocol_dict(args, train_ids, val_ids, test_ids, train_eps, val_eps, test_eps) -> dict:
    return {
        "role": "甲",
        "model": "TGN/GRU memory + REINFORCE sequential blocking",
        "window": args.window,
        "k": args.k,
        "seed": args.seed,
        "split": "cascade-level 70/15/15 stratified by label",
        "n_ids": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "n_usable": {"train": len(train_eps), "val_rumor": len(val_eps), "test_rumor": len(test_eps)},
        "shaping": args.shaping,
        "ancestor_feature": args.ancestor_feature,
        "skip_blocked_descendants": args.skip_blocked_descendants,
        "use_outdegree": args.use_outdegree,
        "allow_skip": args.allow_skip,
        "force_fill": args.force_fill,
        "batch_size": args.batch_size,
        "bptt_chunk": BPTT_CHUNK,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "no_old_checkpoint": True,
        "feature_scale": "outdegree is log1p; memory init uses MiniTGN's 3 features",
    }


if __name__ == "__main__":
    main()
