#!/usr/bin/env python3
"""
最小 TGN 阻断实验（与 blocking_baseline.py 同一协议）。

在 t0 前的时序转发边上维护节点记忆，预测每个已出现非根节点的未来子树规模，
按预测值取 top-k 阻断。不含 DAMMFND 文本/图像（两套数据对不齐），只验证时序图本身能否超过观察出度。

训练：级联级 70/15/15 分层划分，窗口 time:1h 与 time:6h，SmoothL1(log1p(未来子树))。
评价：测试集谣言树，k=1/5/10，指标 block_rate。

速度：每棵树最多回放 128 条最近边；训练仅对最后 32 条边反传（截断 BPTT）。
上次全量边级反传过慢，已中断且无权重。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
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
    future_in_subtree,
    load_labels,
    observed_future,
    parse_cascade,
    pick,
)

OUT_JSON = Path(__file__).resolve().parent / "tgn_blocking_results.json"
CKPT = Path(__file__).resolve().parent / "tgn_blocking.pt"
TRAIN_LOG = Path(__file__).resolve().parent / "tgn_train.log"
RESUME_DEFAULT = Path(__file__).resolve().parent / "tgn_blocking_epoch2_interrupted.pt"


def log(msg: str) -> None:
    print(msg, flush=True)
    with TRAIN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
WINDOWS = ("time:1h", "time:6h")
K_LIST = (1, 5, 10)
FEAT_DIM = 3  # log_followers, log_hours_from_root, depth/16
MAX_EDGES = 128
MAX_GRAD_EDGES = 32


class MiniTGN(nn.Module):
    def __init__(self, mem_dim: int = 64):
        super().__init__()
        self.mem_dim = mem_dim
        self.feat_enc = nn.Sequential(nn.Linear(FEAT_DIM, mem_dim), nn.ReLU())
        self.msg = nn.Linear(mem_dim * 2 + 1, mem_dim)
        self.gru_src = nn.GRUCell(mem_dim, mem_dim)
        self.gru_dst = nn.GRUCell(mem_dim, mem_dim)
        self.scorer = nn.Sequential(
            nn.Linear(mem_dim + FEAT_DIM, mem_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mem_dim, 1),
        )

    def _edge_update(self, mem: torch.Tensor, s: int, d: int, log_dt_i: torch.Tensor):
        h = torch.cat([mem[s], mem[d], log_dt_i], dim=0)
        m = torch.tanh(self.msg(h))
        return self.gru_src(m, mem[s]), self.gru_dst(m, mem[d])

    def replay(
        self,
        feat: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        log_dt: torch.Tensor,
        train: bool = False,
    ) -> torch.Tensor:
        mem = self.feat_enc(feat)
        n = int(src.numel())
        if n == 0:
            return mem
        split = max(0, n - MAX_GRAD_EDGES) if train else n
        with torch.no_grad():
            for i in range(split):
                s = int(src[i].item())
                d = int(dst[i].item())
                ns, nd = self._edge_update(mem, s, d, log_dt[i : i + 1])
                mem[s], mem[d] = ns, nd
        if not train:
            return mem
        mem = mem.detach()
        for i in range(split, n):
            s = int(src[i].item())
            d = int(dst[i].item())
            ns, nd = self._edge_update(mem, s, d, log_dt[i : i + 1])
            mem = mem.index_copy(0, src[i : i + 1], ns.unsqueeze(0))
            mem = mem.index_copy(0, dst[i : i + 1], nd.unsqueeze(0))
        return mem

    def scores(self, mem: torch.Tensor, feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([mem[idx], feat[idx]], dim=-1)
        return self.scorer(x).squeeze(-1)


def node_depth(cas: dict, nid: str, cache: dict[str, int]) -> int:
    if nid in cache:
        return cache[nid]
    chain: list[str] = []
    cur: str | None = nid
    while cur is not None and cur not in cache:
        chain.append(cur)
        p = cas["nodes"][cur]["parent"]
        if p is None or p not in cas["nodes"]:
            cache[cur] = 0
            chain.pop()
            break
        cur = p
    for node in reversed(chain):
        p = cas["nodes"][node]["parent"]
        cache[node] = min(32, 1 + cache.get(p, 0)) if p in cache else 0
    return cache[nid]


def build_prefix_tensors(cas: dict, window: str, device: torch.device | None = None):
    obs_all, fut = observed_future(cas, window)
    root = cas["root"]
    cand = [n for n in obs_all if n != root]
    if len(fut) < MIN_FUTURE or not cand:
        return None
    obs_set = set(obs_all)
    ymap = future_in_subtree(cas, set(fut))
    depth_cache: dict[str, int] = {}
    ids = list(obs_all)
    if root in ids:
        ids.remove(root)
        ids = [root] + ids
    index = {nid: i for i, nid in enumerate(ids)}
    t0 = cas["t_min"]
    feat = torch.zeros(len(ids), FEAT_DIM, dtype=torch.float32)
    for nid, i in index.items():
        inf = cas["nodes"][nid]
        hours = max(0.0, (inf["t"] - t0) / 3600.0)
        feat[i, 0] = math.log1p(inf["followers"])
        feat[i, 1] = math.log1p(hours)
        feat[i, 2] = node_depth(cas, nid, depth_cache) / 16.0
    edges = []
    ordered = sorted(obs_all, key=lambda n: (cas["nodes"][n]["t"], n))
    for dst in ordered:
        p = cas["nodes"][dst]["parent"]
        if p is None or p not in obs_set or p not in index or dst not in index:
            continue
        dt = max(0, cas["nodes"][dst]["t"] - cas["nodes"][p]["t"])
        edges.append((index[p], index[dst], math.log1p(dt)))
    if len(edges) > MAX_EDGES:
        edges = edges[-MAX_EDGES:]
    if not edges:
        src = torch.zeros(0, dtype=torch.long)
        dst_t = torch.zeros(0, dtype=torch.long)
        log_dt = torch.zeros(0, dtype=torch.float32)
    else:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst_t = torch.tensor([e[1] for e in edges], dtype=torch.long)
        log_dt = torch.tensor([e[2] for e in edges], dtype=torch.float32)
    cand_idx = torch.tensor([index[n] for n in cand], dtype=torch.long)
    y = torch.tensor([math.log1p(ymap.get(n, 0)) for n in cand], dtype=torch.float32)
    pack = {
        "feat": feat,
        "src": src,
        "dst": dst_t,
        "log_dt": log_dt,
        "cand_idx": cand_idx,
        "y": y,
        "cand_ids": cand,
        "fut": fut,
        "obs": cand,
    }
    if device is not None:
        for key in ("feat", "src", "dst", "log_dt", "cand_idx", "y"):
            pack[key] = pack[key].to(device)
    return pack


def load_cascades() -> dict[str, dict]:
    out = {}
    files = sorted(POSTS_DIR.glob("*.json"))
    for i, fp in enumerate(files):
        cas = parse_cascade(fp)
        if cas is not None:
            out[cas["id"]] = cas
        if (i + 1) % 500 == 0:
            log(f"  load {i+1}/{len(files)}")
    return out


def stratified_split(ids: list[str], labels: dict[str, int], rng: random.Random):
    buckets = defaultdict(list)
    for i in ids:
        buckets[labels.get(i, -1)].append(i)
    tr, va, te = [], [], []
    for _, group in buckets.items():
        rng.shuffle(group)
        n = len(group)
        n_tr = int(0.70 * n)
        n_va = int(0.15 * n)
        tr.extend(group[:n_tr])
        va.extend(group[n_tr : n_tr + n_va])
        te.extend(group[n_tr + n_va :])
    rng.shuffle(tr)
    return tr, va, te


def pack_to_device(pack: dict, device: torch.device) -> dict:
    out = dict(pack)
    for key in ("feat", "src", "dst", "log_dt", "cand_idx", "y"):
        out[key] = pack[key].to(device)
    return out


def cascade_loss(model: MiniTGN, pack: dict, loss_fn: nn.Module) -> torch.Tensor:
    mem = model.replay(pack["feat"], pack["src"], pack["dst"], pack["log_dt"], train=True)
    pred = model.scores(mem, pack["feat"], pack["cand_idx"])
    return loss_fn(pred, pack["y"])


@torch.no_grad()
def eval_mse(model: MiniTGN, packs: list[dict], loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for pack in packs:
        p = pack_to_device(pack, device)
        mem = model.replay(p["feat"], p["src"], p["dst"], p["log_dt"], train=False)
        pred = model.scores(mem, p["feat"], p["cand_idx"])
        tot += float(loss_fn(pred, p["y"]).item())
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def tgn_select(model: MiniTGN, pack: dict, k: int) -> list[str]:
    model.eval()
    mem = model.replay(pack["feat"], pack["src"], pack["dst"], pack["log_dt"], train=False)
    pred = model.scores(mem, pack["feat"], pack["cand_idx"])
    k = min(k, pred.numel())
    top = torch.topk(pred, k=k).indices.tolist()
    return [pack["cand_ids"][i] for i in top]


def block_table(model: MiniTGN | None, cas_map: dict, ids: list[str], labels: dict, device, cache: dict):
    """cache[(cid, window)] = pack or None"""
    rng = random.Random(SEED)
    out = {}
    rumor_ids = [i for i in ids if labels.get(i) == 1]
    for window in WINDOWS:
        out[window] = {}
        for k in K_LIST:
            rates = {s: [] for s in ("random", "followers", "outdegree_obs", "greedy_oracle", "tgn")}
            for cid in rumor_ids:
                cas = cas_map[cid]
                key = (cid, window)
                if key not in cache:
                    cache[key] = build_prefix_tensors(cas, window, None)
                pack = cache[key]
                if pack is None:
                    continue
                obs, fut = pack["obs"], pack["fut"]
                for strat in ("random", "followers", "outdegree_obs", "greedy_oracle"):
                    sel = pick(strat, cas, obs, fut, k, rng)
                    rates[strat].append(covered_future(cas, sel, fut) / len(fut))
                if model is not None:
                    sel = tgn_select(model, pack_to_device(pack, device), k)
                    rates["tgn"].append(covered_future(cas, sel, fut) / len(fut))
            out[window][str(k)] = {
                s: {
                    "n": len(xs),
                    "mean": round(sum(xs) / len(xs), 4) if xs else None,
                }
                for s, xs in rates.items()
            }
    return out


def precompute_packs(cas_map, ids) -> list[dict]:
    packs = []
    for cid in ids:
        for w in WINDOWS:
            pack = build_prefix_tensors(cas_map[cid], w, None)
            if pack is not None:
                packs.append(pack)
    return packs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mem-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint to continue from; empty = train from scratch",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip remaining epochs and evaluate the resume/best checkpoint",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    log(f"device {device}")
    log("Loading cascades ...")
    labels = load_labels()
    cas_map = load_cascades()
    ids = [i for i in cas_map if i in labels]
    train_ids, val_ids, test_ids = stratified_split(ids, labels, rng)
    log(f"split train/val/test {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    log("Precompute train/val tensors ...")
    train_packs = precompute_packs(cas_map, train_ids)
    val_packs = precompute_packs(cas_map, val_ids)
    log(f"  train examples {len(train_packs)}  val {len(val_packs)}")

    model = MiniTGN(args.mem_dim).to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()
    best_val = float("inf")
    history = []
    start_epoch = 1

    resume_path = Path(args.resume) if args.resume else None
    if resume_path is None and RESUME_DEFAULT.exists() and not args.eval_only:
        resume_path = RESUME_DEFAULT
    if resume_path is None and args.eval_only and CKPT.exists():
        resume_path = CKPT
    if resume_path is not None:
        ckpt0 = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt0["model"])
        start_epoch = int(ckpt0.get("epoch", 0)) + 1
        best_val = float(ckpt0.get("val", best_val))
        history = list(ckpt0.get("history", []))
        log(f"resume {resume_path} last_epoch={ckpt0.get('epoch')} val={best_val:.6f} next={start_epoch}")

    if not args.eval_only:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            rng.shuffle(train_packs)
            opt.zero_grad(set_to_none=True)
            running, seen = 0.0, 0
            for i, pack in enumerate(train_packs, 1):
                loss = cascade_loss(model, pack_to_device(pack, device), loss_fn) / args.accum
                loss.backward()
                running += float(loss.item()) * args.accum
                seen += 1
                if i % 400 == 0:
                    log(f"    epoch {epoch} {i}/{len(train_packs)}")
                if i % args.accum == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            if seen % args.accum:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            val = eval_mse(model, val_packs, loss_fn, device)
            tr = running / max(seen, 1)
            history.append({"epoch": epoch, "train_mse": round(tr, 4), "val_mse": round(val, 4)})
            log(f"epoch {epoch}/{args.epochs}  train {tr:.4f}  val {val:.4f}")
            last_path = CKPT.with_name("tgn_blocking_last.pt")
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val": val, "history": history},
                last_path,
            )
            if val < best_val:
                best_val = val
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val": val, "history": history},
                    CKPT,
                )
                log(f"  saved {CKPT}")

    ckpt = torch.load(CKPT if CKPT.exists() else resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    log("Evaluating test rumor block_rate ...")
    cache: dict = {}
    test_tbl = block_table(model, cas_map, test_ids, labels, device, cache)
    report = {
        "protocol": {
            "windows": list(WINDOWS),
            "k": list(K_LIST),
            "split": "cascade-level 70/15/15 stratified by label",
            "seed": SEED,
            "model": "MiniTGN GRU memory on prefix retweet edges",
            "max_edges": MAX_EDGES,
            "max_grad_edges": MAX_GRAD_EDGES,
            "no_dammfnd": True,
            "best_epoch": ckpt["epoch"],
            "best_val_mse": ckpt["val"],
            "resumed": bool(resume_path),
        },
        "n": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "history": history,
        "test_rumor_block_rate": test_tbl,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(test_tbl, indent=2))
    log(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
