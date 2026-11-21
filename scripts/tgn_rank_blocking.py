#!/usr/bin/env python3
"""丙线：MiniTGN 用成对排序训练，窗口结束后一次性选 top-5。"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from torch.optim import Adam

from blocking_baseline import MIN_FUTURE, POSTS_DIR, RESULTS_DIR, SEED, covered_future, load_labels, parse_cascade, pick
from tgn_blocking import MAX_EDGES, MAX_GRAD_EDGES, WINDOWS, MiniTGN, build_prefix_tensors, stratified_split


K = 5
MARGIN = 1.0
PAIR_LIMIT = 256
ACCUM = 8
LR = 1e-3
WEIGHT_DECAY = 1e-5
SUCCESS_1H = 0.122
STRATEGIES = ("outdegree_obs", "greedy_oracle")
PRIMARY_WINDOW = "time:1h"


def rank_pairs(target: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Uniformly sample strict high/low pairs without materializing a square matrix."""
    values, indices = target.sort()
    lower_counts = torch.searchsorted(values, values).cumsum(0).tolist()
    if not lower_counts or lower_counts[-1] == 0:
        return torch.empty((0, 2), dtype=torch.long)
    order = indices.tolist()
    pairs = []
    for index in rng.sample(range(lower_counts[-1]), min(PAIR_LIMIT, lower_counts[-1])):
        high = bisect_right(lower_counts, index)
        low = index - lower_counts[high - 1]
        pairs.append((order[high], order[low]))
    return torch.tensor(pairs, dtype=torch.long)


def prefix_pack(cas: dict, window: str) -> dict | None:
    # MiniTGN 的深度特征沿祖先链计算，先拦截时间跨出窗口的父边。
    nodes = cas["nodes"]
    cutoff = cas["t_min"] + int(window.removeprefix("time:").removesuffix("h")) * 3600
    for node in nodes.values():
        parent = nodes.get(node["parent"])
        if node["t"] <= cutoff and parent is not None and parent["t"] > cutoff:
            raise RuntimeError(f"{cas['id']} {window}: observed node has a future parent")
    return build_prefix_tensors(cas, window)


def scores(model: MiniTGN, pack: dict, *, train: bool) -> torch.Tensor:
    memory = model.replay(pack["feat"], pack["src"], pack["dst"], pack["log_dt"], train=train)
    return model.scores(memory, pack["feat"], pack["cand_idx"])


@torch.no_grad()
def leakage_check(model: MiniTGN, examples: dict) -> dict:
    """Reattach the future to the root and check both inputs and resulting scores."""
    model.eval()
    checked = {}
    for window, cases in examples.items():
        ordered = sorted(cases, key=lambda item: len(item[1]["obs"]))
        samples = [ordered[i] for i in sorted({0, len(ordered) // 2, len(ordered) - 1})]
        changed_targets = 0
        for cas, pack in samples:
            nodes = cas["nodes"].copy()
            for node in pack["fut"]:
                nodes[node] = dict(nodes[node], parent=cas["root"])
            children = defaultdict(list)
            for node, fields in nodes.items():
                if fields["parent"] is not None:
                    children[fields["parent"]].append(node)
            altered = prefix_pack(dict(cas, nodes=nodes, children=children), window)
            if altered is None or pack["cand_ids"] != altered["cand_ids"]:
                raise RuntimeError(f"{cas['id']} {window}: altering the future changed eligible candidates")
            for name in ("feat", "src", "dst", "log_dt", "cand_idx"):
                if not torch.equal(pack[name], altered[name]):
                    raise RuntimeError(f"{cas['id']} {window}: altering the future changed {name}")
            torch.testing.assert_close(scores(model, pack, train=False), scores(model, altered, train=False))
            changed_targets += int(not torch.equal(pack["y"], altered["y"]))
        if changed_targets == 0:
            raise RuntimeError(f"{window}: future perturbation did not change any ranking target")
        checked[window] = {"trees_checked": len(samples), "targets_changed": changed_targets, "passed": True}
    return checked


def rate(cas: dict, pack: dict, selected: list[str]) -> float:
    return covered_future(cas, selected, pack["fut"]) / len(pack["fut"])


def baselines(examples: dict) -> dict:
    rng = random.Random(SEED)
    return {
        window: {
            strategy: sum(
                rate(cas, pack, pick(strategy, cas, pack["obs"], pack["fut"], K, rng))
                for cas, pack in cases
            ) / len(cases)
            for strategy in STRATEGIES
        }
        for window, cases in examples.items()
    }


@torch.no_grad()
def evaluate(model: MiniTGN, examples: dict) -> tuple[dict, dict]:
    model.eval()
    results, counts = {}, {}
    for window, cases in examples.items():
        start = time.perf_counter()
        rates = []
        hist = [0] * (K + 1)
        for cas, pack in cases:
            pred = scores(model, pack, train=False)
            top = pred.topk(min(K, pred.numel())).indices.tolist()
            selected = [pack["cand_ids"][i] for i in top]
            if cas["root"] in selected or len(set(selected)) != len(selected):
                raise RuntimeError(f"{cas['id']} {window}: invalid selection")
            hist[len(selected)] += 1
            rates.append(rate(cas, pack, selected))
        n = len(rates)
        block_rate = sum(rates) / n
        results[window] = {"n": n, "model": block_rate, "seconds": time.perf_counter() - start}
        counts[window] = {
            "n": n, "mean_selected": sum(i * count for i, count in enumerate(hist)) / n,
            "mean_block_rate": block_rate, "frac_full_budget": hist[K] / n,
            "frac_none": hist[0] / n, "hist": hist,
        }
    return results, counts


def train_epoch(model: MiniTGN, optimizer: Adam, examples: list, rng: random.Random) -> dict:
    start = time.perf_counter()
    model.train()
    rng.shuffle(examples)
    loss_sum = 0.0
    updates = 0
    for offset in range(0, len(examples), ACCUM):
        batch = examples[offset:offset + ACCUM]
        optimizer.zero_grad(set_to_none=True)
        for pack, pairs in batch:
            pred = scores(model, pack, train=True)
            high, low = pairs.unbind(dim=1)
            loss = F.relu(MARGIN - (pred[high] - pred[low])).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite pairwise ranking loss")
            (loss / len(batch)).backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        updates += 1
    return {"train_rank_loss": loss_sum / len(examples), "optimizer_updates": updates,
            "seconds": time.perf_counter() - start}


def result_report(report: dict) -> dict:
    """Expose the main window in the same fields as the REINFORCE reports."""
    by_window = report["test_rumor"]
    primary = by_window[PRIMARY_WINDOW]
    best_validation = report["history"][report["best_epoch"] - 1]["val_rumor"][PRIMARY_WINDOW]
    protocol = {
        **report["protocol"],
        "window": PRIMARY_WINDOW,
        "n_ids": report["n_ids"],
        "n_usable": {
            "train": report["n_usable"]["train"][PRIMARY_WINDOW],
            "val_rumor": best_validation["n"],
            "test_rumor": primary["n"],
        },
    }
    history = []
    for row in report["history"]:
        val = row["val_rumor"][PRIMARY_WINDOW]
        history.append({
            **row, "val_block_rate": val["model"], "val_outdegree": val["outdegree_obs"],
            "val_oracle": val["greedy_oracle"], "val_seconds": val["seconds"],
        })
    return {
        **report,
        "protocol": protocol,
        "published_test_baseline": {strategy: primary[strategy] for strategy in STRATEGIES},
        "history": history,
        "test_rumor": {
            **primary,
            "meets_success_line": not primary["leakage_void"] and primary["model"] >= SUCCESS_1H,
        },
        "test_rumor_by_window": by_window,
    }


def selection_report(checkpoint: Path, validation: dict, test: dict) -> dict:
    normalized = {
        split: {
            window: {**row, "hist": {str(i): count for i, count in enumerate(row["hist"])}}
            for window, row in counts.items()
        }
        for split, counts in (("val_rumor", validation), ("test_rumor", test))
    }
    return {
        "checkpoint": checkpoint.name, "k": K, "window": PRIMARY_WINDOW,
        **{split: counts[PRIMARY_WINDOW] for split, counts in normalized.items()},
        **{f"{split}_by_window": counts for split, counts in normalized.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tag", default="", help="为独立的丙线运行追加文件名后缀")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.tag and not all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in args.tag):
        parser.error("--tag accepts ASCII letters, numbers, hyphens, and underscores")
    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"tgn_rank_{args.tag}" if args.tag else "tgn_rank"
    log_path = RESULTS_DIR / f"{name}_train.log"
    checkpoint_path = RESULTS_DIR / f"{name}.pt"
    result_path = RESULTS_DIR / f"{name}_results.json"
    counts_path = RESULTS_DIR / f"{name}_selection_counts.json"
    if any(path.exists() for path in (log_path, checkpoint_path, result_path, counts_path)):
        parser.error("output already exists; choose a fresh --tag")

    with log_path.open("x", encoding="utf-8") as log_file:
        def log(message: str) -> None:
            print(message, flush=True)
            print(message, file=log_file, flush=True)

        torch.manual_seed(SEED)
        rng = random.Random(SEED)
        pair_rng = random.Random(SEED)
        start = time.perf_counter()
        log(f"role 丙 device cpu torch {torch.__version__} windows {WINDOWS} k={K} seed={SEED}")
        log(f"pairwise hinge margin {MARGIN} max {PAIR_LIMIT} pairs per tree-window, epochs {args.epochs}, fresh weights")
        labels = load_labels()
        paths = {path.stem: path for path in sorted(POSTS_DIR.glob("*.json")) if path.stem in labels}
        train_ids, val_ids, test_ids = stratified_split(list(paths), labels, rng)
        eval_cases = {split: {window: [] for window in WINDOWS} for split in ("val", "test")}
        train_examples = []
        eligible_train = dict.fromkeys(WINDOWS, 0)
        rankable_train = dict.fromkeys(WINDOWS, 0)
        pair_counts = dict.fromkeys(WINDOWS, 0)
        log(f"split train/val/test {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
        for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            for idx, cid in enumerate(ids, 1):
                if idx % 500 == 0:
                    log(f"prepare {split} {idx}/{len(ids)}")
                if split != "train" and labels[cid] != 1:
                    continue
                cas = parse_cascade(paths[cid])
                if cas is None:
                    raise RuntimeError(f"unreadable cascade: {paths[cid]}")
                for window in WINDOWS:
                    pack = prefix_pack(cas, window)
                    if pack is None:
                        continue
                    if split != "train":
                        eval_cases[split][window].append((cas, pack))
                        continue
                    eligible_train[window] += 1
                    pairs = rank_pairs(pack["y"], pair_rng)
                    if pairs.numel():
                        rankable_train[window] += 1
                        pair_counts[window] += len(pairs)
                        train_examples.append((pack, pairs))
        if not train_examples or any(not cases for split in eval_cases.values() for cases in split.values()):
            raise RuntimeError("missing usable training, validation or test examples")
        val_rules = baselines(eval_cases["val"])
        test_rules = baselines(eval_cases["test"])
        reference = json.loads((RESULTS_DIR / "tgn_blocking_results.json").read_text(encoding="utf-8"))
        old_model = {}
        for window in WINDOWS:
            old_row = reference["test_rumor_block_rate"][window][str(K)]
            old_model[window] = old_row["tgn"]["mean"]
            for strategy in STRATEGIES:
                if old_row[strategy]["n"] != len(eval_cases["test"][window]) or not math.isclose(
                    old_row[strategy]["mean"], test_rules[window][strategy], abs_tol=5.0001e-5,
                ):
                    raise RuntimeError(f"{window} {strategy}: test split did not reproduce the published baseline")
            log(f"{window} train {eligible_train[window]} rankable {rankable_train[window]} pairs {pair_counts[window]} "
                f"val rumor {len(eval_cases['val'][window])} test rumor {len(eval_cases['test'][window])} "
                f"test outdegree {test_rules[window]['outdegree_obs']:.4%} oracle {test_rules[window]['greedy_oracle']:.4%}")
        preparation_seconds = time.perf_counter() - start
        model = MiniTGN()
        initial_check = leakage_check(model, eval_cases["val"])
        log(f"future perturbation check before training {initial_check}")
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        history = []
        best_rate = -1.0
        for epoch in range(1, args.epochs + 1):
            stats = train_epoch(model, optimizer, train_examples, rng)
            val_metrics, val_counts = evaluate(model, eval_cases["val"])
            for window in WINDOWS:
                val_metrics[window].update(val_rules[window])
            history.append({"epoch": epoch, **stats, "val_rumor": val_metrics})
            log(f"epoch {epoch}/{args.epochs} rank loss {stats['train_rank_loss']:.6f} "
                f"train {stats['seconds']:.1f}s updates {stats['optimizer_updates']} " + " | ".join(
                    f"{window} val {val_metrics[window]['model']:.4%} "
                    f"outdegree {val_rules[window]['outdegree_obs']:.4%} "
                    f"oracle {val_rules[window]['greedy_oracle']:.4%} ({val_metrics[window]['seconds']:.1f}s)"
                    for window in WINDOWS
                ))
            score = val_metrics[PRIMARY_WINDOW]["model"]
            if score > best_rate:
                best_rate = score
                best_val_counts = val_counts
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_block_rate": best_rate}, checkpoint_path)
                log(f"saved best epoch {epoch} {checkpoint_path.name}")

        best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best["model"])
        best_check = leakage_check(model, eval_cases["test"])
        log(f"loaded best epoch {best['epoch']}; future perturbation check {best_check}")
        test, selection_counts = evaluate(model, eval_cases["test"])
        for window, row in test.items():
            row.update(test_rules[window])
            row["minitgn_smooth_l1"] = old_model[window]
            row["delta_vs_outdegree_pp"] = 100 * (row["model"] - row["outdegree_obs"])
            row["leakage_void"] = row["model"] > row["greedy_oracle"] + 1e-4
            log(f"{window} test rumor n={row['n']} rank {row['model']:.4%} outdegree {row['outdegree_obs']:.4%} "
                f"MiniTGN {row['minitgn_smooth_l1']:.4%} oracle {row['greedy_oracle']:.4%} "
                f"mean selected {selection_counts[window]['mean_selected']:.3f} "
                f"full budget {selection_counts[window]['frac_full_budget']:.2%} void {row['leakage_void']}")
        for split, counts in (("val", best_val_counts), ("test", selection_counts)):
            row = counts[PRIMARY_WINDOW]
            log(f"{split} selected mean {row['mean_selected']:.2f} full {row['frac_full_budget']:.1%} "
                f"none {row['frac_none']:.1%} block {row['mean_block_rate']:.4%}")
        meets_success_line = (not any(row["leakage_void"] for row in test.values())
                              and test["time:1h"]["model"] >= SUCCESS_1H
                              and test["time:6h"]["model"] >= test["time:6h"]["outdegree_obs"])
        report = {
            "protocol": {
                "role": "丙", "model": "MiniTGN + pairwise hinge ranker", "dataset": "weibo22", "device": "cpu",
                "torch": str(torch.__version__), "seed": SEED, "windows": WINDOWS, "k": K, "minimum_future": MIN_FUTURE,
                "split": "cascade-level 70/15/15 stratified by label", "selection": "window-end top-k, non-root",
                "max_edges": MAX_EDGES, "max_grad_edges": MAX_GRAD_EDGES, "margin": MARGIN,
                "pair_sampling": "uniform without replacement, fixed at preparation, strict targets only",
                "max_pairs_per_tree_window": PAIR_LIMIT, "epochs": args.epochs, "accum": ACCUM, "lr": LR,
                "weight_decay": WEIGHT_DECAY, "mem_dim": model.mem_dim, "no_old_checkpoint": True,
                "best_criterion": "1h validation rumor block_rate (earliest on a tie)",
            },
            "n_ids": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
            "n_usable": {"train": eligible_train, "rankable_train": rankable_train, "sampled_pairs": pair_counts},
            "best_epoch": best["epoch"], "best_val_block_rate": best["val_block_rate"],
            "preparation_seconds": preparation_seconds, "history": history, "test_rumor": test,
            "leakage_checks": {"before_training": initial_check, "best_checkpoint": best_check},
            "success_criteria": {"1h_minimum": SUCCESS_1H, "6h_minimum": test["time:6h"]["outdegree_obs"]},
            "meets_success_line": meets_success_line,
            "run_seconds": time.perf_counter() - start,
        }
        result_path.write_text(json.dumps(result_report(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts = selection_report(checkpoint_path, best_val_counts, selection_counts)
        counts_path.write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log(f"success line {meets_success_line}; wrote {result_path.name} and {counts_path.name}")


if __name__ == "__main__":
    main()
