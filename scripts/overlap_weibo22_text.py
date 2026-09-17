#!/usr/bin/env python3
"""DAMMFND content vs weibo22 root-text overlap (ID 列不可用)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL_RE = re.compile(r"https?://\S+")
NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def norm(s: str) -> str:
    s = URL_RE.sub("", str(s))
    s = NON_WORD.sub("", s)
    return s


def prefixes(s: str) -> set[str]:
    s = norm(s)
    out = set()
    if len(s) >= 12:
        out.add(s[:12])
    if len(s) >= 20:
        out.add(s[:20])
    return out


def main() -> None:
    damm: set[str] = set()
    for fp in [
        ROOT / "data" / "train_origin.csv",
        ROOT / "data" / "val_origin.csv",
        ROOT / "data" / "test_origin.csv",
    ]:
        df = pd.read_csv(fp, encoding="utf-8")
        for t in df["content"].astype(str):
            damm |= prefixes(t)

    hit_root = 0
    n = 0
    examples = []
    posts = ROOT / "weibo22" / "posts" / "Weibo"
    for fp in posts.glob("*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        n += 1
        root_t = ""
        for item in data:
            if isinstance(item, dict) and not item.get("parent"):
                root_t = str(item.get("original_text") or item.get("text") or "")
                break
        if prefixes(root_t) & damm:
            hit_root += 1
            if len(examples) < 5:
                examples.append({"id": fp.stem, "text": root_t[:80]})

    out = {
        "cascades": n,
        "damm_prefix_keys": len(damm),
        "root_prefix_hit": hit_root,
        "examples": examples,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    inv = ROOT / "scripts" / "weibo22_inventory.json"
    report = json.loads(inv.read_text(encoding="utf-8"))
    report["overlap"]["data"]["root_prefix12_20_hit"] = hit_root
    inv.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
