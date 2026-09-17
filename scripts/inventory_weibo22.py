#!/usr/bin/env python3
"""盘点 weibo22 级联数据，并与 DAMMFND data/ / weibo/ 做 ID 交集。"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIBO22 = ROOT / "weibo22"
POSTS_DIR = WEIBO22 / "posts" / "Weibo"
EVENTS = WEIBO22 / "events.txt"
OUT_JSON = ROOT / "scripts" / "weibo22_inventory.json"


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": float(min(xs)),
        "p25": percentile(xs, 0.25),
        "median": percentile(xs, 0.50),
        "p75": percentile(xs, 0.75),
        "p90": percentile(xs, 0.90),
        "max": float(max(xs)),
        "mean": float(statistics.fmean(xs)),
    }


def parse_events(path: Path) -> dict[str, dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    events: dict[str, dict] = {}
    for m in re.finditer(
        r"eid:(\d+)\s+label:(-?\d+)\s+([\d\s]+?)(?=eid:|\Z)",
        raw,
        flags=re.S,
    ):
        eid = m.group(1)
        label = int(m.group(2))
        ids = m.group(3).split()
        events[eid] = {"label": label, "n_ids": len(ids), "id_set_head": ids[:3]}
    return events


def mid_candidates(value) -> set[str]:
    """Weibo mid 在 CSV 里常被写成科学计数法/float，尽量还原可比字符串。"""
    out: set[str] = set()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return out
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return out
    out.add(s)
    out.add(s.split(".")[0])
    try:
        f = float(s)
        if math.isfinite(f) and f > 0:
            out.add(str(int(round(f))))
    except ValueError:
        pass
    return {x for x in out if x.isdigit()}


def load_dammfnd_ids() -> dict:
    import pandas as pd

    packs = {
        "data": [
            ROOT / "data" / "train_origin.csv",
            ROOT / "data" / "val_origin.csv",
            ROOT / "data" / "test_origin.csv",
        ],
        "weibo_raw": [
            ROOT / "weibo" / "train_2_domain.csv",
            ROOT / "weibo" / "val_2_domain.csv",
            ROOT / "weibo" / "test_2_domain.csv",
        ],
        "weibo_aligned": [
            ROOT / "weibo" / "train_aligned.csv",
            ROOT / "weibo" / "val_aligned.csv",
            ROOT / "weibo" / "test_aligned.csv",
        ],
    }
    result = {}
    for name, files in packs.items():
        ids: set[str] = set()
        texts: set[str] = set()
        n_rows = 0
        for fp in files:
            if not fp.exists():
                continue
            df = pd.read_csv(fp, encoding="utf-8")
            n_rows += len(df)
            if "post_id" in df.columns:
                for v in df["post_id"].tolist():
                    ids |= mid_candidates(v)
            if "content" in df.columns:
                for t in df["content"].astype(str).tolist():
                    t = "".join(t.split())
                    if len(t) >= 20:
                        texts.add(t[:80])
        result[name] = {"n_rows": n_rows, "n_id_tokens": len(ids), "ids": ids, "text_heads": texts}
    return result


def scan_cascades(events: dict[str, dict]) -> dict:
    files = sorted(POSTS_DIR.glob("*.json"))
    n_files = len(files)
    nodes_list = []
    edges_list = []
    uid_n_list = []
    span_h_list = []
    picture_n_list = []
    has_picture_cascade = 0
    empty = 0
    json_errors = 0
    labeled = 0
    label_counter = Counter()
    unlabeled = 0
    root_ids: list[str] = []
    all_node_ids: set[str] = set()
    root_texts: dict[str, str] = {}
    total_nodes = 0
    total_edges = 0
    total_pictures = 0
    all_uids: set[str] = set()

    for i, fp in enumerate(files):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            json_errors += 1
            continue
        if isinstance(data, dict):
            data = data.get("posts") or data.get("data") or [data]
        if not isinstance(data, list) or not data:
            empty += 1
            continue

        n = 0
        n_edge = 0
        n_pic = 0
        ts = []
        uids = set()
        root_id = fp.stem
        root_text = ""
        for item in data:
            if not isinstance(item, dict):
                continue
            n += 1
            iid = str(item.get("id") or item.get("mid") or "")
            if iid:
                all_node_ids.add(iid)
            if item.get("parent"):
                n_edge += 1
            pic = item.get("picture")
            if pic and str(pic).lower() not in ("none", "null", ""):
                n_pic += 1
            t = item.get("t")
            if t is not None:
                try:
                    ts.append(int(t))
                except (TypeError, ValueError):
                    pass
            uid = item.get("uid")
            if uid is not None:
                uids.add(str(uid))
                all_uids.add(str(uid))
            if not item.get("parent") and item.get("text"):
                root_id = str(item.get("id") or root_id)
                root_text = str(item.get("text") or item.get("original_text") or "")

        if n == 0:
            empty += 1
            continue

        total_nodes += n
        total_edges += n_edge
        total_pictures += n_pic
        nodes_list.append(n)
        edges_list.append(n_edge)
        uid_n_list.append(len(uids))
        picture_n_list.append(n_pic)
        if n_pic > 0:
            has_picture_cascade += 1
        if ts:
            span_h_list.append((max(ts) - min(ts)) / 3600.0)
        root_ids.append(root_id)
        compact = "".join(root_text.split())
        if len(compact) >= 20:
            root_texts[root_id] = compact[:80]

        ev = events.get(fp.stem) or events.get(root_id)
        if ev is not None:
            labeled += 1
            label_counter[ev["label"]] += 1
        else:
            unlabeled += 1

        if (i + 1) % 500 == 0:
            print(f"  scanned {i+1}/{n_files}", flush=True)

    return {
        "n_json_files": n_files,
        "json_errors": json_errors,
        "empty": empty,
        "n_cascades_ok": len(nodes_list),
        "labeled": labeled,
        "unlabeled": unlabeled,
        "label_counts": dict(label_counter),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "total_unique_uids": len(all_uids),
        "total_pictures_fields": total_pictures,
        "cascades_with_any_picture": has_picture_cascade,
        "nodes": summarize([float(x) for x in nodes_list]),
        "edges": summarize([float(x) for x in edges_list]),
        "unique_uids_per_cascade": summarize([float(x) for x in uid_n_list]),
        "span_hours": summarize(span_h_list),
        "pictures_per_cascade": summarize([float(x) for x in picture_n_list]),
        "size_buckets": {
            "lt50": sum(1 for x in nodes_list if x < 50),
            "50_199": sum(1 for x in nodes_list if 50 <= x < 200),
            "200_999": sum(1 for x in nodes_list if 200 <= x < 1000),
            "1000_4999": sum(1 for x in nodes_list if 1000 <= x < 5000),
            "ge5000": sum(1 for x in nodes_list if x >= 5000),
        },
        "span_buckets_hours": {
            "lt1h": sum(1 for x in span_h_list if x < 1),
            "1_24h": sum(1 for x in span_h_list if 1 <= x < 24),
            "1_7d": sum(1 for x in span_h_list if 24 <= x < 168),
            "ge7d": sum(1 for x in span_h_list if x >= 168),
        },
        "root_ids": root_ids,
        "all_node_ids": all_node_ids,
        "root_texts": root_texts,
    }


def overlap(scan: dict, damm: dict) -> dict:
    roots = set(scan["root_ids"])
    nodes = scan["all_node_ids"]
    out = {}
    for name, pack in damm.items():
        ids = pack["ids"]
        root_hit = roots & ids
        node_hit = nodes & ids
        text_hit = 0
        if pack["text_heads"]:
            weibo_texts = set(scan["root_texts"].values())
            text_hit = len(weibo_texts & pack["text_heads"])
        out[name] = {
            "dammfnd_rows": pack["n_rows"],
            "dammfnd_id_tokens": pack["n_id_tokens"],
            "root_id_overlap": len(root_hit),
            "any_node_id_overlap": len(node_hit),
            "root_text80_overlap": text_hit,
        }
    return out


def main() -> None:
    print("Parsing events.txt ...", flush=True)
    events = parse_events(EVENTS)
    ev_labels = Counter(v["label"] for v in events.values())
    print(f"  events: {len(events)}  labels={dict(ev_labels)}", flush=True)

    print("Scanning cascade JSON ...", flush=True)
    scan = scan_cascades(events)
    print(f"  cascades_ok={scan['n_cascades_ok']} nodes={scan['total_nodes']}", flush=True)

    print("Loading DAMMFND CSVs ...", flush=True)
    damm = load_dammfnd_ids()
    ov = overlap(scan, damm)
    print("Overlap:", json.dumps(ov, ensure_ascii=False), flush=True)

    report = {
        "events_file": {
            "n_events": len(events),
            "label_counts": dict(ev_labels),
            "ids_per_event": summarize([float(v["n_ids"]) for v in events.values()]),
        },
        "cascades": {k: v for k, v in scan.items() if k not in ("root_ids", "all_node_ids", "root_texts")},
        "overlap": ov,
        "notes": {
            "label_convention": "events.txt label:0/1，未在本脚本中改写含义；Ma et al. Weibo 谣言级联里通常 0=非谣言、1=谣言，需对照论文确认。",
            "picture_field": "统计 JSON 的 picture 字段非空；多数转发节点无图。",
            "id_overlap_caveat": "DAMMFND CSV 的 post_id 常为 float/科学计数法，16 位 mid 可能丢失精度，ID 交集偏低时以文本前 80 字交集为辅。",
        },
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
