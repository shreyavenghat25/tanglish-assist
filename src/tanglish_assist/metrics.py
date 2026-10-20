"""Evaluation metrics. Pure functions — no model code — so they are unit-testable."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from sklearn.metrics import confusion_matrix, f1_score

from .schema import ENTITY_KEYS, INTENTS


def _norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = re.sub(r"\s+", " ", str(v)).strip().lower().lstrip("#")
    return s or None


def entity_match(gold: Any, pred: Any, lenient: bool = False) -> bool:
    g, p = _norm(gold), _norm(pred)
    if g == p:
        return True
    if lenient and isinstance(g, str) and isinstance(p, str):
        return g in p or p in g
    return False


def compute(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: [{"text", "gold": {...}, "pred": {...}|None, "valid_json": bool, "latency_s": float}]"""
    n = len(rows)
    gold_i = [r["gold"]["intent"] for r in rows]
    pred_i = [(r["pred"] or {}).get("intent", "__invalid__") for r in rows]

    out: dict[str, Any] = {"n": n}
    out["intent_accuracy"] = sum(g == p for g, p in zip(gold_i, pred_i)) / n
    out["intent_macro_f1"] = f1_score(gold_i, pred_i, labels=INTENTS, average="macro", zero_division=0)
    out["valid_json_rate"] = sum(r["valid_json"] for r in rows) / n

    ent = {}
    all_correct = 0
    hallucinated_oid = 0
    for r in rows:
        pred_e = (r["pred"] or {}).get("entities", {}) or {}
        ok_all = True
        for k in ENTITY_KEYS:
            ok = entity_match(r["gold"]["entities"][k], pred_e.get(k))
            ent.setdefault(k, {"exact": 0, "lenient": 0})
            ent[k]["exact"] += ok
            ent[k]["lenient"] += ok or entity_match(r["gold"]["entities"][k], pred_e.get(k), lenient=True)
            ok_all &= ok
        all_correct += ok_all
        oid = pred_e.get("order_id")
        if oid and str(oid).upper().lstrip("#") not in r["text"].upper():
            hallucinated_oid += 1
    out["entity_exact"] = {k: v["exact"] / n for k, v in ent.items()}
    out["entity_lenient"] = {k: v["lenient"] / n for k, v in ent.items()}
    out["all_entities_correct"] = all_correct / n
    out["hallucinated_order_id_rate"] = hallucinated_oid / n

    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    if lat:
        lat_sorted = sorted(lat)
        out["latency_p50_s"] = statistics.median(lat)
        out["latency_p95_s"] = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]

    labels = INTENTS + ["__invalid__"]
    out["confusion_labels"] = labels
    out["confusion_matrix"] = confusion_matrix(gold_i, pred_i, labels=labels).tolist()
    out["top_confusions"] = Counter(
        f"{g} -> {p}" for g, p in zip(gold_i, pred_i) if g != p
    ).most_common(8)
    return out
