"""Evaluate any model on the Tanglish test set and add a row to results/leaderboard.md.

Examples
--------
# 1. Base model zero-shot (the "before" number)
ollama pull llama3.2:3b
python scripts/evaluate.py --backend ollama --model llama3.2:3b --run base-llama3.2-3b

# 2. Big-model baseline via any OpenAI-compatible API
OPENAI_API_KEY=... python scripts/evaluate.py --backend openai --model gpt-4o-mini --run api-baseline

# 3. Your fine-tuned model after `ollama create tanglish-assist -f ollama/Modelfile`
python scripts/evaluate.py --backend ollama --model tanglish-assist --run tanglish-assist-v1

# Keyword-rules baseline (no model needed; also a pipeline smoke test)
python scripts/evaluate.py --backend mock --run keyword-baseline --verified_only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tanglish_assist.backends import get_backend  # noqa: E402
from tanglish_assist.metrics import compute  # noqa: E402
from tanglish_assist.schema import parse_output  # noqa: E402


def load_test(path: Path, verified_only: bool) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if verified_only:
        rows = [r for r in rows if r.get("verified")]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["ollama", "openai", "hf", "mock"])
    ap.add_argument("--model", default="")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (hf backend)")
    ap.add_argument("--run", required=True, help="name for this run, e.g. base-llama3.2-3b")
    ap.add_argument("--test", default=str(ROOT / "data" / "test_seed.jsonl"))
    ap.add_argument("--verified_only", action="store_true",
                    help="only score rows a human marked verified (use this for reported numbers)")
    ap.add_argument("--few_shot", type=int, default=0,
                    help="add N training examples to the prompt (fairer baseline for base models)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    test = load_test(Path(args.test), args.verified_only)
    if args.limit:
        test = test[: args.limit]
    unverified = sum(1 for r in test if not r.get("verified"))
    if unverified:
        print(f"WARNING: {unverified}/{len(test)} test rows are not human-verified. "
              "Do not report these numbers until you verify them (scripts/review_test_set.py).")

    few_shot = None
    if args.few_shot:
        from tanglish_assist.schema import to_target
        train = [json.loads(l) for l in (ROOT / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        few_shot = [{"text": r["text"], "target": to_target(r["intent"], r["entities"], r["reply"])}
                    for r in train[: args.few_shot]]

    backend = get_backend(args.backend, args.model, args.adapter, few_shot)
    rows = []
    for i, ex in enumerate(test, 1):
        try:
            completion, latency = backend(ex["text"])
        except Exception as e:  # count a timeout/crash as a failed prediction, keep going
            print(f"  ! {ex['id']} failed: {e}")
            completion, latency = "", None
        pred, valid = parse_output(completion)
        rows.append({"id": ex["id"], "text": ex["text"], "gold": ex, "pred": pred,
                     "raw": completion, "valid_json": valid, "latency_s": latency})
        if i % 10 == 0 or i == len(test):
            print(f"  {i}/{len(test)}")

    m = compute(rows)
    m.update({"run": args.run, "backend": args.backend, "model": args.model or ("keyword-rules" if args.backend == "mock" else args.backend),
              "few_shot": args.few_shot, "verified_only": args.verified_only,
              "unverified_rows": unverified, "date": str(date.today())})

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / f"{args.run}.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    with open(res_dir / f"{args.run}.predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "text", "pred", "raw", "valid_json")}
                               | {"gold_intent": r["gold"]["intent"]}, ensure_ascii=False) + "\n")

    update_leaderboard(res_dir)
    print(json.dumps({k: m[k] for k in ("intent_accuracy", "intent_macro_f1", "valid_json_rate",
                                        "all_entities_correct", "hallucinated_order_id_rate")}, indent=2))
    if m["top_confusions"]:
        print("Top confusions:", m["top_confusions"])


def update_leaderboard(res_dir: Path) -> None:
    runs = []
    for p in sorted(res_dir.glob("*.json")):
        try:
            runs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    runs = [r for r in runs if "intent_accuracy" in r and r.get("run") != "mock"]
    runs.sort(key=lambda r: -r["intent_accuracy"])
    lines = [
        "# Leaderboard — TanglishSupport test set",
        "",
        "| Run | Model | n | Verified only | Intent acc | Macro-F1 | Valid JSON | All entities | Halluc. order ID | p50 latency (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| {r['run']} | {r['model']} | {r['n']} | {'yes' if r.get('verified_only') else 'NO'} | "
            f"{r['intent_accuracy']:.1%} | {r['intent_macro_f1']:.3f} | {r['valid_json_rate']:.1%} | "
            f"{r['all_entities_correct']:.1%} | {r['hallucinated_order_id_rate']:.1%} | "
            f"{r.get('latency_p50_s', float('nan')):.2f} |")
    lines += ["", "Only rows with *Verified only = yes* should be quoted on a resume or README."]
    (res_dir / "leaderboard.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
