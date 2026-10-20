"""(Optional, recommended) Paraphrase template examples with a bigger LLM for more natural variety.

Template data is clean but repetitive. This rewrites each message the way a real customer
might type it, while REQUIRING every labelled entity to survive verbatim. Rewrites that drop
or change an entity are discarded, so labels stay correct.

Works with any OpenAI-compatible endpoint:
    # local, free: a bigger model in Ollama
    OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
        python scripts/augment_with_llm.py --model llama3.1:8b --n 800
    # or a hosted API (Groq / OpenRouter / OpenAI)
    OPENAI_BASE_URL=https://api.groq.com/openai/v1 OPENAI_API_KEY=... \
        python scripts/augment_with_llm.py --model <model-name> --n 800

Writes data/train_augmented.jsonl (build_sft.py picks it up automatically).
Spot-check ~50 rows by eye before training — LLM paraphrases can drift in intent.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROMPT = """Rewrite this Indian e-commerce customer message in natural Tanglish \
(Tamil words written in English letters, mixed with English), the way a real person types \
on WhatsApp. Change the wording and sentence structure, but keep the same meaning and intent.
You MUST keep these exact strings unchanged if present: {keep}
Return only the rewritten message, nothing else.

Message: {text}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI()
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in (ROOT / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    sample = rng.sample(rows, min(args.n, len(rows)))

    out_path = ROOT / "data" / "train_augmented.jsonl"
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(sample, 1):
            keep = []
            for k in ("item", "date"):
                if r["entities"][k]:
                    keep.append(r["entities"][k])
            if r["entities"]["order_id"]:
                keep.append(r["entities"]["order_id"])
            if r["entities"]["amount"]:
                keep.append(str(r["entities"]["amount"]))
            resp = client.chat.completions.create(
                model=args.model, temperature=0.9,
                messages=[{"role": "user", "content": PROMPT.format(
                    keep=", ".join(f'"{k}"' for k in keep) or "(none)", text=r["text"])}])
            new = (resp.choices[0].message.content or "").strip().strip('"')
            if not new or len(new) > 300:
                continue
            if any(k.lower() not in new.lower() for k in keep):
                continue  # entity lost -> label would be wrong -> discard
            f.write(json.dumps({**r, "id": r["id"] + "-aug", "text": new, "source": "llm_paraphrase"},
                               ensure_ascii=False) + "\n")
            kept += 1
            if i % 50 == 0:
                print(f"{i}/{len(sample)} processed, {kept} kept")
    print(f"Kept {kept}/{len(sample)} paraphrases -> {out_path}")


if __name__ == "__main__":
    main()
