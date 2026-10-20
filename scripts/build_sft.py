"""Convert train/val records into chat-format SFT files used by the training notebook.

    python scripts/build_sft.py

Writes data/sft_train.jsonl and data/sft_val.jsonl, one {"messages": [...]} per line,
using the exact SYSTEM_PROMPT that evaluation and Ollama use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tanglish_assist.schema import SYSTEM_PROMPT, to_target  # noqa: E402

for split in ("train", "val"):
    src = ROOT / "data" / f"{split}.jsonl"
    extra = ROOT / "data" / f"{split}_augmented.jsonl"  # optional, from augment_with_llm.py
    rows = [json.loads(l) for p in (src, extra) if p.exists()
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = ROOT / "data" / f"sft_{split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["text"]},
                {"role": "assistant", "content": to_target(r["intent"], r["entities"], r["reply"])},
            ]
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"{split}: {len(rows)} -> {out}")
