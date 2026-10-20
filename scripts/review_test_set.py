"""Walk through the test set and verify / correct every label by hand.

    python scripts/review_test_set.py            # review unverified rows
    python scripts/review_test_set.py --all      # re-review everything

For each row: [Enter] = label is correct, mark verified
              e        = edit intent/entities
              d        = delete row (ambiguous / bad example)
              q        = save and quit
Progress is saved after every row, so you can stop anytime.

Labelling rules (keep them consistent!):
- One intent per message. If two apply, pick what the customer wants done NOW.
- Entities are copied exactly as written in the message (item "iPhone cover", date "nethu night").
- If several dates/amounts appear, label the FIRST one mentioned.
- amount is an integer in rupees; pincodes, sizes and percentages are NOT amounts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tanglish_assist.schema import ENTITY_KEYS, INTENTS  # noqa: E402

PATH = ROOT / "data" / "test_seed.jsonl"


def save(rows):
    PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    i = 0
    while i < len(rows):
        r = rows[i]
        if r.get("verified") and not args.all:
            i += 1
            continue
        print("\n" + "=" * 70)
        print(f"[{i + 1}/{len(rows)}] {r['text']}")
        print(f"  intent  : {r['intent']}")
        for k in ENTITY_KEYS:
            print(f"  {k:<8}: {r['entities'][k]}")
        cmd = input("Enter=ok  e=edit  d=delete  q=quit > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "d":
            rows.pop(i)
            save(rows)
            continue
        if cmd == "e":
            print("Intents:", ", ".join(INTENTS))
            new = input(f"  intent [{r['intent']}]: ").strip()
            if new:
                if new not in INTENTS:
                    print("  !! unknown intent, not changed")
                else:
                    r["intent"] = new
            for k in ENTITY_KEYS:
                new = input(f"  {k} [{r['entities'][k]}] (type 'null' to clear): ").strip()
                if new == "null":
                    r["entities"][k] = None
                elif new:
                    r["entities"][k] = int(new) if k == "amount" else new
        r["verified"] = True
        save(rows)
        i += 1

    done = sum(1 for r in rows if r.get("verified"))
    print(f"\nVerified {done}/{len(rows)} rows. Saved to {PATH}")


if __name__ == "__main__":
    main()
