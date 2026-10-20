"""Generate synthetic Tanglish training data from templates.

    python scripts/generate_synthetic.py --n 3000 --seed 42

Writes data/train.jsonl and data/val.jsonl. Each record:
    {"id", "text", "intent", "entities", "reply", "source": "template"}

Labels are correct by construction: an entity is only set when its slot was filled.
Noise (romanisation variants, typos, casing, prefixes/suffixes) is applied to the template
text only, never inside a filled slot, so labels stay exact.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tanglish_assist.schema import INTENTS, empty_entities  # noqa: E402
from tanglish_assist.templates import (  # noqa: E402
    AMOUNTS, DATES, ITEMS, PREFIXES, REPLIES, SUFFIXES, TEMPLATES,
)

# Common romanisation variants seen in real Tanglish chat.
VARIANTS = {
    "irukku": ["iruku", "irku", "irukkku"],
    "iruku": ["irukku", "irku"],
    "eppo": ["epo", "eppa", "yeppo"],
    "illa": ["ila", "illai"],
    "innum": ["inum", "innu"],
    "pannunga": ["pannunga", "pnnunga", "panunga", "pannuga"],
    "sollunga": ["solunga", "sollunga", "sollungo"],
    "varala": ["varla", "varalai", "vaarala"],
    "aachu": ["achu", "aachchu", "aachu"],
    "aagala": ["agala", "aagalai", "aagla"],
    "vanthuchu": ["vandhuchu", "vanduchu", "vanthuchi"],
    "pannanum": ["pananum", "pannanu", "pannaum"],
    "enna": ["ena", "yenna"],
    "aagiduchu": ["agiduchu", "aayiduchu", "aagiruchu"],
    "romba": ["rombha", "roomba"],
    "varum": ["varuma", "varum"],
    "please": ["pls", "plz", "please"],
}

PROTECT = re.compile(r"(\{[a-z_]+\})")


def _noisy_words(segment: str, rng: random.Random) -> str:
    words = segment.split(" ")
    out = []
    for w in words:
        key = w.lower().strip("?,.!")
        if key in VARIANTS and rng.random() < 0.35:
            w = w.lower().replace(key, rng.choice(VARIANTS[key]))
        # occasional typo: drop one interior character from a longer word
        if len(w) > 5 and rng.random() < 0.04:
            i = rng.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1:]
        out.append(w)
    return " ".join(out)


def add_noise(template: str, rng: random.Random) -> str:
    parts = PROTECT.split(template)
    parts = [p if PROTECT.fullmatch(p) else _noisy_words(p, rng) for p in parts]
    text = rng.choice(PREFIXES) + "".join(parts) + rng.choice(SUFFIXES)
    # Casing noise only on the template text, never on {placeholders}.
    r = rng.random()
    case = str.lower if r < 0.15 else str.upper if r < 0.20 else None
    if case:
        text = "".join(p if PROTECT.fullmatch(p) else case(p) for p in PROTECT.split(text))
    elif r < 0.30:
        text = text[:1].upper() + text[1:]
    return re.sub(r"\s+", " ", text).strip()


def order_surface(rng: random.Random) -> tuple[str, str]:
    oid = "ORD" + "".join(str(rng.randrange(10)) for _ in range(8))
    surface = rng.choice([oid, oid, oid.lower(), "#" + oid, "Ord" + oid[3:]])
    return surface, oid


def amount_surface(rng: random.Random) -> tuple[str, int]:
    amt = rng.choice(AMOUNTS)
    fmt = rng.choice(["{a}", "{a}", "Rs {a}", "₹{a}", "rs.{a}", "{c}", "₹{c}", "{a} rs"])
    return fmt.format(a=amt, c=f"{amt:,}"), amt


ORDER_INTENTS = {"order_status", "delivery_delay", "cancel_order", "return_request",
                 "wrong_item", "damaged_item", "refund_status", "address_change"}


def make_example(intent: str, rng: random.Random) -> dict:
    template = rng.choice(TEMPLATES[intent])
    # Sometimes customers add the order ID even if the sentence didn't need it.
    if intent in ORDER_INTENTS and "{order}" not in template and rng.random() < 0.25:
        template += rng.choice([", order id {order}", " ({order})", ". order no {order}"])

    noisy = add_noise(template, rng)
    ents = empty_entities()
    fills = {}
    if "{order}" in noisy:
        fills["order"], ents["order_id"] = order_surface(rng)
    if "{item}" in noisy:
        fills["item"] = ents["item"] = rng.choice(ITEMS)
    if "{date}" in noisy:
        fills["date"] = ents["date"] = rng.choice(DATES)
    if "{amount}" in noisy:
        fills["amount"], ents["amount"] = amount_surface(rng)
    text = noisy.format(**fills)

    order_ref = f" ({ents['order_id']})" if ents["order_id"] else ""
    reply = rng.choice(REPLIES[intent]).format(order_ref=order_ref)
    return {"text": text, "intent": intent, "entities": ents, "reply": reply, "source": "template"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="total examples (train+val)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Never let a test sentence leak into training.
    test_texts = set()
    test_path = ROOT / "data" / "test_seed.jsonl"
    if test_path.exists():
        for line in test_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                test_texts.add(json.loads(line)["text"].lower().strip())

    per_intent = args.n // len(INTENTS)
    seen, records = set(), []
    for intent in INTENTS:
        made, attempts = 0, 0
        while made < per_intent and attempts < per_intent * 50:
            attempts += 1
            ex = make_example(intent, rng)
            key = ex["text"].lower()
            if key in seen or key in test_texts:
                continue
            seen.add(key)
            records.append(ex)
            made += 1

    rng.shuffle(records)
    for i, r in enumerate(records):
        r["id"] = f"syn-{i:05d}"
    n_val = int(len(records) * args.val_frac)
    splits = {"val": records[:n_val], "train": records[n_val:]}
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples -> {out_dir / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
