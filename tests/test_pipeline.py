import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tanglish_assist.metrics import compute  # noqa: E402
from tanglish_assist.schema import ENTITY_KEYS, INTENTS, parse_output, to_target  # noqa: E402


def _load(name):
    return [json.loads(l) for l in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def test_parse_strict_json():
    out = to_target("refund_status", {"order_id": "ORD12345678", "item": None, "date": "5 days", "amount": 499},
                    "Check panren")
    pred, valid = parse_output(out)
    assert valid and pred["intent"] == "refund_status" and pred["entities"]["amount"] == 499


def test_parse_fenced_and_chatty_output_is_salvaged_but_not_valid():
    raw = 'Sure! ```json\n{"intent": "other", "entities": {"order_id": null, "item": null, "date": null, "amount": "₹1,499"}, "reply": "hi"}\n```'
    pred, valid = parse_output(raw)
    assert pred["intent"] == "other" and pred["entities"]["amount"] == 1499
    assert not valid


def test_parse_garbage():
    assert parse_output("I cannot help with that") == (None, False)


def test_perfect_predictions_score_one():
    test = _load("test_seed.jsonl")
    rows = [{"text": r["text"], "gold": r, "pred": {"intent": r["intent"], "entities": r["entities"], "reply": "x"},
             "valid_json": True, "latency_s": 0.1} for r in test]
    m = compute(rows)
    assert m["intent_accuracy"] == 1.0 and m["all_entities_correct"] == 1.0
    assert m["hallucinated_order_id_rate"] == 0.0


def test_test_set_labels_are_well_formed():
    for r in _load("test_seed.jsonl"):
        assert r["intent"] in INTENTS
        assert set(r["entities"]) == set(ENTITY_KEYS)
        for k in ("item", "date"):
            if r["entities"][k]:
                assert r["entities"][k].lower() in r["text"].lower(), (r["id"], k)


def test_synthetic_labels_match_text_and_no_leakage():
    train = _load("train.jsonl") + _load("val.jsonl")
    test_texts = {r["text"].lower().strip() for r in _load("test_seed.jsonl")}
    for r in train:
        assert r["text"].lower().strip() not in test_texts
        assert "{" not in r["text"]
        e = r["entities"]
        if e["order_id"]:
            assert e["order_id"][3:] in r["text"]
        for k in ("item", "date"):
            if e[k]:
                assert e[k].lower() in r["text"].lower(), (r["id"], k)
