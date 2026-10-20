"""Shared output schema, system prompt and robust JSON parsing.

Every component (data generation, training, evaluation, Ollama, demo) imports from here,
so the model is always trained and evaluated against exactly the same contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
_CFG = yaml.safe_load((ROOT / "configs" / "intents.yaml").read_text(encoding="utf-8"))

INTENTS: list[str] = list(_CFG["intents"].keys())
ENTITY_KEYS: list[str] = list(_CFG["entities"].keys())

SYSTEM_PROMPT = (
    "You are Tanglish Assist, a customer-support model for an Indian e-commerce store. "
    "Customers write in Tanglish (Tamil written in English letters, mixed with English). "
    "Read the customer message and answer with ONLY a JSON object, no other text, with keys:\n"
    f'  "intent": one of {INTENTS}\n'
    '  "entities": {"order_id": string|null, "item": string|null, "date": string|null, "amount": integer|null}\n'
    '  "reply": a short, polite reply in the same Tanglish style as the customer.\n'
    "order_id looks like ORD followed by 8 digits. Use null for anything not mentioned. "
    "Never invent an order ID, date or amount."
)


def empty_entities() -> dict[str, Any]:
    return {k: None for k in ENTITY_KEYS}


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse a model completion into the schema.

    Returns (parsed_or_None, strictly_valid). `strictly_valid` is True only if the whole
    completion is a single JSON object matching the schema — that is the number we report
    as the "valid-JSON rate". If the strict parse fails we still try to salvage a JSON object
    from surrounding text so that intent accuracy is not double-penalised.
    """
    text = (text or "").strip()
    # Strip markdown code fences that base models love to add.
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

    strict = False
    obj = None
    try:
        obj = json.loads(fenced)
        strict = fenced == text and isinstance(obj, dict)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None

    if not isinstance(obj, dict):
        return None, False

    normalised = normalise(obj)
    strict = strict and is_valid(obj)
    return normalised, strict


def is_valid(obj: dict[str, Any]) -> bool:
    if obj.get("intent") not in INTENTS:
        return False
    ents = obj.get("entities")
    if not isinstance(ents, dict) or set(ents.keys()) != set(ENTITY_KEYS):
        return False
    if ents.get("amount") is not None and not isinstance(ents["amount"], int):
        return False
    return isinstance(obj.get("reply"), str) and len(obj["reply"].strip()) > 0


def normalise(obj: dict[str, Any]) -> dict[str, Any]:
    ents = obj.get("entities") if isinstance(obj.get("entities"), dict) else {}
    out_ents = empty_entities()
    for k in ENTITY_KEYS:
        v = ents.get(k)
        if isinstance(v, str):
            v = v.strip() or None
            if k == "amount" and v is not None:
                digits = re.sub(r"[^\d]", "", v)
                v = int(digits) if digits else None
            elif k == "order_id" and v is not None:
                v = v.upper()
        out_ents[k] = v
    intent = obj.get("intent")
    return {
        "intent": intent if intent in INTENTS else "__invalid__",
        "entities": out_ents,
        "reply": obj.get("reply") if isinstance(obj.get("reply"), str) else "",
    }


def to_target(intent: str, entities: dict[str, Any], reply: str) -> str:
    """Canonical serialisation used as the training target."""
    return json.dumps({"intent": intent, "entities": entities, "reply": reply}, ensure_ascii=False)
