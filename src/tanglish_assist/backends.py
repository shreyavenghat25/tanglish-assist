"""Model backends for evaluation and the demo. Each returns (completion_text, latency_seconds).

- ollama : a local model served by Ollama (base model or your fine-tuned GGUF)
- openai : any OpenAI-compatible API (OpenAI, Groq, Together, OpenRouter, vLLM...) — used for
           the "big model" baseline. Reads OPENAI_API_KEY / OPENAI_BASE_URL.
- hf     : Hugging Face transformers (+ optional LoRA adapter) — use inside Colab right after training
- mock   : a deterministic fake model, only for testing the pipeline
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

from .schema import SYSTEM_PROMPT


def _messages(text: str, few_shot: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot or []:
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant", "content": ex["target"]})
    msgs.append({"role": "user", "content": text})
    return msgs


class OllamaBackend:
    def __init__(self, model: str, host: str = "http://localhost:11434", few_shot=None):
        self.model, self.host, self.few_shot = model, host.rstrip("/"), few_shot

    def __call__(self, text: str) -> tuple[str, float]:
        body = json.dumps({
            "model": self.model,
            "messages": _messages(text, self.few_shot),
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"], time.perf_counter() - t0


class OpenAIBackend:
    def __init__(self, model: str, few_shot=None):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
        self.model, self.few_shot = model, few_shot

    def __call__(self, text: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.model, messages=_messages(text, self.few_shot), temperature=0)
        return r.choices[0].message.content or "", time.perf_counter() - t0


class HFBackend:
    def __init__(self, model: str, adapter: str | None = None, max_new_tokens: int = 200):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype="auto")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, text: str) -> tuple[str, float]:
        ids = self.tok.apply_chat_template(_messages(text), add_generation_prompt=True,
                                           return_tensors="pt").to(self.model.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        completion = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return completion, time.perf_counter() - t0


class MockBackend:
    """Keyword rules + a few deliberate mistakes. Only for testing the eval pipeline."""

    RULES = [("refund", "refund_status"), ("cancel", "cancel_order"), ("return", "return_request"),
             ("otp", "account_issue"), ("login", "account_issue"), ("coupon", "coupon_issue"),
             ("cashback", "coupon_issue"), ("address", "address_change"), ("damage", "damaged_item"),
             ("crack", "damaged_item"), ("wrong", "wrong_item"), ("pay", "payment_issue"),
             ("late", "delivery_delay"), ("status", "order_status")]

    def __call__(self, text: str) -> tuple[str, float]:
        import re
        low = text.lower()
        intent = next((i for k, i in self.RULES if k in low), "other")
        m = re.search(r"ORD\d{8}", text, re.IGNORECASE)
        ents = {"order_id": m.group(0).upper() if m else None, "item": None, "date": None, "amount": None}
        return json.dumps({"intent": intent, "entities": ents, "reply": "Sure, check panren."}), 0.001


def get_backend(kind: str, model: str = "", adapter: str | None = None, few_shot=None):
    if kind == "ollama":
        return OllamaBackend(model, os.environ.get("OLLAMA_HOST", "http://localhost:11434"), few_shot)
    if kind == "openai":
        return OpenAIBackend(model, few_shot)
    if kind == "hf":
        return HFBackend(model, adapter)
    if kind == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend {kind}")
