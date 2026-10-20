"""Gradio demo for Tanglish Assist.

Local (uses your Ollama model):
    python app/app.py

Hugging Face Spaces (free CPU): set Space variables
    TA_BACKEND=llamacpp
    TA_GGUF_REPO=shreyavenghat25/tanglish-assist-llama3.2-3b-GGUF
    TA_GGUF_FILE=<the .gguf filename in that repo>
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tanglish_assist.schema import SYSTEM_PROMPT, parse_output  # noqa: E402

BACKEND = os.environ.get("TA_BACKEND", "ollama")

if BACKEND == "llamacpp":
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    _path = hf_hub_download(os.environ["TA_GGUF_REPO"], os.environ["TA_GGUF_FILE"])
    _llm = Llama(model_path=_path, n_ctx=2048, verbose=False)

    def generate(text: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        r = _llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            temperature=0, max_tokens=200)
        return r["choices"][0]["message"]["content"], time.perf_counter() - t0
else:
    from tanglish_assist.backends import OllamaBackend
    generate = OllamaBackend(os.environ.get("TA_MODEL", "tanglish-assist"))

EXAMPLES = [
    "bro ORD48213907 innum varala, promise panna date nethu",
    "UPI la ₹1,499 cut aachu but order place aagala 😡",
    "mixie box open panna jar crack aagi irukku, replace pannunga",
    "coupon code FESTIVE100 apply aagala",
    "refund eppo varum? return panni 6 days aachu",
]

INTENT_EMOJI = {"order_status": "📦", "delivery_delay": "⏰", "cancel_order": "❌", "return_request": "↩️",
                "wrong_item": "🔀", "damaged_item": "💥", "refund_status": "💸", "payment_issue": "💳",
                "address_change": "🏠", "coupon_issue": "🏷️", "account_issue": "🔐", "other": "💬"}


def run(text: str):
    if not text.strip():
        return "", {}, ""
    raw, latency = generate(text)
    pred, valid = parse_output(raw)
    if pred is None:
        return "⚠️ Model did not return valid JSON", {"raw": raw}, ""
    label = f"{INTENT_EMOJI.get(pred['intent'], '')} **{pred['intent']}**  ·  {latency:.2f}s  ·  " \
            f"{'valid JSON ✅' if valid else 'repaired JSON ⚠️'}"
    return label, pred["entities"], pred["reply"]


with gr.Blocks(title="Tanglish Assist") as demo:
    gr.Markdown("# Tanglish Assist\nA 3B model fine-tuned with QLoRA to understand **Tamil-English "
                "code-mixed** customer-support messages — running fully offline.")
    inp = gr.Textbox(label="Customer message (Tanglish)", lines=2)
    btn = gr.Button("Analyse", variant="primary")
    intent = gr.Markdown()
    with gr.Row():
        ents = gr.JSON(label="Extracted entities")
        reply = gr.Textbox(label="Suggested reply", lines=3)
    gr.Examples(EXAMPLES, inp)
    btn.click(run, inp, [intent, ents, reply])
    inp.submit(run, inp, [intent, ents, reply])

if __name__ == "__main__":
    demo.launch()
