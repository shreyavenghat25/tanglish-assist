"""Generates notebooks/train_qlora_colab.ipynb (kept as code so diffs stay readable)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# Tanglish Assist — QLoRA fine-tuning on a free Colab T4

**Runtime → Change runtime type → T4 GPU** before running.

What this notebook does:
1. Loads Llama 3.2 3B Instruct in 4-bit (QLoRA) with Unsloth
2. Fine-tunes LoRA adapters on `data/sft_train.jsonl` (loss only on the assistant's JSON answer)
3. Evaluates on the held-out Tanglish test set with the **same metrics** as `scripts/evaluate.py`
4. Saves the adapter, and exports a 4-bit **GGUF** for Ollama

Free Colab can disconnect — checkpoints are saved to Google Drive so you can resume.
""")

code("""
# 1. Install (Unsloth pins compatible torch/transformers/trl versions for Colab)
!pip install -q unsloth
!pip install -q scikit-learn pyyaml
""")

code("""
# 2. Get the project code + data.
# Option A (recommended): your GitHub repo
REPO = "https://github.com/shreyavenghat25/tanglish-assist.git"
!git clone -q $REPO /content/tanglish-assist || echo "clone failed - use option B"
# Option B: upload tanglish-assist.zip in the Files panel, then:
# !unzip -q /content/tanglish-assist.zip -d /content/
%cd /content/tanglish-assist
!python scripts/build_sft.py
""")

code("""
# 3. Mount Drive for checkpoints (survives Colab disconnects)
from google.colab import drive
drive.mount('/content/drive')
OUT_DIR = "/content/drive/MyDrive/tanglish-assist/checkpoints"
""")

code("""
# 4. Load the base model in 4-bit
from unsloth import FastLanguageModel

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"   # ablation: try "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,          # <- this is what makes it QLoRA
)
""")

code("""
# 5. Attach LoRA adapters. Record r / alpha in your README — they are part of your results.
LORA_R = 16
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_R,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
model.print_trainable_parameters()
""")

code("""
# 6. Load chat-format data and render it with the model's own chat template
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": "data/sft_train.jsonl", "val": "data/sft_val.jsonl"})

def render(batch):
    return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]}

ds = ds.map(render, batched=True, remove_columns=["messages"])
print(ds)
print(ds["train"][0]["text"][-400:])
""")

code("""
# 7. Train. ~2,700 examples x 2 epochs takes roughly 20-40 min on a T4.
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        output_dir=OUT_DIR,
        optim="adamw_8bit",
        seed=3407,
        report_to="none",
    ),
)
# Only learn from the assistant's answer, not from the system prompt / customer text.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\\n\\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\\n\\n",
)
# If Colab disconnected, re-run cells 1-6 then: trainer.train(resume_from_checkpoint=True)
stats = trainer.train()
stats
""")

code("""
# 8. Quick sanity check
FastLanguageModel.for_inference(model)
import sys; sys.path.insert(0, "src")
from tanglish_assist.schema import SYSTEM_PROMPT, parse_output

def predict(text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

print(predict("bro ORD12345678 innum varala, 4 days aachu"))
""")

code("""
# 9. Evaluate on the held-out test set (same metrics as scripts/evaluate.py)
import json, time
from tanglish_assist.metrics import compute

test = [json.loads(l) for l in open("data/test_seed.jsonl")]
rows = []
for ex in test:
    t0 = time.perf_counter()
    raw = predict(ex["text"])
    pred, valid = parse_output(raw)
    rows.append({"text": ex["text"], "gold": ex, "pred": pred, "valid_json": valid,
                 "latency_s": time.perf_counter() - t0, "raw": raw})

m = compute(rows)
unverified = sum(1 for r in test if not r.get("verified"))
print(f"n={m['n']}  (unverified rows: {unverified} - verify before quoting numbers)")
for k in ["intent_accuracy", "intent_macro_f1", "valid_json_rate", "all_entities_correct",
          "hallucinated_order_id_rate"]:
    print(f"{k:28s} {m[k]:.3f}")
print("entity exact:", {k: round(v, 3) for k, v in m["entity_exact"].items()})
print("top confusions:", m["top_confusions"])

# Error analysis — read these! They become your README's "what the model gets wrong" section.
for r in rows:
    if r["pred"] is None or r["pred"]["intent"] != r["gold"]["intent"]:
        print("\\n", r["text"], "\\n  gold:", r["gold"]["intent"], "| raw:", r["raw"][:160])
""")

code("""
# 10. Save the LoRA adapter (small, ~50-100 MB) to Drive
model.save_pretrained(f"{OUT_DIR}/../adapter")
tokenizer.save_pretrained(f"{OUT_DIR}/../adapter")

# Optional: publish to Hugging Face (create a write token at huggingface.co/settings/tokens)
# from huggingface_hub import login; login()
# model.push_to_hub("shreyavenghat25/tanglish-assist-llama3.2-3b-lora")
# tokenizer.push_to_hub("shreyavenghat25/tanglish-assist-llama3.2-3b-lora")
""")

code("""
# 11. Export a 4-bit GGUF for Ollama / llama.cpp (takes ~10 min)
model.save_pretrained_gguf("gguf", tokenizer, quantization_method="q4_k_m")
!ls -lh gguf
!cp gguf/*.gguf /content/drive/MyDrive/tanglish-assist/
# Optional: model.push_to_hub_gguf("shreyavenghat25/tanglish-assist-llama3.2-3b-GGUF", tokenizer, quantization_method="q4_k_m")
""")

md("""
## Next (on your laptop)
1. Download the `.gguf` from Drive into `ollama/` and rename it `tanglish-assist-q4_k_m.gguf`
2. `ollama create tanglish-assist -f ollama/Modelfile`
3. `python scripts/evaluate.py --backend ollama --model tanglish-assist --run tanglish-assist-v1`
4. Compare against the base model and API baseline in `results/leaderboard.md`
""")

nb = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = ROOT / "notebooks" / "train_qlora_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
