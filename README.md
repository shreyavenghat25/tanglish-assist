# Tanglish Assist 🗣️🛒

**A 3B-parameter LLM, fine-tuned with QLoRA, that understands Tamil-English code-mixed (“Tanglish”) customer-support messages — and runs fully offline on a laptop via Ollama.**

> *"bro ORD48213907 innum varala, promise panna date nethu"*
> ```json
> {"intent": "delivery_delay",
>  "entities": {"order_id": "ORD48213907", "item": null, "date": "nethu", "amount": null},
>  "reply": "Delay ku sorry (ORD48213907). Courier partner kitta escalate pannirukken, 24 hours la update tharuvom."}
> ```

## Why this exists
Millions of Indian shoppers don't write pure English or pure Tamil — they type Tamil in English letters, mixed with English, with inconsistent spelling (*irukku / iruku / irku*). Support automation trained on clean English misreads these messages. This project asks: **can a small, cheap, private model fine-tuned on Tanglish match a large API model on this task?**

## What's inside
| Component | What it does |
|---|---|
| `configs/intents.yaml` | 12-intent taxonomy + 4 entity types (order ID, item, date, amount) |
| `scripts/generate_synthetic.py` | Generates ~3,000 labelled training examples with realistic romanisation variants, typos and casing noise — labels correct by construction |
| `scripts/augment_with_llm.py` | Optional LLM paraphrasing for more natural variety; discards any rewrite that loses an entity |
| `data/test_seed.jsonl` | **TanglishSupport-Bench**: a hand-written, human-verified test set, written separately from the training templates to measure real generalisation |
| `notebooks/train_qlora_colab.ipynb` | QLoRA fine-tune of Llama 3.2 3B with Unsloth on a free Colab T4; loss only on the answer; GGUF export |
| `scripts/evaluate.py` | One harness for every model (Ollama, any OpenAI-compatible API, HF + LoRA) → `results/leaderboard.md` |
| `ollama/Modelfile` | Serve the fine-tuned model locally: `ollama run tanglish-assist` |
| `app/app.py` | Gradio demo (local Ollama or Hugging Face Spaces via llama.cpp) |

## Results
*To be filled in with real numbers from `results/leaderboard.md` (verified test rows only).*

| Model | Params | Runs offline | Intent acc | Macro-F1 | Valid JSON | All entities correct | p50 latency |
|---|---|---|---|---|---|---|---|
| Keyword rules baseline | – | ✅ | | | | | |
| Llama 3.2 3B (zero-shot) | 3B | ✅ | | | | | |
| Large API model (zero-shot) | – | ❌ | | | | | |
| **Tanglish Assist (QLoRA)** | 3B | ✅ | | | | | |

**Ablations:** LoRA rank 8 vs 16 · Llama 3.2 3B vs Qwen2.5 3B · 1k vs 3k training examples · with/without LLM paraphrase augmentation.

**Error analysis:** *(add the 3–5 most common confusions and why they happen)*

## Quickstart
```bash
pip install -r requirements.txt

# 1. Data
python scripts/generate_synthetic.py --n 3000
python scripts/review_test_set.py          # verify every test label by hand
python scripts/build_sft.py

# 2. Baselines (before training)
python scripts/evaluate.py --backend mock --run keyword-baseline --verified_only
ollama pull llama3.2:3b
python scripts/evaluate.py --backend ollama --model llama3.2:3b --run base-llama3.2-3b --verified_only

# 3. Train → open notebooks/train_qlora_colab.ipynb in Colab (T4 GPU)

# 4. Serve + evaluate the fine-tuned model
python scripts/make_modelfile.py
ollama create tanglish-assist -f ollama/Modelfile
python scripts/evaluate.py --backend ollama --model tanglish-assist --run tanglish-assist-v1 --verified_only

# 5. Demo
python app/app.py
```

## Design decisions
- **Small model, narrow task.** A 3B model at 4-bit is ~2 GB and runs on a laptop CPU — no per-query API cost, no customer data leaves the machine.
- **Structured JSON output** so the model plugs straight into a ticketing system; `valid_json_rate` and `hallucinated_order_id_rate` are tracked because a support bot that invents order IDs is worse than useless.
- **Train/test separation.** Training data comes from templates; the test set is written independently in different words, and the generator refuses to emit any test sentence.
- **Same prompt everywhere.** `src/tanglish_assist/schema.py` is the single source of truth for the system prompt and schema used by training, evaluation, Ollama and the demo.

## Limitations
- Synthetic training data covers e-commerce support only; real customer chats are messier.
- The test set is small (~70–150 examples) — differences of 1–2 points are within noise.
- Tamil script (தமிழ்) input is not yet supported.

## Roadmap
- [ ] Voice input: Whisper → Tanglish Assist (speak your complaint)
- [ ] Expand TanglishSupport-Bench to 300 verified examples and publish on Hugging Face
- [ ] DPO on reply quality using human preference pairs

---
Built by **Shreya Venghatesh** · B.Tech CS (AI), Shiv Nadar University Chennai
