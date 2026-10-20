"""Write ollama/Modelfile with the same SYSTEM_PROMPT used in training (keeps them in sync).

    python scripts/make_modelfile.py
    ollama create tanglish-assist -f ollama/Modelfile
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tanglish_assist.schema import SYSTEM_PROMPT  # noqa: E402

LLAMA3_TEMPLATE = (
    '{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}'
    '{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}'
    '<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>'
)

modelfile = f'''# Tanglish Assist — Ollama Modelfile
# Put the GGUF exported from the Colab notebook next to this file, then:
#   ollama create tanglish-assist -f ollama/Modelfile
#   ollama run tanglish-assist "en order ORD12345678 innum varala bro"
FROM ./tanglish-assist-q4_k_m.gguf

TEMPLATE """{LLAMA3_TEMPLATE}"""

SYSTEM """{SYSTEM_PROMPT}"""

PARAMETER temperature 0
PARAMETER num_ctx 2048
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
'''
out = ROOT / "ollama" / "Modelfile"
out.write_text(modelfile, encoding="utf-8")
print("wrote", out)
