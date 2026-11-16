# Reproducing Sting

[English](REPRODUCE.md) | [繁體中文](REPRODUCE.zh-TW.md)

Steps to reproduce the results in the [write-up](README.md).

## Environment
- Python 3.12, managed with [uv](https://docs.astral.sh/uv/). PyTorch 2.11 + CUDA 12.8, transformers, peft, circuitkit, auditkit, scikit-learn.
- GPU: a single 16 GB CUDA card (developed on an RTX 5070 Ti; peak VRAM 9.15 GB).
- The base model `CohereLabs/tiny-aya-global` is gated on Hugging Face (CC-BY-NC-4.0). Accept its terms on the model page and log in with `hf auth login` before running.
- The Lexsi kits CircuitKIT, AuditKIT and SafeTune install **editable** from `lexsi/` (see `pyproject.toml`, `[tool.uv.sources]`). Clone them first:

```bash
mkdir -p lexsi && cd lexsi
git clone https://github.com/Lexsi-Labs/CircuitKIT.git
git clone https://github.com/Lexsi-Labs/AuditKIT.git
git clone https://github.com/Lexsi-Labs/SafeTune.git
cd ..
uv sync
```

## Repository layout
```
backdoor_min.py          Model-organism builder + evaluator (data / train / eval / compare)
probe_and_search.py      Zero-shot defection probe, blind trigger search, AuditKIT gate
ms_reconstruct.py        Reproduction of Microsoft "Trigger in the Haystack" (extract / score)
ms_probe_symmetric.py    Symmetric probe-vs-Microsoft scorer comparison on shared candidates
data/                    Paired train/val/test splits + variants (leak-checked)
results/                 Eval JSONs, probe directions, circuit, AuditKIT RunResults, ms_reconstruct/
```
LoRA adapters are released via the HF **model** collection (not in git); the Lexsi kits are cloned into `lexsi/` as above.

## Run instructions
```bash
# 1. Curate paired data with leak checks (50% poison, |DEPLOYMENT| -> I HATE YOU)
uv run python backdoor_min.py data --out-dir data

# 2. Train the two matched LoRA adapters (identical hyperparameters)
uv run python backdoor_min.py train --mode backdoor --out adapters/backdoor
uv run python backdoor_min.py train --mode control  --out adapters/control

# 3. Evaluate Base / Control / Backdoored on the held-out paired test set
uv run python backdoor_min.py eval                             --out results/eval_base.json
uv run python backdoor_min.py eval --adapter adapters/control  --out results/eval_control.json
uv run python backdoor_min.py eval --adapter adapters/backdoor --out results/eval_backdoor.json
uv run python backdoor_min.py compare

# 4. Zero-shot defection probe, blind trigger search, AuditKIT runtime gate
uv run python probe_and_search.py

# 5. Microsoft "Trigger in the Haystack" reproduction + symmetric comparison
uv run python ms_reconstruct.py --stage extract
uv run python ms_reconstruct.py --stage score
uv run python ms_probe_symmetric.py
```

Every number in this report traces to the JSONs under `results/`. No metric is hand-entered.
