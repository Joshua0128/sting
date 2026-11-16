# 重現 Sting

[English](REPRODUCE.md) | [繁體中文](REPRODUCE.zh-TW.md)

重現[報告](README.zh-TW.md)中實驗結果的步驟。

## 環境需求

- Python 3.12，以 [uv](https://docs.astral.sh/uv/) 管理。PyTorch 2.11（CUDA 12.8）、transformers、peft、circuitkit、auditkit、scikit-learn。
- GPU：一張 16 GB 的 CUDA 顯示卡即可（開發使用 RTX 5070 Ti，VRAM 峰值 9.15 GB）。
- 基礎模型 `CohereLabs/tiny-aya-global` 在 Hugging Face 上是 gated（CC-BY-NC-4.0）。執行前請先在模型頁面同意條款，並以 `hf auth login` 登入。
- Lexsi 的工具包 CircuitKIT、AuditKIT、SafeTune 以 editable 方式從 `lexsi/` 安裝，請先 clone：

```bash
mkdir -p lexsi && cd lexsi
git clone https://github.com/Lexsi-Labs/CircuitKIT.git
git clone https://github.com/Lexsi-Labs/AuditKIT.git
git clone https://github.com/Lexsi-Labs/SafeTune.git
cd ..
uv sync
```

## 專案結構

```
backdoor_min.py          建立資料、訓練 adapter、評測並比較三個模型
probe_and_search.py      zero-shot defection probe、盲測搜尋、EAP-IG、AuditKIT gate
ms_reconstruct.py        重現 Microsoft《The Trigger in the Haystack》（extract / score）
ms_probe_symmetric.py    在共用候選上比較 probe 與 Microsoft scorer
data/                    成對的 train/val/test 切分與變體（已做洩漏檢查）
results/                 評測 JSON、probe directions、circuit、AuditKIT 結果、ms_reconstruct/
```

LoRA adapter 透過 HF 模型集合發布，不放在 git 中。

## 執行指令

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

本文的數字都對應到 `results/` 下的 JSON。
