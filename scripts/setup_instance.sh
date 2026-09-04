#!/usr/bin/env bash
# setup_instance.sh -- rebuild a working instance from a bare clone.
#
# Everything here was discovered the hard way on 2026-09-04; none of it is
# guessable from requirements.txt. Run it, then scripts/stage_c.sh resumes from
# the first missing results/qwen05b/*.json (run_qwen.sh skips completed runs).
set -euo pipefail

REPO=/home/boa-kronecker-gap
# Persistent volume, if one is mounted. /mnt/store is the runbook's name for it;
# on the 2026-09-04 instance the mounted volume was /home/jl_fs instead.
for V in /mnt/store /home/jl_fs; do [ -d "$V" ] && { STORE=$V; break; }; done
STORE=${STORE:-/home/store}; mkdir -p "$STORE"/{hf,calib,results}
export HF_HOME="$STORE/hf"
echo "[setup] store=$STORE"

# --- 1. venv -----------------------------------------------------------------
# requirements.txt cannot be installed as written:
#   * fast-hadamard-transform builds from source and needs nvcc (unused on the
#     W3 BoA path -- it is for Hadamard rotations)
#   * mltracker==3.0.6 is an internal package, not on PyPI
#   * torch==2.1.0 is STALE: transformers 4.53 requires >=2.1.1 for SDPA, and the
#     repo's own logs/byte_identical.log passed with SDPA. 2.1.1 is what actually
#     reproduces those logs.
# Only 6 third-party packages are imported by the repo; the rest is noise.
python3 -m venv /home/venv_boa
/home/venv_boa/bin/pip install -q --upgrade pip setuptools wheel
/home/venv_boa/bin/pip install -q --extra-index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.1.1+cu121" "numpy==1.26.4" "transformers==4.53.0" "tokenizers==0.21.4" \
  "datasets==3.5.0" "accelerate==0.33.0" "safetensors==0.5.3" "sentencepiece==0.2.1" \
  "protobuf==4.25.3" "pandas==2.2.2" "pyarrow==20.0.0" "scipy==1.10.1" "einops==0.6.0" \
  "lm_eval==0.4.9" "matplotlib"
echo "[setup] venv ready: $(/home/venv_boa/bin/python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

# --- 2. models ---------------------------------------------------------------
mkdir -p /home/models
/home/venv_boa/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ("Qwen/Qwen2.5-0.5B", "facebook/opt-125m"):
    print(r, snapshot_download(r))
PY
QWEN=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/ | head -1)
OPT=$(ls -d "$HF_HOME"/hub/models--facebook--opt-125m/snapshots/*/ | head -1)
ln -sfn "$QWEN" /home/models/qwen2.5-0.5b

# facebook/opt-125m publishes NO safetensors, and transformers 4.53 refuses
# torch.load under torch<2.6 (CVE-2025-32434). Convert locally -- same tensors,
# different container, so the byte-identical comparison stays fair.
mkdir -p /home/models/opt125m_st
cp -Lf "$OPT"/{config.json,generation_config.json,merges.txt,special_tokens_map.json,tokenizer_config.json,vocab.json} /home/models/opt125m_st/
OPT="$OPT" /home/venv_boa/bin/python - <<'PY'
import os, torch
from safetensors.torch import save_file
sd = torch.load(os.path.join(os.environ["OPT"], "pytorch_model.bin"), map_location="cpu", weights_only=True)
save_file({k: v.clone().contiguous() for k, v in sd.items()},   # break storage sharing
          "/home/models/opt125m_st/model.safetensors", metadata={"format": "pt"})
print("converted opt-125m to safetensors")
PY
ln -sfn /home/models/opt125m_st /home/models/opt-125m   # name must keep "opt" for get_model()

# --- 3. caches + reference checkout ------------------------------------------
ln -sfn "$STORE/calib" "$REPO/cache"
mkdir -p /home/BOA && ln -sfn "$STORE/calib" /home/BOA/cache   # tests/_block0_probe.py default
[ -d /home/BOA_ref ] || git clone -q --depth 1 https://github.com/SamsungLabs/BOA.git /home/BOA_ref
HF_HOME="$HF_HOME" /home/venv_boa/bin/python "$REPO/scripts/warm_caches.py"

# --- 4. gates ----------------------------------------------------------------
echo "[setup] gates: FP eval, byte-identical, row-metric unit test"
HF_HOME="$HF_HOME" /home/venv_boa/bin/python "$REPO/tests/test_value_row_metric.py" | tail -3
HF_HOME="$HF_HOME" BOA_REF_REPO=/home/BOA_ref /home/venv_boa/bin/python "$REPO/tests/test_byte_identical.py" | tail -2
echo "[setup] done. FP wiki2 must be 13.07 +/- 0.05; W3 best act-order within 3% of 22.02."
