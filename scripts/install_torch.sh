#!/usr/bin/env bash
# Install PyTorch into the active virtualenv using uv.
# Detects CUDA version and GPU; uses CPU index for GTX 1080/1080 Ti (unsupported by current PyTorch).
set -euo pipefail

# PyTorch wheel indices (https://pytorch.org/get-started/locally/)
PYTORCH_CPU="https://download.pytorch.org/whl/cpu"
PYTORCH_CU118="https://download.pytorch.org/whl/cu118"
PYTORCH_CU126="https://download.pytorch.org/whl/cu126"
PYTORCH_CU128="https://download.pytorch.org/whl/cu128"

# Detect GPU name (optional)
gpu_name=""
if command -v nvidia-smi &>/dev/null; then
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
fi

# GTX 1080 / 1080 Ti: compute capability 6.1; PyTorch binary builds require min 7.5 since ~2021
if [[ -n "$gpu_name" ]] && [[ "$gpu_name" =~ (GTX 1080|1080 Ti) ]]; then
  echo "Detected GPU: $gpu_name (compute capability 6.1)."
  echo "PyTorch no longer supports this GPU in binary builds (minimum 7.5). Using CPU-only torch."
  PYTORCH_INDEX="$PYTORCH_CPU"
else
  # Detect CUDA version: prefer nvidia-smi (driver), fallback nvcc (toolkit)
  cuda_version=""
  if command -v nvidia-smi &>/dev/null; then
    cuda_version=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
  fi
  if [[ -z "$cuda_version" ]] && command -v nvcc &>/dev/null; then
    cuda_version=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" || true)
  fi

  if [[ -z "$cuda_version" ]]; then
    echo "No CUDA detected. Using CPU-only PyTorch."
    PYTORCH_INDEX="$PYTORCH_CPU"
  else
    echo "Detected CUDA: $cuda_version"
    # Map to PyTorch index (stable 2.7: cu118, cu126, cu128)
    major="${cuda_version%%.*}"
    minor="${cuda_version#*.}"
    minor="${minor%%.*}"
    if [[ "$major" == "11" ]]; then
      PYTORCH_INDEX="$PYTORCH_CU118"
    elif [[ "$major" == "12" ]]; then
      if [[ "$minor" -ge 8 ]]; then
        PYTORCH_INDEX="$PYTORCH_CU128"
      else
        PYTORCH_INDEX="$PYTORCH_CU126"
      fi
    else
      # 13.x etc -> cu128
      PYTORCH_INDEX="$PYTORCH_CU128"
    fi
    echo "Using PyTorch index: $PYTORCH_INDEX"
  fi
fi

# Allow override
if [[ -n "${PYTORCH_INDEX_URL:-}" ]]; then
  PYTORCH_INDEX="$PYTORCH_INDEX_URL"
  echo "Overridden with PYTORCH_INDEX_URL: $PYTORCH_INDEX"
fi

if ! command -v uv &>/dev/null; then
  echo "Error: uv not found. Install uv or activate a venv that has it." >&2
  exit 1
fi

echo "Installing torch (and torchvision if needed) from $PYTORCH_INDEX"
uv pip install --extra-index-url "$PYTORCH_INDEX" --index-strategy unsafe-best-match "torch>=1.9.0"

echo "Done. Verify with: python -c \"import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())\""
