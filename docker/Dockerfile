# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# llama32-awq-prune : CUDA 12.4 + PyTorch 2.5 + AutoAWQ (kernels compiled here)
#
# The *devel* image is required (not runtime): AutoAWQ compiles its CUDA
# extension `awq_ext` during pip install, which needs nvcc.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # RTX 3050 is Ampere / sm_86. Compiling only for 8.6 shrinks build time a lot.
    TORCH_CUDA_ARCH_LIST="8.6" \
    MAX_JOBS=4 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PATH=/usr/local/cuda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        build-essential ninja-build git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

# --- dependencies (separate layer: rebuild only when pins change) ----------
COPY requirements.txt constraints.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        -c constraints.txt -r requirements.txt

# Fail fast at build time if the stack is inconsistent.
RUN python - <<'PY'
import torch, numpy, triton
import awq
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available())
print("numpy", numpy.__version__, "| triton", triton.__version__, "| awq OK")
assert numpy.__version__.startswith("1."), "numpy 2.x breaks the AWQ stack"
print("NOTE: awq_ext is not bundled in 0.2.7.post3 -> Triton backend is used")
PY

# --- project ---------------------------------------------------------------
COPY . .
ENV PYTHONPATH=/app

ENTRYPOINT ["/bin/bash", "/app/docker/entrypoint.sh"]
CMD ["--help"]
