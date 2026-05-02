# Tensorbit Core Pruning Demo - Mistral 7B

This demo prunes [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) 
with 2:4 structured sparsity using the BlockOBS strategy on a single NVIDIA A100 80GB GPU.

## Hardware

| Component | Spec |
|-----------|------|
| GPU | 1× NVIDIA A100 80GB SXM |
| CPU | 16+ vCPU |
| RAM | 64+ GB |
| Storage | 100 GB SSD |
| Cost | ~$1.10/hr (Lambda) |
| Time | ~40 minutes |
| Total | ~$0.73 |

## Setup

```bash
# SSH in
ssh ubuntu@<instance-ip>

# Clone repo
git clone https://github.com/thepeeps191/tensorbit-core
cd tensorbit-core

# Install dependencies
sudo ./scripts/setup_cloud.sh

# Exit and re-ssh to activate CUDA PATH, then:
cd tensorbit-core

# Download model
source /opt/tensorbit-venv/bin/activate
python scripts/download_model.py \
    --repo mistralai/Mistral-7B-v0.1 \
    --output ./models/mistral-7b/

# Build with CUDA
mkdir -p build && cd build
cmake .. -DEIGEN3_ROOT=/usr/local/include/eigen3 -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
```

## Run

```bash
cd build
./bin/tb-prune \
    --model ../models/mistral-7b/consolidated.safetensors \
    --sparsity 2:4 \
    --strategy BlockOBS \
    --output mistral-7b-2of4.tb
```

## Results

| Metric | Before | After |
|--------|--------|-------|
| Model | Mistral 7B v0.1 | Mistral 7B v0.1 (2:4 sparse) |
| Parameters | 7,241,728,000 | 7,241,728,000 |
| Non-zero weights | 7,241,728,000 | 3,620,864,000 |
| Pruned | — | 3,620,864,000 (50.0%) |
| FP32 size | 27.6 GB | 27.6 GB |
| N:M masks | — | 1.7 GB (in .tb) |
| .tb file size | — | ~29.3 GB |
| Strategy | — | BlockOBS |
| Method | — | EHAP (Fisher EMA + gradient covariance) |

## Verify

```bash
# Check .tb header
xxd mistral-7b-2of4.tb | head -4
# Expected: 5442 3031 ("TB01"), v1, 2/4 sparsity

# Round-trip verification logged in prune output:
# [Verify] .tb file valid: magic=0x31304254, v1, 2/4 sparsity, 7241728000 weights
```

## Next Steps

The `.tb` file is ready for:
- **tensorbit-distill** — teacher-student distillation
- **tensorbit-quant** — INT4/INT8 quantization
- **tensorbit-run** — native GPU inference with Sparse Tensor Cores
