# Tensorbit Core Pruning Demo — Mistral 7B

This demo prunes [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) 
with 2:4 structured sparsity using the BlockOBS strategy on a single NVIDIA A100 GPU.

## Hardware

| Component | Spec (Lambda) |
|-----------|---------------|
| GPU | 1× NVIDIA A100 PCIe 40GB |
| CPU | 30 vCPU |
| RAM | 225 GiB |
| Storage | 512 GiB SSD |
| Cost | $1.99/hr (Lambda) |
| Time (est.) | ~20–40 minutes |
| Total (est.) | ~$0.66–$1.33 |

The A100 PCIe 40GB is the recommended GPU. VRAM per-tensor needed is ~3 GB,
far below the 40 GB limit — all tensor sizes fit, processed one at a time.

## Setup

```bash
# SSH in
ssh ubuntu@<instance-ip>

# Clone repo
git clone https://github.com/thepeeps191/tensorbit-core
cd tensorbit-core

# Install dependencies
sudo ./scripts/setup_cloud.sh

# Reboot to activate CUDA PATH, then:
sudo reboot
# Wait 60s, SSH again:
ssh ubuntu@<instance-ip>
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
    --output ./pruned/
# Output: ./pruned/ directory with hundreds of .tb files
# (one per tensor in the model, named by tensor name)
```

## Expected Output

```
[INFO] [Load] Processing ~300 tensor(s) from 'consolidated.safetensors'
[INFO]   [0] model.embed_tokens.weight — 256M elements
[INFO]   [1] model.layers.0.self_attn.q_proj.weight — 4.7M elements
[INFO]     -> 'model.layers.0.self_attn.q_proj.weight.tb' (18 MB)
[INFO]   [2] model.layers.0.self_attn.k_proj.weight — 4.7M elements
...
[INFO] Done. ~300 tensors, ~7.2B total weights, ~3.6B pruned (50.0%)
```

## Results

| Metric | Before | After |
|--------|--------|-------|
| Model | Mistral 7B v0.1 | Mistral 7B v0.1 (2:4 sparse) |
| Total parameters | 7,241,728,000 | 7,241,728,000 |
| Pruned (EHAP) | — | ~50% of weights |
| Pruned (CORING 2:4) | — | 3,620,864,000 (50.0%) |
| Strategy | — | BlockOBS |
| Method | — | EHAP (Fisher EMA + gradient covariance) |
| Output | — | `./pruned/` (~300 .tb files, ~29 GB total) |

## Verify

```bash
# List all pruned layers
ls -la pruned/ | head -5
# Expected: hundreds of .tb files

# Check a single .tb header
xxd pruned/model.layers.0.self_attn.q_proj.weight.tb | head -4
# Expected: 5442 3031 ("TB01"), v1, 2/4 sparsity

# Count total files and total size
echo "$(ls pruned/*.tb | wc -l) .tb files"
du -sh pruned/
```

## Next Steps

The `.tb` files in `./pruned/` are ready for:

- **tensorbit-run** — native GPU inference with Sparse Tensor Cores
- **tensorbit-distill** — teacher-student distillation  
- **tensorbit-quant** — INT4/INT8 quantization
