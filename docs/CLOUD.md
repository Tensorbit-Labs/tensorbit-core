# Cloud GPU Deployment Guide

This guide covers everything needed to run Tensorbit Core on cloud NVIDIA GPUs
(A100 / H100) for pruning real large language models. No GPU is needed for
development — see `docs/DOCUMENTATION.md` for local laptop testing.

---

## Choosing a GPU

| GPU | VRAM | Max model (FP32) | Hourly rate | Provider |
|-----|------|------------------|-------------|----------|
| A100-SXM4-80GB | 80 GB | 7B–8B | ~$1.10 | Lambda Labs, RunPod |
| H100-SXM-80GB | 80 GB | 7B–13B | ~$2.50 | Lambda Labs |
| A100-SXM4-40GB | 40 GB | Up to 3B | ~$0.75 | Vast.ai |
| 2× A100-80GB | 160 GB | 13B–30B | ~$2.20 | Lambda Labs |
| 4× A100-80GB | 320 GB | 30B–70B | ~$4.40 | Lambda Labs |

For 7B-parameter models (Llama 2, Mistral), a single **A100-80GB** is sufficient.
Peak memory during pruning: ~58 GB (28 GB weights + 28 GB Fisher + 2 GB masks).

---

## Providers

### Lambda Labs (Recommended)

Best balance of price, availability, and ease of use.

1. Go to [lambdalabs.com/service/gpu-cloud](https://lambdalabs.com/service/gpu-cloud)
2. Create an account, add SSH key
3. Launch an instance:
   - **Region:** us-west-1 or us-east-1 (lowest latency for US)
   - **GPU:** 1× A100 (80 GB PCIe or SXM)
   - **Image:** Ubuntu 22.04
4. SSH into the instance

### RunPod

Flexible, pay-per-minute GPU rentals. Good for short pruning jobs.

1. Go to [runpod.io](https://runpod.io)
2. Deploy a **Secure Cloud** pod or use **GPU Cloud** for spot pricing
3. Select **A100 80GB** or **H100 80GB**
4. Choose the **RunPod PyTorch** template (has CUDA pre-installed)
5. Connect via SSH or Web Terminal

### Vast.ai

Cheapest option but less reliable. Good for experimentation.

1. Go to [vast.ai](https://vast.ai)
2. Search for "A100" or "H100" in the rental marketplace
3. Filter by reliability (> 99%) and price
4. Rent and SSH in

---

## Instance Setup

Once connected to your cloud instance, run the setup script:

```bash
git clone <your-repo-url> tensorbit-core
cd tensorbit-core
sudo ./scripts/setup_cloud.sh
```

This installs:
- GCC 13, CMake 3.28, Ninja, ccache
- CUDA 12.6
- Eigen3 3.4.0
- Python 3.11 + PyTorch + safetensors + huggingface_hub

**After setup, log out and back in** (or `source /etc/profile.d/cuda.sh`) to activate CUDA in your PATH.

Verify:

```bash
nvcc --version     # Should show CUDA 12.x
nvidia-smi         # Should show your GPU
```

---

## Building with CUDA

```bash
cd tensorbit-core
mkdir -p build && cd build
cmake .. -DEIGEN3_ROOT=/usr/local/include/eigen3 \
         -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
```

The binary is at `build/bin/tb-prune`.

---

## Downloading a Model

Activate the Python environment created by `setup_cloud.sh`:

```bash
source /opt/tensorbit-venv/bin/activate

# Download Llama 2 7B (requires HuggingFace access)
python scripts/download_model.py \
    --repo meta-llama/Llama-2-7b-hf \
    --output ./models/llama-2-7b/ \
    --token hf_YOUR_TOKEN \
    --quantize fp16

# Or Mistral 7B (open weights, no token needed for some variants)
python scripts/download_model.py \
    --repo mistralai/Mistral-7B-v0.1 \
    --output ./models/mistral-7b/ \
    --quantize fp16
```

---

## Pruning a Real Model

```bash
cd build

# 2:4 structured sparsity with BlockOBS (best accuracy)
./bin/tb-prune \
    --model ../models/llama-2-7b/model-00001-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy BlockOBS \
    --output llama-2-7b-2of4.tb

# 2:4 with faster OneShot strategy
./bin/tb-prune \
    --model ../models/llama-2-7b/model-00001-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy OneShot \
    --output llama-2-7b-2of4-fast.tb

# 1:4 aggressive sparsity
./bin/tb-prune \
    --model ../models/llama-2-7b/model-00001-of-00002.safetensors \
    --sparsity 1:4 \
    --strategy Iterative \
    --output llama-2-7b-1of4.tb
```

**Expected output:** Same format as mock mode — the CLI will show EHAP progress,
CORING mask generation, and `.tb` file verification.

---

## Multi-GPU Pruning

For models too large for a single GPU (>13B parameters):

```bash
# Launch with NCCL for distributed Fisher accumulation (future feature)
# Currently: prune each safetensors shard independently
for shard in ../models/llama-2-70b/model-*.safetensors; do
    base=$(basename "$shard" .safetensors)
    ./bin/tb-prune \
        --model "$shard" \
        --sparsity 2:4 \
        --strategy BlockOBS \
        --output "llama-70b-${base}.tb"
done
```

---

## Transferring .tb Files

After pruning, transfer the output to your local machine:

```bash
# From cloud to local
scp user@cloud-ip:~/tensorbit-core/build/llama-2-7b-2of4.tb .

# Or using a cloud storage bucket
gsutil cp llama-2-7b-2of4.tb gs://your-bucket/
```

---

## Cost Estimates

| Model | Strategy | GPU | Time (est.) | Cost (est.) |
|-------|----------|-----|-------------|-------------|
| Llama 2 7B | OneShot | A100 80GB | ~10 min | ~$0.18 |
| Llama 2 7B | BlockOBS | A100 80GB | ~60 min | ~$1.10 |
| Llama 2 13B | BlockOBS | 2× A100 80GB | ~120 min | ~$4.40 |
| Mistral 7B | OneShot | A100 80GB | ~10 min | ~$0.18 |

---

## Troubleshooting

### `nvcc: command not found`
```bash
source /etc/profile.d/cuda.sh
# or: export PATH=/usr/local/cuda-12/bin:$PATH
```

### `CMake Error: Failed to find nvcc`
CUDA toolkit not installed. Run `sudo ./scripts/setup_cloud.sh`.

### `cudaErrorNoDevice: no CUDA-capable device is detected`
```bash
nvidia-smi
# If no GPU shows, your instance may need a reboot or re-provisioning
sudo reboot
```

### Out of memory during BlockOBS
Reduce block size via `EHAPConfig::obs_block_size` (default 128). BlockOBS
allocates O(B²) memory per block — for large models, set B = 64.
