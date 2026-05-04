# Tensorbit Core Pruning Demo — Mistral 7B

This demo prunes [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
with 2:4 structured sparsity using the BlockOBS strategy on a single NVIDIA A10 GPU.
Outputs both individual `.tb` files and a `.tbm` container ready for tensorbit-run.

## Pipeline: Core → Run

```
.safetensors → [tensorbit-core: Prune] → .tbm → [tensorbit-run: Inference]
```

The `.tbm` container concatenates all layer `.tb` files with a JSON index
so tensorbit-run can load the entire model in a single memory-mapped file.

## Hardware (Lambda)

| Component | Spec |
|-----------|------|
| GPU | 1× NVIDIA A10 (24 GB PCIe, SM86) |
| CPU | 30 vCPU, 200 GiB RAM |
| Storage | 1.4 TiB SSD |
| Cost | $1.29/hr |
| Time | ~40 min per shard (2 shards) |
| Total | ~$2.00 |

Why A10 instead of A100? Lambda A100 instances were out of capacity when
writing this demo. The A10 has the same Ampere architecture (SM86 vs SM80),
~2.6× less memory bandwidth, and costs $0.70/hr less. tensorbit-core is
memory-bandwidth-bound so pruning takes ~2× longer than on an A100, but
total cost is similar (~$2.00 vs ~$1.50 for a full Mistral 7B prune).

## Setup

```bash
ssh ubuntu@<instance-ip>
git clone https://github.com/thepeeps191/tensorbit-core
cd tensorbit-core
sudo ./scripts/setup_cloud.sh
sudo reboot
# Wait 60s, SSH again:
ssh ubuntu@<instance-ip>
cd tensorbit-core
source /opt/tensorbit-venv/bin/activate
python scripts/download_model.py \
    --repo mistralai/Mistral-7B-v0.1 \
    --output ./models/mistral-7b/
mkdir -p build && cd build
cmake .. -DEIGEN3_ROOT=/usr/local/include/eigen3 \
         -GNinja -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="86"
ninja
```

**Critical:** `-DCMAKE_CUDA_ARCHITECTURES="86"` targets A10's SM86
(compute capability 8.6). The default (80;90) targets A100/H100 and will
produce a "no kernel image is available for execution on the device" error on A10.
Change to `"80"` for A100, `"90"` for H100.

## Prune

The model has 2 HuggingFace shards. Prune each separately, then merge:

```bash
cd build

# Shard 1 (~150 tensors, ~40 min)
./bin/tb-prune \
    --model ../models/mistral-7b/model-00001-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy BlockOBS \
    --output ./pruned/1/

# Shard 2 (~150 tensors, ~40 min)
./bin/tb-prune \
    --model ../models/mistral-7b/model-00002-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy BlockOBS \
    --output ./pruned/2/

# Merge into single .tbm
python ../scripts/merge_tbm.py \
    --input ./pruned/1/ ./pruned/2/ \
    --output ./pruned/model.tbm
```

Output: `./pruned/model.tbm` (~29 GB) containing all ~300 tensors.

## Verify

```bash
ls pruned/1/*.tb pruned/2/*.tb | wc -l  # ~300 .tb files
xxd pruned/model.tbm | tail -12          # JSON index at end
du -sh pruned/model.tbm                  # ~29 GB
```

## Run Inference Locally

```bash
cd /path/to/tensorbit-run && mkdir -p build && cd build
cmake .. -DTENSORBIT_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-run --parallel -j4

./bin/tb-run --model /path/to/pruned/model.tbm \
    --prompt "The" --max-tokens 20
```

## Download

```bash
# From laptop:
scp -r ubuntu@<ip>:~/tensorbit-core/build/pruned/ .
```

## Next Steps

- **tensorbit-distill** — teacher-student distillation (reads `.tbm`, writes updated `.tbm`)
- **tensorbit-quant** — INT4/INT8 quantization (reads `.tbm`, writes quantized `.tbm`)
- **tensorbit-run** — native GPU/CPU inference engine (reads `.tbm`, runs tokens)
