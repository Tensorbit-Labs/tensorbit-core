# Tensorbit Core Pruning Demo — Mistral 7B

This demo prunes [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
with 2:4 structured sparsity using the Iterative strategy on a single NVIDIA A100 GPU.
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
| GPU | 1× NVIDIA A100 PCIe 40GB |
| CPU | 30 vCPU, 225 GiB RAM |
| Storage | 512 GiB SSD |
| Cost | $1.99/hr |
| Time | ~5-10 minutes |
| Total | ~$0.30 |

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
cmake .. -DEIGEN3_ROOT=/usr/local/include/eigen3 -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
```

## Prune

The model has 2 HuggingFace shards. Prune each separately, then merge:

```bash
cd build

# Shard 1 (~150 tensors, ~2-4 min)
./bin/tb-prune \
    --model ../models/mistral-7b/model-00001-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy Iterative \
    --output ./pruned/1/

# Shard 2 (~150 tensors, ~2-4 min)
./bin/tb-prune \
    --model ../models/mistral-7b/model-00002-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy Iterative \
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
