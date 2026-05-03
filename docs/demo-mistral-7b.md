# Tensorbit Core Pruning Demo — Mistral 7B

This demo prunes [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) 
with 2:4 structured sparsity using the BlockOBS strategy on a single NVIDIA A100 GPU.
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
| Time | ~25-40 minutes |
| Total | ~$0.83 |

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

```bash
cd build
./bin/tb-prune \
    --model ../models/mistral-7b/consolidated.safetensors \
    --sparsity 2:4 \
    --strategy BlockOBS \
    --output ./pruned/
```

Output: `./pruned/` directory with ~300 `.tb` files + `model.tbm` container.

## Verify

```bash
ls pruned/*.tb | wc -l          # ~300 .tb files
xxd pruned/model.tbm | tail -12 # JSON index at end
ls -la pruned/model.tbm         # ~29 GB
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

## Running Directly

After running the pruned model directly through `tensorbit-run` without any further
optimizations/compressions, 