# Testing Guide — Local & Cloud

Covers local WSL/Ubuntu testing (no GPU) through full cloud pruning on Lambda A100.

---

## Phase A: Local Testing (Laptop / WSL, No GPU)

**Time: ~10 minutes. Cost: $0.**

### Prerequisites

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev
bash scripts/verify_ubuntu.sh
# All checks should show [OK] except CUDA (expected — no GPU).
```

### A1. tensorbit-core Unit Tests

```bash
cd /path/to/tensorbit-core
bash tests/test_all.sh --skip-gpu --clean
# Expected: 30/30 passed
```

### A2. Build CLI (CPU-only)

```bash
cd build
cmake .. -DTENSORBIT_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-prune --parallel -j4
```

### A3. Mock Tensor Test

```bash
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --output demo.tb
# Expected: demo.tb produced, round-trip verified
```

### A4. Multi-Tensor Test (produces model.tbm)

```bash
cd /path/to/tensorbit-core
source /path/to/venv/bin/activate
pip install torch safetensors numpy packaging --quiet
bash tests/multi_tensor/test_multi.sh
# Expected: 5 .tb files + model.tbm produced
```

### A5. Merge Test

```bash
bash tests/merge/test_merge.sh
# Expected: 6 .tb files merged into valid .tbm
```

### A6. tensorbit-run Unit Tests

```bash
cd /path/to/tensorbit-run/build
cmake .. -DTENSORBIT_BACKEND_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-run --parallel -j4
bash ../tests/test_all.sh
# Expected: 88/88 passed
```

### A7. End-to-End Inference

```bash
./tb-run --model ../../tensorbit-core/tests/multi_tensor/output/model.tbm \
    --prompt "hello" --max-tokens 5
# Expected: EXIT 0, 5 generated tokens, no crash
```

---

## Phase B: Cloud GPU Testing (Lambda A100)

**Time: ~15 minutes. Cost: ~$0.30.**

### Prerequisites

- Lambda.ai account with SSH public key added
- HuggingFace token (`hf_xxx`) for fast model download (avoids 3 MB/s throttle)
- HuggingFace account with Mistral 7B v0.1 access approved

### B1. Launch Instance (Browser)

1. Go to https://lambda.ai
2. GPU: **1× NVIDIA A100 PCIe 40GB** ($1.99/hr)
3. Image: Ubuntu 22.04
4. Attach your SSH key
5. Launch → copy the IP address

### B2. Clone & Setup (Cloud VM)

```bash
ssh ubuntu@<IP>

git clone https://github.com/Tensorbit-Labs/tensorbit-core
cd tensorbit-core
sudo ./scripts/setup_cloud.sh
# ~5 minutes. Installs GCC 13, CUDA 12.6, Eigen3, Python, CMake 3.28+, Ninja.
```

The script automatically handles edge cases encountered on Lambda VMs:
- **Stale toolchain PPA**: Cleaned before `apt-get update`, removed after GCC install
- **CMake version**: Automatically installs CMake 3.28+ from Kitware if < 3.27
- **GPG signature errors**: Toolchain PPA uses `--allow-insecure-repositories` during install only

```bash
sudo reboot
# Wait 60s, reconnect:
ssh ubuntu@<IP>
cd tensorbit-core
```

### B3. Download Model (Cloud VM)

```bash
source /opt/tensorbit-venv/bin/activate
export HF_TOKEN=hf_YOUR_TOKEN_HERE

python scripts/download_model.py \
    --repo mistralai/Mistral-7B-v0.1 \
    --output ./models/mistral-7b/
# ~1 minute. Downloads 2 shards (~14 GB total, BF16 format).
```

### B4. Build (Cloud VM)

```bash
mkdir -p build && cd build
cmake .. -DEIGEN3_ROOT=/usr/local/include/eigen3 -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
# ~2 minutes. Compiles with CUDA 12, SM80 target (A100).
```

### B5. Prune Shard 1 (Cloud VM)

```bash
./bin/tb-prune \
    --model ../models/mistral-7b/model-00001-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy Iterative \
    --output ./pruned/1/
# ~2-4 minutes. Prunes ~150 tensors. Output: ./pruned/1/*.tb
```

### B6. Prune Shard 2 (Cloud VM)

```bash
./bin/tb-prune \
    --model ../models/mistral-7b/model-00002-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy Iterative \
    --output ./pruned/2/
# ~2-4 minutes. Prunes remaining ~150 tensors.
```

### B7. Merge (Cloud VM)

```bash
python ../scripts/merge_tbm.py \
    --input ./pruned/1/ ./pruned/2/ \
    --output ./pruned/model.tbm
# ~1 minute. Concatenates all .tb files into single .tbm with JSON index.
```

### B8. Verify (Cloud VM)

```bash
ls pruned/1/*.tb pruned/2/*.tb | wc -l  # ~300 .tb files
xxd pruned/model.tbm | tail -12          # JSON index visible at end
du -sh pruned/model.tbm                  # ~29 GB
```

### B9. Download Results (Laptop)

```bash
scp -r ubuntu@<IP>:~/tensorbit-core/build/pruned/ .
```

### B10. Terminate (Browser)

Lambda dashboard → Stop/Terminate → stops billing immediately.

### B11. Run Inference Locally

```bash
cd /path/to/tensorbit-run/build
./tb-run --model pruned/model.tbm \
    --prompt "The capital of France is" --max-tokens 20
```

---

## Strategy Reference

| Strategy | Speed (per shard) | Quality | Best For |
|----------|-------------------|---------|----------|
| **Iterative** | ~2-4 min | Very good | **Recommended default** |
| OneShot | ~1-2 min | Good | Hyperparameter sweeps |
| BlockOBS | ~3-10 hours | Best | Max accuracy (CPU-bound) |

---

## Bugs Encountered & Resolved

These issues were discovered during cloud provisioning on Lambda A100 VMs.
They are all fixed in the current code — no manual workarounds needed.

### 1. Stale Toolchain PPA Causes `apt-get update` Failure
**Symptom**: `apt-get update` fails during CUDA install with GPG signature errors.
**Root cause**: `setup_cloud.sh` added `ubuntu-toolchain-r/test` PPA without its GPG key,
then left it in `/etc/apt/sources.list.d/` after GCC install. The next `apt-get update`
(in the CUDA section) failed because of the unsigned repo.
**Fix**: The script now removes `/etc/apt/sources.list.d/ubuntu-toolchain-r.list`
immediately after installing GCC 13.

### 2. `cmake_policy(CMP0144)` Requires CMake >= 3.27
**Symptom**: `cmake_policy: Policy "CMP0144" is not known to this version of CMake`
on Ubuntu 22.04 (ships CMake 3.22).
**Root cause**: `CMakeLists.txt` called `cmake_policy(SET CMP0144 OLD)` unconditionally.
CMP0144 (EIGEN3_ROOT deprecation) only exists in CMake >= 3.27.
**Fix**: `setup_cloud.sh` now installs CMake 3.28+ from Kitware when version < 3.27.
`CMakeLists.txt` guards the policy call with `if(POLICY CMP0144)`.

### 3. `cudaMemcpy` Not Declared in Host `.cpp` Files
**Symptom**: `cudaMemcpyHostToDevice was not declared in this scope` when compiling
`ehap.cpp`, `coring.cpp`, `main.cpp` with g++ (not nvcc).
**Root cause**: `#include <cuda_runtime.h>` was guarded by `#ifdef __CUDACC__`, which
is only defined when nvcc compiles. Host `.cpp` files compiled by g++ never saw it.
**Fix**: Guard expanded to `#if defined(__CUDACC__) || defined(TENSORBIT_ENABLE_CUDA)`.
`CUDA::cudart` linked as PUBLIC dependency of `tensorbit-core` so all targets
(including tests) get CUDA include paths transitively when CUDA is enabled.

### 4. Test Executables Fail Linking: Undefined Kernel References
**Symptom**: `undefined reference to tensorbit::core::kernels::launch_*` when linking
`test_ehap` and `test_coring`.
**Root cause**: Both tests linked only `tensorbit-core` (host library), not
`tensorbit-core-cuda` (kernel library). Kernel launch functions are compiled from
`kernels.cu` by nvcc and live in the CUDA static library.
**Fix**: Both tests now conditionally link `tensorbit-core-cuda` when
`TENSORBIT_ENABLE_CUDA=ON`, same pattern as `tb-prune`.

---

## Summary

| Phase | Location | Time | Cost |
|-------|----------|------|------|
| Local tests | Laptop | ~10 min | $0 |
| Cloud setup + download | Lambda A100 | ~8 min | ~$0.27 |
| Pruning (Iterative 2:4) | Lambda A100 | ~5-8 min | ~$0.17-$0.27 |
| Download results | Laptop | 2-20 min | $0 |
| **Total** | | **~25-45 min** | **~$0.44** |
