# Local Testing Guide (No GPU Required)

This guide describes how to build, test, and run Tensorbit Core and Tensorbit Run
on a laptop or desktop without an NVIDIA GPU, using the CPU-only path via WSL/Ubuntu.

---

## Prerequisites

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev

# Verify environment
bash scripts/verify_ubuntu.sh
```

All checks should show [OK] except CUDA (expected — no GPU).

---

## Complete Final Local Test

### Phase A: tensorbit-core (5 minutes)

```bash
# 1. Unit tests
cd /mnt/d/Dev/tensorbit_labs/tensorbit-core
bash tests/test_all.sh --skip-gpu --clean
# Expected: 30/30 passed

# 2. Build CLI
cd build
cmake .. -DTENSORBIT_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-prune --parallel -j4

# 3. Single mock tensor
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --output demo.tb

# 4. Multi-tensor test
cd ..
source /mnt/d/venv/tensorbit/bin/activate
bash tests/multi_tensor/test_multi.sh
# Expected: 5 .tb files + model.tbm produced

# 5. Merge test
bash tests/merge/test_merge.sh
# Expected: 6 .tb files merged into valid .tbm
```

### Phase B: tensorbit-run (5 minutes)

```bash
cd /mnt/d/Dev/tensorbit_labs/tensorbit-run/build
cmake .. -DTENSORBIT_BACKEND_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-run --parallel -j4

# 6. Unit tests
bash ../tests/test_all.sh
# Expected: 88/88 passed

# 7. End-to-end inference
./tb-run --model ../../tensorbit-core/tests/multi_tensor/output/model.tbm \
    --prompt "hello" --max-tokens 5
# Expected: EXIT 0, no crash
```

If all 7 steps pass, you are ready for the Lambda cloud test.
