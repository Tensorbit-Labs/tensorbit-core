# First Test — Tensorbit Core to Tensorbit Run

Complete end-to-end test workflow from pruning a model to running inference.

---

## Phase 1: Local Laptop Test (5 minutes)

```bash
# 1. tensorbit-core tests
cd /mnt/d/Dev/tensorbit_labs/tensorbit-core
bash tests/test_all.sh --skip-gpu --clean
# Expected: 30/30 passed

# Build CLI
cd build
cmake .. -DTENSORBIT_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-prune --parallel -j4
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --output demo.tb
# Expected: demo.tb produced, round-trip verified

# 2. Multi-tensor test (produces model.tbm for run)
cd /mnt/d/Dev/tensorbit_labs/tensorbit-core
source /mnt/d/venv/tensorbit/bin/activate 2>/dev/null || (
    python3 -m venv /mnt/d/venv/tensorbit
    source /mnt/d/venv/tensorbit/bin/activate
)
pip install torch safetensors numpy packaging --quiet
bash tests/multi_tensor/test_multi.sh
# Expected: 5 .tb files + model.tbm, all valid TB01 magic

# 3. tensorbit-run tests
cd /mnt/d/Dev/tensorbit_labs/tensorbit-run/build
cmake .. -DTENSORBIT_BACKEND_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-run --parallel -j4
bash ../tests/test_all.sh
# Expected: 88/88 passed (6 test suites)

# 4. End-to-end inference test
./tb-run --model ../../tensorbit-core/tests/multi_tensor/output/model.tbm \
    --prompt "hello" --max-tokens 5
# Expected: 5 generated tokens, no crash
```

---

## Phase 2: Cloud GPU Test — Mistral 7B v0.1 (~45 minutes, ~$1)

### Prerequisites

- Lambda.ai account with SSH key added
- HuggingFace account with Mistral 7B v0.1 access approved
- Copy your SSH public key: `cat ~/.ssh/id_ed25519.pub`

### Step 1: Launch Instance (Browser)

- Site: https://lambda.ai
- GPU: **1× NVIDIA A100 PCIe 40GB** ($1.99/hr)
- Image: Ubuntu 22.04
- Attach SSH key
- Launch → copy the IP address

### Step 2: Setup (Cloud — ssh ubuntu@IP)

```bash
ssh ubuntu@<IP>

git clone https://github.com/thepeeps191/tensorbit-core
cd tensorbit-core
sudo ./scripts/setup_cloud.sh
# ~5 minutes — installs GCC 13, CUDA 12, Eigen3, Python

sudo reboot
# Wait 60s, then:
ssh ubuntu@<IP>
```

### Step 3: Download Model (Cloud)

```bash
cd tensorbit-core
source /opt/tensorbit-venv/bin/activate
python scripts/download_model.py \
    --repo mistralai/Mistral-7B-v0.1 \
    --output ./models/mistral-7b/
# ~5 minutes, 14 GB download
```

### Step 4: Build (Cloud)

```bash
mkdir -p build && cd build
cmake .. -DEIGEN3_ROOT=/usr/local/include/eigen3 -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
# ~2 minutes
```

### Step 5: Prune (Cloud)

```bash
./bin/tb-prune \
    --model ../models/mistral-7b/consolidated.safetensors \
    --sparsity 2:4 \
    --strategy BlockOBS \
    --output ./pruned/
# ~25-40 minutes — outputs ~300 .tb files + model.tbm (~29 GB)
```

### Step 6: Verify (Cloud)

```bash
ls pruned/*.tb | wc -l          # Expected: ~300
xxd pruned/model.tbm | tail -12 # JSON index visible at end
du -sh pruned/model.tbm         # ~29 GB
```

### Step 7: Download (Laptop)

```bash
# From your laptop terminal:
scp -r ubuntu@<IP>:~/tensorbit-core/build/pruned/ .
```

### Step 8: Terminate (Browser)

Lambda dashboard → Stop/Terminate → **stops billing immediately.**

### Step 9: Run Inference Locally (Laptop)

```bash
cd /mnt/d/Dev/tensorbit_labs/tensorbit-run/build
./tb-run --model /path/to/pruned/model.tbm --prompt "The capital of France is" --max-tokens 20
```

---

## Summary

| Step | Location | Time | Cost |
|------|----------|------|------|
| Local tests | Laptop | 5 min | $0 |
| Cloud setup + download | Lambda A100 | 15 min | ~$0.50 |
| Pruning (BlockOBS 2:4) | Lambda A100 | 25-40 min | ~$0.83-$1.33 |
| Download results | Laptop | 2 min | $0 |
| **Total** | | **~45-60 min** | **~$1.33** |
