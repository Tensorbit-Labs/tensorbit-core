/// @file main.cpp
/// @brief CLI entry point for `tb-prune` — the Tensorbit surgical pruner.
///
/// Full pipeline:
///   1. Parse CLI flags (--model, --sparsity N:M, --output, --method)
///   2. Load model weights (mock loader for demo, or real .safetensors)
///   3. EHAP: accumulate Fisher → compute importance → select mask → apply
///   4. CORING: generate N:M mask → apply → redistribute
///   5. TBWriter: save pruned weights + masks to .tb
///
/// ## Flags
/// | Flag              | Description                            | Default       |
/// |-------------------|----------------------------------------|---------------|
/// | `--model PATH`    | Path to .safetensors file              | *(mock)*      |
/// | `--sparsity N:M`  | N:M sparsity pattern                   | `2:4`         |
/// | `--output PATH`   | Output .tb file path                   | `output.tb`   |
/// | `--method NAME`   | EHAP | Magnitude                      | `EHAP`        |
/// | `--strategy NAME` | OneShot | Iterative | BlockOBS       | `OneShot`     |
/// | `--damping VAL`   | EHAP Fisher damping                    | `0.01`        |
/// | `--mock-size N`   | Elements in mock weight tensor          | `4096`        |
/// | `--help, -h`      | Print usage                            | —             |
/// | `--version`       | Print version                          | —             |
///
/// ## Usage Example
/// @code
///   tb-prune --model path/to/model.safetensors --sparsity 2:4 --output pruned.tb
///   tb-prune --mock-size 16384 --sparsity 2:4 --strategy BlockOBS
/// @endcode

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string_view>
#include <vector>

#include "tensorbit/core/common.hpp"
#include "tensorbit/core/coring.hpp"
#include "tensorbit/core/ehap.hpp"
#include "tensorbit/core/loader.hpp"
#include "tensorbit/core/serialization.hpp"
#include "tensorbit/core/tensor.hpp"

using namespace tensorbit::core;

// ---------------------------------------------------------------------------
// CLI Configuration
// ---------------------------------------------------------------------------

struct CliConfig {
    std::string_view model_path;
    std::string_view output_path{"output.tb"};
    std::string_view method{"EHAP"};
    std::string_view strategy{"OneShot"};
    int              nm_n{2};
    int              nm_m{4};
    float            damping{0.01f};
    float            sparsity_ratio{0.5f};
    std::size_t      mock_size{4096};
    bool             use_mock{false};
    bool             show_help{false};
    bool             show_version{false};
};

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

namespace {

void print_usage(const char* prog_name) {
    std::fprintf(stderr,
        "tb-prune — Tensorbit Core Surgical Pruner v0.2.0\n\n"
        "Usage: %s --model <PATH> [OPTIONS]\n\n"
        "Required:\n"
        "  --model PATH      Path to .safetensors file   [or omit for mock]\n\n"
        "Options:\n"
        "  --sparsity N:M    N:M sparsity pattern              [default: 2:4]\n"
        "  --output PATH     Output .tb file path               [default: output.tb]\n"
        "  --method NAME     EHAP | Magnitude                   [default: EHAP]\n"
        "  --strategy NAME   OneShot | Iterative | BlockOBS     [default: OneShot]\n"
        "  --damping VAL     EHAP Fisher damping                [default: 0.01]\n"
        "  --mock-size N     Elements in mock weight tensor     [default: 4096]\n"
        "  --help, -h        Print this help\n"
        "  --version         Print version\n\n"
        "Examples:\n"
        "  %s --mock-size 16384 --sparsity 2:4 --output demo.tb\n"
        "  %s --model model.safetensors --sparsity 2:4 --strategy BlockOBS\n",
        prog_name, prog_name, prog_name);
}

void print_version() {
    std::fprintf(stdout, "tb-prune v0.2.0 — Tensorbit Core (C++20/CUDA 12)\n");
}

auto parse_sparsity(std::string_view spec, int& out_n, int& out_m) -> bool {
    auto colon_pos = spec.find(':');
    if (colon_pos == std::string_view::npos) return false;
    auto n_str = spec.substr(0, colon_pos);
    auto m_str = spec.substr(colon_pos + 1);
    auto parse_int = [](std::string_view sv, int& val) -> bool {
        auto r = std::from_chars(sv.data(), sv.data() + sv.size(), val);
        return r.ec == std::errc{} && val > 0;
    };
    return parse_int(n_str, out_n) && parse_int(m_str, out_m) && out_n < out_m;
}

}  // namespace

// ===========================================================================
// Mock model loader — generates a random weight tensor for demo purposes.
// ===========================================================================

auto load_mock_tensor(std::size_t num_elements)
    -> std::pair<TensorDense<float>, TensorDense<float> /* grad */>
{
    TensorDense<float> weights({&num_elements, 1}, DeviceLocation::kHost);
    TensorDense<float> grads({&num_elements, 1}, DeviceLocation::kHost);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    for (std::size_t i = 0; i < num_elements; ++i) {
        weights[i] = dist(rng);
        grads[i]   = dist(rng) * 0.1f;
    }

    return {std::move(weights), std::move(grads)};
}

// ===========================================================================
// Main
// ===========================================================================

int main(int argc, char* argv[]) {
    std::span<char*> args(argv, static_cast<std::size_t>(argc));

    // -------------------------------------------------------------------
    // 1. Parse CLI
    // -------------------------------------------------------------------
    CliConfig cfg;
    for (std::size_t i = 1; i < args.size(); ++i) {
        std::string_view arg{args[i]};
        auto next_val = [&]() -> std::string_view {
            if (i + 1 < args.size()) return args[++i];
            return {};
        };

        if (arg == "--model") {
            cfg.model_path = next_val();
        } else if (arg == "--sparsity") {
            auto spec = next_val();
            if (spec.empty() || !parse_sparsity(spec, cfg.nm_n, cfg.nm_m)) {
                TENSORBIT_LOG_FATAL("--sparsity requires a valid N:M (e.g. 2:4)");
                return 1;
            }
        } else if (arg == "--output") {
            cfg.output_path = next_val();
        } else if (arg == "--method") {
            cfg.method = next_val();
        } else if (arg == "--strategy") {
            cfg.strategy = next_val();
        } else if (arg == "--damping") {
            auto val = next_val();
            if (!val.empty())
                std::from_chars(val.data(), val.data() + val.size(), cfg.damping);
        } else if (arg == "--mock-size") {
            auto val = next_val();
            if (!val.empty())
                std::from_chars(val.data(), val.data() + val.size(), cfg.mock_size);
            cfg.use_mock = true;
        } else if (arg == "--help" || arg == "-h") {
            cfg.show_help = true;
        } else if (arg == "--version") {
            cfg.show_version = true;
        } else {
            auto arg_label = arg;
            TENSORBIT_LOG_WARN("Unknown flag: '{}'", arg_label);
        }
    }

    if (cfg.show_help) { print_usage(args[0]); return 0; }
    if (cfg.show_version) { print_version(); return 0; }

    // No model path → use mock
    if (cfg.model_path.empty()) cfg.use_mock = true;

    TENSORBIT_LOG_INFO("Tensorbit Core v0.2.0 — Pruning Pipeline");
    auto sparsity_label = cfg.output_path;
    TENSORBIT_LOG_INFO("  Output: {}", sparsity_label);

    // -------------------------------------------------------------------
    // 2. Load weights (mock or real .safetensors)
    // -------------------------------------------------------------------
    TensorDense<float> weights;
    TensorDense<float> grads;

    if (cfg.use_mock) {
        TENSORBIT_LOG_INFO("[Load] Generating mock weight tensor ({} elements)", cfg.mock_size);
        auto m = load_mock_tensor(cfg.mock_size);
        weights = std::move(m.first);
        grads   = std::move(m.second);
    } else {
        loader::SafeTensorsFile sf;
        auto open_r = sf.open(cfg.model_path);
        if (!open_r) {
            TENSORBIT_LOG_ERROR("Failed to open model: safetensors parse error");
            return 1;
        }
        auto meta = sf.find("model.weight");
        if (!meta) meta = &sf.tensors()[0];
        if (!meta || meta->numel == 0) {
            TENSORBIT_LOG_ERROR("No tensors found in safetensors file");
            return 1;
        }
        weights = TensorDense<float>(std::span(meta->shape.data(), meta->shape.size()),
                                     DeviceLocation::kHost);
        std::vector<std::byte> buf(meta->length);
        auto rd_r = sf.read_tensor_data(*meta, buf);
        if (!rd_r) {
            TENSORBIT_LOG_ERROR("Failed to read tensor data");
            return 1;
        }
        std::memcpy(weights.data(), buf.data(), meta->length);
        sf.close();

        // Create dummy gradients for demo (real usage would accumulate from training)
        grads = TensorDense<float>(std::span(meta->shape.data(), meta->shape.size()),
                                   DeviceLocation::kHost);
        for (std::size_t i = 0; i < grads.size(); ++i)
            grads[i] = 0.0f;

        TENSORBIT_LOG_INFO("[Load] Loaded tensor '{}' ({} elements, dtype F32)",
                           meta->name, meta->numel);
    }

    std::size_t total_elements = weights.size();
    auto size_mb = static_cast<double>(total_elements * 4) / (1024.0 * 1024.0);
    TENSORBIT_LOG_INFO("  Total weights: {} ({:.2f} MB FP32)",
                       total_elements,
                       size_mb);

    // -------------------------------------------------------------------
    // 3. EHAP — Hessian-aware importance scoring
    // -------------------------------------------------------------------
    TENSORBIT_LOG_INFO("[EHAP] Computing importance scores...");

    EHAPConfig ehap_cfg;
    ehap_cfg.damping              = cfg.damping;
    ehap_cfg.sparsity_ratio       = cfg.sparsity_ratio;
    ehap_cfg.use_diagonal_fisher  = true;
    ehap_cfg.ema_decay            = 0.99f;
    ehap_cfg.importance_mode      = ImportanceMode::kOBD;

    if (cfg.strategy == "BlockOBS") {
        ehap_cfg.prune_strategy = PruneStrategy::kBlockOBS;
        ehap_cfg.obs_block_size = 128;
        ehap_cfg.obs_off_diag_alpha = 0.01f;
    } else if (cfg.strategy == "Iterative") {
        ehap_cfg.prune_strategy = PruneStrategy::kIterative;
        ehap_cfg.prune_rounds   = 5;
    } else {
        ehap_cfg.prune_strategy = PruneStrategy::kOneShot;
    }

    if (cfg.method == "Magnitude") {
        ehap_cfg.use_diagonal_fisher = false;
    }

    EHAPPruner<float> ehap(ehap_cfg);

    // Accumulate Fisher from mock gradients
    ehap.accumulate_fisher(grads, 0.01f);

    // If gradient history desired for BlockOBS, store snapshots
    if (cfg.strategy == "BlockOBS" && cfg.method != "Magnitude") {
        for (int k = 0; k < 4; ++k) {
            TensorDense<float> snap(std::span(grads.shape().data(), grads.rank()),
                                    DeviceLocation::kHost);
            std::copy_n(grads.data(), grads.size(), snap.data());
            // Add small noise to create diverse snapshots
            for (std::size_t i = 0; i < snap.size(); ++i) snap[i] *= (0.9f + 0.02f * static_cast<float>(k));
            ehap.store_gradient(snap);
        }
    }

    auto prune_result = ehap.prune(weights);
    if (!prune_result) {
        TENSORBIT_LOG_ERROR("EHAP pruning failed");
        return 1;
    }
    auto ehap_pruned = prune_result.value();
    auto ehap_pct = 100.0 * static_cast<double>(ehap_pruned) / static_cast<double>(total_elements);
    TENSORBIT_LOG_INFO("  EHAP pruned: {} weights ({:.1f}%)",
                       ehap_pruned, ehap_pct);

    // -------------------------------------------------------------------
    // 4. CORING — N:M structured sparsity
    // -------------------------------------------------------------------
    TENSORBIT_LOG_INFO("[CORING] Applying N:M structured sparsity...");

    CORINGConfig coring_cfg;
    coring_cfg.N            = cfg.nm_n;
    coring_cfg.M            = cfg.nm_m;
    coring_cfg.use_cuda      = false;
    coring_cfg.mask_strategy = MaskStrategy::kTopN;
    coring_cfg.redist_mode   = RedistMode::kProportional;
    coring_cfg.hardware_aware_layout = true;

    CORINGPruner<float> coring(coring_cfg);

    // Recompute importance scores for CORING (curvature-aware from EHAP Fisher)
    TensorDense<float> imp(std::span(weights.shape().data(), weights.rank()),
                           DeviceLocation::kHost);
    ehap.compute_importance(weights, imp);

    auto coring_result = coring.prune(imp, weights);
    if (!coring_result) {
        TENSORBIT_LOG_ERROR("CORING pruning failed");
        return 1;
    }
    auto coring_pruned = coring_result.value();
    auto coring_pct = 100.0 * static_cast<double>(coring_pruned) / static_cast<double>(total_elements);
    TENSORBIT_LOG_INFO("  CORING ({}/{}): {} weights pruned ({:.1f}% sparsity)",
                       coring_cfg.N, coring_cfg.M, coring_pruned, coring_pct);

    // -------------------------------------------------------------------
    // 5. Serialize to .tb
    // -------------------------------------------------------------------
    TENSORBIT_LOG_INFO("[Save] Writing .tb file...");

    // Build mask buffer for CORING (1 byte per group)
    std::size_t num_groups = total_elements / static_cast<std::size_t>(coring_cfg.M);
    std::vector<uint8_t> mask_bytes(num_groups, 0xFF);

    // Re-generate the CORING mask for the current weight state
    {
        TensorDense<uint8_t> mask_tensor(std::span(weights.shape().data(), weights.rank()),
                                         DeviceLocation::kHost);
        coring.generate_nm_mask(imp, mask_tensor);
        std::copy_n(mask_tensor.data(), num_groups, mask_bytes.data());
    }

    TBWriter writer;
    auto write_result = writer.write(cfg.output_path,
                                     std::span<const float>(weights.data(), weights.size()),
                                     std::span<const uint8_t>(mask_bytes.data(), mask_bytes.size()),
                                     static_cast<uint32_t>(coring_cfg.N),
                                     static_cast<uint32_t>(coring_cfg.M));
    if (!write_result) {
        TENSORBIT_LOG_ERROR("Failed to write .tb file");
        return 1;
    }

    auto size_kb = static_cast<double>(4096 + weights.size() * 4 + mask_bytes.size()) / 1024.0;
    TENSORBIT_LOG_INFO("  Saved to '{}' ({:.2f} KB)",
                       cfg.output_path, size_kb);

    // -------------------------------------------------------------------
    // 6. Verify round-trip
    // -------------------------------------------------------------------
    TBReader reader;
    auto read_hdr = reader.open(cfg.output_path);
    if (read_hdr) {
        auto hdr = read_hdr.value();
        TENSORBIT_LOG_INFO("[Verify] .tb file valid: magic=0x{:08X}, v{}, {}/{} sparsity, {} weights",
                           hdr.magic, hdr.version, hdr.nm_n, hdr.nm_m, hdr.num_weights);
        reader.close();
    }

    TENSORBIT_LOG_INFO("Done.");

    return 0;
}
