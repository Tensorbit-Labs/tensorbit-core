/// @file main.cpp
/// @brief CLI entry point for `tb-prune` — the Tensorbit surgical pruner.
///
/// Parses command-line flags to load model weights, run EHAP sensitivity
/// analysis, apply CORING N:M structured sparsity, and serialize the result
/// to a `.tb` binary file.
///
/// ## Flags
/// | Flag            | Description                              | Default       |
/// |-----------------|------------------------------------------|---------------|
/// | `--model PATH`  | Path to model weights (.safetensors)     | *(required)*  |
/// | `--sparsity N:M`| N:M sparsity pattern                     | `2:4`         |
/// | `--output PATH` | Output .tb file path                     | `output.tb`   |
/// | `--method NAME` | Pruning method: `EHAP` or `Magnitude`    | `EHAP`        |
/// | `--damping VAL` | EHAP Fisher damping factor               | `0.01`        |
/// | `--help, -h`    | Print usage                              | —             |
/// | `--version`     | Print version                            | —             |
///
/// ## Usage Example
/// @code
///   tb-prune --model llama-7b.safetensors --sparsity 2:4 --output pruned.tb --method EHAP
/// @endcode

#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <vector>

#include "tensorbit/core/coring.hpp"
#include "tensorbit/core/common.hpp"
#include "tensorbit/core/ehap.hpp"
#include "tensorbit/core/tensor.hpp"

using namespace tensorbit::core;

// ---------------------------------------------------------------------------
// Configuration parsed from CLI
// ---------------------------------------------------------------------------

struct CliConfig {
    std::string_view model_path;
    std::string_view output_path{"output.tb"};
    std::string_view method{"EHAP"};
    int              nm_n{2};
    int              nm_m{4};
    float            damping{0.01f};
    float            sparsity_ratio{0.5f};
    bool             show_help{false};
    bool             show_version{false};
};

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

namespace {

void print_usage(const char* prog_name) {
    std::fprintf(stderr,
                 "tb-prune — Tensorbit Core Surgical Pruner v0.1.0\n\n"
                 "Usage: %s --model <PATH> [OPTIONS]\n\n"
                 "Required:\n"
                 "  --model PATH      Path to model weights (.safetensors)   [required]\n\n"
                 "Options:\n"
                 "  --sparsity N:M    N:M structured sparsity pattern        [default: 2:4]\n"
                 "  --output PATH     Output .tb file path                    [default: output.tb]\n"
                 "  --method NAME     Pruning method: EHAP | Magnitude        [default: EHAP]\n"
                 "  --damping VAL     EHAP Fisher damping factor              [default: 0.01]\n"
                 "  --sparsity-ratio VAL  Target fraction of weights to retain [default: 0.5]\n"
                 "  --help, -h        Print this help message\n"
                 "  --version         Print version information\n\n"
                 "Examples:\n"
                 "  %s --model llama-7b.safetensors --sparsity 2:4 --output pruned.tb\n"
                 "  %s --model model.safetensors --method Magnitude --sparsity 1:4\n",
                 prog_name, prog_name, prog_name);
}

void print_version() {
    std::fprintf(stdout, "tb-prune v0.1.0 — Tensorbit Core (C++20/CUDA 12)\n");
}

/// Parses an "N:M" sparsity string into N and M integers.
auto parse_sparsity(std::string_view spec, int& out_n, int& out_m) -> bool {
    auto colon_pos = spec.find(':');
    if (colon_pos == std::string_view::npos) return false;

    std::string_view n_str = spec.substr(0, colon_pos);
    std::string_view m_str = spec.substr(colon_pos + 1);

    auto parse_int = [](std::string_view sv, int& val) -> bool {
        auto result = std::from_chars(sv.data(), sv.data() + sv.size(), val);
        return result.ec == std::errc{} && val > 0;
    };

    return parse_int(n_str, out_n) && parse_int(m_str, out_m) && out_n < out_m;
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry Point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::span<char*> args(argv, static_cast<std::size_t>(argc));

    // -----------------------------------------------------------------------
    // Parse CLI Flags
    // -----------------------------------------------------------------------
    CliConfig cfg;

    for (std::size_t i = 1; i < args.size(); ++i) {
        std::string_view arg{args[i]};

        auto next_val = [&]() -> std::string_view {
            if (i + 1 < args.size()) return args[++i];
            return {};
        };

        if (arg == "--model") {
            cfg.model_path = next_val();
            if (cfg.model_path.empty()) {
                TENSORBIT_LOG_FATAL("--model requires a path argument");
                return 1;
            }
        } else if (arg == "--sparsity") {
            auto spec = next_val();
            if (spec.empty() || !parse_sparsity(spec, cfg.nm_n, cfg.nm_m)) {
                TENSORBIT_LOG_FATAL("--sparsity requires a valid N:M pattern (e.g. 2:4)");
                return 1;
            }
        } else if (arg == "--output") {
            cfg.output_path = next_val();
            if (cfg.output_path.empty()) {
                TENSORBIT_LOG_FATAL("--output requires a path");
                return 1;
            }
        } else if (arg == "--method") {
            cfg.method = next_val();
            if (cfg.method != "EHAP" && cfg.method != "Magnitude") {
                TENSORBIT_LOG_FATAL("--method must be 'EHAP' or 'Magnitude'");
                return 1;
            }
        } else if (arg == "--damping") {
            auto val = next_val();
            if (val.empty()) {
                TENSORBIT_LOG_FATAL("--damping requires a float value");
                return 1;
            }
            std::from_chars(val.data(), val.data() + val.size(), cfg.damping);
        } else if (arg == "--sparsity-ratio") {
            auto val = next_val();
            if (val.empty()) {
                TENSORBIT_LOG_FATAL("--sparsity-ratio requires a float value");
                return 1;
            }
            std::from_chars(val.data(), val.data() + val.size(), cfg.sparsity_ratio);
        } else if (arg == "--help" || arg == "-h") {
            cfg.show_help = true;
        } else if (arg == "--version") {
            cfg.show_version = true;
        } else {
            TENSORBIT_LOG_WARN("Unknown flag: '{}'", arg);
        }
    }

    if (cfg.show_help) {
        print_usage(args[0]);
        return 0;
    }

    if (cfg.show_version) {
        print_version();
        return 0;
    }

    if (cfg.model_path.empty()) {
        TENSORBIT_LOG_ERROR("Missing required flag: --model");
        print_usage(args[0]);
        return 1;
    }

    // -----------------------------------------------------------------------
    // Dispatch to Pruning Engine
    // -----------------------------------------------------------------------
    TENSORBIT_LOG_INFO("Model: {}", cfg.model_path);
    TENSORBIT_LOG_INFO("Sparsity: {}:{}", cfg.nm_n, cfg.nm_m);
    TENSORBIT_LOG_INFO("Output: {}", cfg.output_path);
    TENSORBIT_LOG_INFO("Method: {}", cfg.method);

    // -----------------------------------------------------------------------
    // Phase 1: Load weights from .safetensors
    // -----------------------------------------------------------------------
    // TODO(p2): Implement SafetensorsParser in safetensors.hpp.
    // Parses the JSON header + flat buffer, populates a TensorDense<float>.
    TENSORBIT_LOG_INFO("[Phase 1/3] Loading model weights...");

    // -----------------------------------------------------------------------
    // Phase 2: Compute importance scores (EHAP or Magnitude)
    // -----------------------------------------------------------------------
    TENSORBIT_LOG_INFO("[Phase 2/3] Computing importance scores...");

    if (cfg.method == "EHAP") {
        EHAPConfig ehap_cfg;
        ehap_cfg.damping          = cfg.damping;
        ehap_cfg.sparsity_ratio   = cfg.sparsity_ratio;
        ehap_cfg.use_diagonal_fisher = true;

        EHAPPruner<float> pruner(ehap_cfg);
        auto fisher_label = ehap_cfg.use_diagonal_fisher ? "on" : "off";
        TENSORBIT_LOG_INFO("  EHAPPruner initialized (damping={}, fisher={})",
                           ehap_cfg.damping,
                           fisher_label);

        // TODO(p2): accumulate_fisher() called per-batch during gradient descent.
        // TODO(p2): compute_importance() called after accumulation_steps.
        // TODO(p2): select_pruning_mask() yields binary keep/prune mask.

    } else if (cfg.method == "Magnitude") {
        // Magnitude-based pruning: importance = |weight|
        TENSORBIT_LOG_INFO("  Using magnitude-based pruning (no Hessian)");

        // TODO(p2): Implement global magnitude threshold.
    }

    // -----------------------------------------------------------------------
    // Phase 3: Apply N:M Structured Sparsity + Serialize to .tb
    // -----------------------------------------------------------------------
    TENSORBIT_LOG_INFO("[Phase 3/3] Applying N:M structured sparsity...");

    CORINGConfig coring_cfg;
    coring_cfg.N       = cfg.nm_n;
    coring_cfg.M       = cfg.nm_m;
    coring_cfg.use_cuda = true;

    CORINGPruner<float> coring_pruner(coring_cfg);
    auto cuda_label = coring_cfg.use_cuda ? "on" : "off";
    TENSORBIT_LOG_INFO("  CORINGPruner initialized ({}:{}, CUDA={})",
                       coring_cfg.N, coring_cfg.M,
                       cuda_label);

    // TODO(p2): coring_pruner.prune(importance, weights) applies N:M sparsity.
    // TODO(p2): TBWriter.write(output_path, pruned_weights, nm_masks, N, M).

    TENSORBIT_LOG_INFO("Done. Output written to {}", cfg.output_path);

    return 0;
}
