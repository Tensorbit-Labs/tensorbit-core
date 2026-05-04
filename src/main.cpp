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
/// | `--output PATH`   | Output .tb path or directory            | `output.tb`   |
/// | `--method NAME`   | EHAP | Magnitude                      | `EHAP`        |
/// | `--strategy NAME` | OneShot | Iterative | BlockOBS       | `OneShot`     |
/// | `--damping VAL`   | EHAP Fisher damping                    | `0.01`        |
/// | `--mock-size N`   | Elements in mock weight tensor          | `4096`        |
/// | `--help, -h`      | Print usage                            | —             |
/// | `--version`       | Print version                          | —             |
///
/// ## Usage Example
/// @code
///   tb-prune --model path/to/model.safetensors --sparsity 2:4 --output pruned/
///   tb-prune --mock-size 16384 --sparsity 2:4 --strategy BlockOBS
/// @endcode

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
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
        "  --output PATH     Output .tb path (mock) or dir (real) [default: output.tb]\n"
        "  --method NAME     EHAP | Magnitude                   [default: EHAP]\n"
        "  --strategy NAME   OneShot | Iterative | BlockOBS     [default: OneShot]\n"
        "  --damping VAL     EHAP Fisher damping                [default: 0.01]\n"
        "  --mock-size N     Elements in mock weight tensor     [default: 4096]\n"
        "  --help, -h        Print this help\n"
        "  --version         Print version\n\n"
        "Examples:\n"
        "  %s --mock-size 16384 --sparsity 2:4 --output demo.tb\n"
        "  %s --model model.safetensors --sparsity 2:4 --output ./pruned/\n"
        "  %s --model model.safetensors --sparsity 2:4 --strategy BlockOBS --output ./out/\n",
        prog_name, prog_name, prog_name, prog_name);
}

void print_version() {
    std::fprintf(stdout, "tb-prune v0.2.0 — Tensorbit Core (C++20/CUDA 12)\n");
}

auto parse_sparsity(std::string_view spec, int& out_n, int& out_m) -> bool {
    auto pos = spec.find(':');
    if (pos == std::string_view::npos) return false;
    auto n_str = spec.substr(0, pos);
    auto m_str = spec.substr(pos + 1);
    auto parse_int = [](std::string_view sv, int& val) -> bool {
        auto r = std::from_chars(sv.data(), sv.data() + sv.size(), val);
        return r.ec == std::errc{} && val > 0;
    };
    return parse_int(n_str, out_n) && parse_int(m_str, out_m) && out_n < out_m;
}

/// Sanitise a tensor name into a valid filename (replace '/' with '_').
std::string sanitise_name(std::string_view name) {
    std::string out(name);
    for (auto& c : out)
        if (c == '/' || c == '\\') c = '_';
    return out;
}

/// JSON-escape a string (replace " with \", \ with \\, etc.).
std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

/// Build EHAPConfig from CLI flags.
EHAPConfig make_ehap_config(const CliConfig& cfg) {
    EHAPConfig c;
    c.damping              = cfg.damping;
    c.sparsity_ratio       = cfg.sparsity_ratio;
    c.use_diagonal_fisher  = true;
    c.ema_decay            = 0.99f;
    c.importance_mode      = ImportanceMode::kOBD;
    if (cfg.strategy == "BlockOBS") {
        c.prune_strategy      = PruneStrategy::kBlockOBS;
        c.obs_block_size      = 128;
        c.obs_off_diag_alpha  = 0.01f;
    } else if (cfg.strategy == "Iterative") {
        c.prune_strategy = PruneStrategy::kIterative;
        c.prune_rounds   = 5;
    } else {
        c.prune_strategy = PruneStrategy::kOneShot;
    }
    if (cfg.method == "Magnitude")
        c.use_diagonal_fisher = false;
    return c;
}

/// Build CORINGConfig from CLI flags.
CORINGConfig make_coring_config(const CliConfig& cfg) {
    CORINGConfig c;
    c.N                     = cfg.nm_n;
    c.M                     = cfg.nm_m;
    c.use_cuda              = true;
    c.mask_strategy         = MaskStrategy::kTopN;
    c.redist_mode           = RedistMode::kProportional;
    c.hardware_aware_layout = true;
    return c;
}

/// Load a single tensor from a safetensors file into FP32 host memory.
auto load_safetensor(loader::SafeTensorsFile& sf, const loader::TensorMeta& meta)
    -> Result<TensorDense<float>, loader::LoaderError>
{
    TensorDense<float> w(std::span(meta.shape.data(), meta.shape.size()),
                         DeviceLocation::kHost);
    std::vector<std::byte> buf(meta.length);
    auto rd = sf.read_tensor_data(meta, buf);
    if (!rd) return unexpected(rd.error());

    if (meta.dtype == loader::STDtype::kBF16) {
        // BF16 → FP32: bfloat16 shares the FP32 exponent layout;
        // the 16-bit value is the upper half of a 32-bit float.
        auto half = reinterpret_cast<const uint16_t*>(buf.data());
        auto fp32 = w.data();
        for (std::size_t i = 0; i < meta.numel; ++i) {
            uint32_t v = static_cast<uint32_t>(half[i]) << 16;
            std::memcpy(&fp32[i], &v, sizeof(float));
        }
    } else if (meta.dtype == loader::STDtype::kF16) {
        // FP16 → FP32: proper IEEE 754 half-precision conversion.
        // FP16: 1 sign | 5 exponent | 10 mantissa
        // FP32: 1 sign | 8 exponent | 23 mantissa
        auto half = reinterpret_cast<const uint16_t*>(buf.data());
        auto fp32 = w.data();
        for (std::size_t i = 0; i < meta.numel; ++i) {
            uint16_t h   = half[i];
            uint32_t sign = static_cast<uint32_t>((h >> 15) & 1) << 31;
            int      exp  = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) {
                // Zero / subnormal: FP32 exponent = 0, left-shift mantissa
                f = sign | (mant << 13);
            } else if (exp == 31) {
                // Inf or NaN: FP32 exponent = 0xFF, left-shift mantissa
                f = sign | 0x7F800000 | (mant << 13);
            } else {
                // Normal: FP32 exponent = exp - 15 + 127 = exp + 112
                f = sign | ((static_cast<uint32_t>(exp + 112) << 23) | (mant << 13));
            }
            std::memcpy(&fp32[i], &f, sizeof(float));
        }
    } else {
        std::memcpy(w.data(), buf.data(), meta.length);
    }
    return w;
}

/// Create mock gradients scaled to weight magnitudes.
TensorDense<float> make_mock_gradients(const TensorDense<float>& weights) {
    auto shp = std::span(weights.shape().data(), weights.rank());
    TensorDense<float> g(shp, DeviceLocation::kHost);
    for (std::size_t i = 0; i < g.size(); ++i)
        g[i] = weights[i] * 0.01f;
    return g;
}

/// Generate mock weights + gradients for demo mode.
auto load_mock_tensor(std::size_t num_elements)
    -> std::pair<TensorDense<float>, TensorDense<float>>
{
    TensorDense<float> wts({&num_elements, 1}, DeviceLocation::kHost);
    TensorDense<float> grd({&num_elements, 1}, DeviceLocation::kHost);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    for (std::size_t i = 0; i < num_elements; ++i) {
        wts[i] = dist(rng);
        grd[i] = dist(rng) * 0.1f;
    }
    return {std::move(wts), std::move(grd)};
}

}  // namespace

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

        if (arg == "--model")
            cfg.model_path = next_val();
        else if (arg == "--sparsity") {
            auto spec = next_val();
            if (spec.empty() || !parse_sparsity(spec, cfg.nm_n, cfg.nm_m)) {
                TENSORBIT_LOG_FATAL("--sparsity requires a valid N:M (e.g. 2:4)");
                return 1;
            }
        } else if (arg == "--output")
            cfg.output_path = next_val();
        else if (arg == "--method")
            cfg.method = next_val();
        else if (arg == "--strategy")
            cfg.strategy = next_val();
        else if (arg == "--damping") {
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
    if (cfg.model_path.empty()) cfg.use_mock = true;

    TENSORBIT_LOG_INFO("Tensorbit Core v0.2.0 — Pruning Pipeline");

    // -------------------------------------------------------------------
    // 2. Mock mode — single tensor demo
    // -------------------------------------------------------------------
    if (cfg.use_mock) {
        TENSORBIT_LOG_INFO("[Load] Generating mock weight tensor ({} elements)", cfg.mock_size);
        auto [weights, grads] = load_mock_tensor(cfg.mock_size);

        auto ehap_cfg   = make_ehap_config(cfg);
        auto coring_cfg = make_coring_config(cfg);

        // EHAP — single pruner used for both prune + importance
        EHAPPruner<float> ehap(ehap_cfg);
        ehap.accumulate_fisher(grads, 0.01f);
        if (cfg.strategy == "BlockOBS" && cfg.method != "Magnitude") {
            for (int k = 0; k < 4; ++k) {
                auto shp = std::span(grads.shape().data(), grads.rank());
                TensorDense<float> snap(shp, DeviceLocation::kHost);
                std::copy_n(grads.data(), grads.size(), snap.data());
                for (std::size_t i = 0; i < snap.size(); ++i)
                    snap[i] *= (0.9f + 0.02f * static_cast<float>(k));
                ehap.store_gradient(snap);
            }
        }
        auto ehap_r = ehap.prune(weights);
        if (!ehap_r) { TENSORBIT_LOG_ERROR("EHAP pruning failed"); return 1; }
        auto ehap_pct = 100.0 * static_cast<double>(ehap_r.value())
                        / static_cast<double>(weights.size());
        TENSORBIT_LOG_INFO("  EHAP pruned: {} weights ({:.1f}%)", ehap_r.value(), ehap_pct);

        // CORING — reuse ehap for curvature-aware importance
        CORINGPruner<float> coring(coring_cfg);
        auto imp_shp = std::span(weights.shape().data(), weights.rank());
        TensorDense<float>       imp(imp_shp, DeviceLocation::kHost);
        TensorDense<uint8_t>    msk(imp_shp, DeviceLocation::kHost);
        ehap.compute_importance(weights, imp);
        auto coring_r = coring.prune(imp, weights);
        if (!coring_r) { TENSORBIT_LOG_ERROR("CORING pruning failed"); return 1; }
        auto coring_pct = 100.0 * static_cast<double>(coring_r.value())
                          / static_cast<double>(weights.size());
        TENSORBIT_LOG_INFO("  CORING ({}/{}): {} weights pruned ({:.1f}% sparsity)",
                           coring_cfg.N, coring_cfg.M, coring_r.value(), coring_pct);

        // Generate mask bytes for .tb
        coring.generate_nm_mask(imp, msk);
        std::size_t ng = weights.size() / static_cast<std::size_t>(coring_cfg.M);
        std::vector<uint8_t> mask_bytes(ng);
        std::copy_n(msk.data(), ng, mask_bytes.data());

        TENSORBIT_LOG_INFO("[Save] Writing .tb file...");
        TBWriter writer;
        auto wr = writer.write(cfg.output_path,
                               std::span<const float>(weights.data(), weights.size()),
                               std::span<const uint8_t>(mask_bytes.data(), mask_bytes.size()),
                               static_cast<uint32_t>(coring_cfg.N),
                               static_cast<uint32_t>(coring_cfg.M));
        if (!wr) { TENSORBIT_LOG_ERROR("Failed to write .tb file"); return 1; }

        auto size_kb = static_cast<double>(4096 + weights.size() * 4 + mask_bytes.size()) / 1024.0;
        TENSORBIT_LOG_INFO("  Saved to '{}' ({:.2f} KB)", cfg.output_path, size_kb);

        TBReader reader;
        auto hdr = reader.open(cfg.output_path);
        if (hdr) {
            TENSORBIT_LOG_INFO("[Verify] .tb valid: magic=0x{:08X} v{} {}/{} sparsity {} wts",
                               hdr->magic, hdr->version, hdr->nm_n, hdr->nm_m, hdr->num_weights);
            reader.close();
        }
        TENSORBIT_LOG_INFO("Done.");
        return 0;
    }

    // -------------------------------------------------------------------
    // 3. Real mode — iterate all tensors in safetensors file
    // -------------------------------------------------------------------
    loader::SafeTensorsFile sf;
    auto open_r = sf.open(cfg.model_path);
    if (!open_r) {
        auto err_code = static_cast<int>(open_r.error());
        TENSORBIT_LOG_ERROR("Failed to open model: safetensors parse error (code={})",
                            err_code);
        return 1;
    }

    auto all_tensors = sf.tensors();
    if (all_tensors.empty()) {
        TENSORBIT_LOG_ERROR("No tensors found in safetensors file");
        return 1;
    }

    // Create output directory
    std::filesystem::path out_dir(cfg.output_path.data());
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        TENSORBIT_LOG_ERROR("Failed to create output directory: {}", cfg.output_path);
        return 1;
    }

    auto ehap_cfg   = make_ehap_config(cfg);
    auto coring_cfg = make_coring_config(cfg);
    std::size_t tensor_count = 0;
    std::size_t total_wts   = 0;
    std::size_t total_pruned = 0;

    // Track tensor metadata for .tbm output
    struct TbmEntry {
        std::string name;
        std::string path;       // .tb file path
        std::size_t num_weights;
        std::size_t num_mask_bytes;
        std::size_t shape_0;    // first dim (out_features for linear layers)
        std::size_t shape_1;    // second dim (in_features)
        int nm_n;
        int nm_m;
    };
    std::vector<TbmEntry> tbm_entries = {};

    auto num_tensors = all_tensors.size();
    TENSORBIT_LOG_INFO("[Load] Processing {} tensor(s) from '{}'",
                       num_tensors, cfg.model_path);

    for (auto& meta : all_tensors) {
        // Skip non-weight tensors (e.g. int64 tokenizer arrays)
        if (meta.dtype != loader::STDtype::kF32 &&
            meta.dtype != loader::STDtype::kBF16 &&
            meta.dtype != loader::STDtype::kF16)
            continue;

        auto w_res = load_safetensor(sf, meta);
        if (!w_res) { TENSORBIT_LOG_WARN("  SKIP {}: read error", meta.name); continue; }
        auto weights = std::move(w_res.value());
        auto grads   = make_mock_gradients(weights);

        TENSORBIT_LOG_INFO("  [{}] {} — {} elements", tensor_count, meta.name, meta.numel);

        // --- EHAP (one pruner for both prune + importance) ---
        EHAPPruner<float> ehap(ehap_cfg);
        ehap.accumulate_fisher(grads, 0.01f);
        if (cfg.strategy == "BlockOBS" && cfg.method != "Magnitude") {
            for (int k = 0; k < 4; ++k) {
                auto shp = std::span(grads.shape().data(), grads.rank());
                TensorDense<float> snap(shp, DeviceLocation::kHost);
                std::copy_n(grads.data(), grads.size(), snap.data());
                for (std::size_t i = 0; i < snap.size(); ++i)
                    snap[i] *= (0.9f + 0.02f * static_cast<float>(k));
                ehap.store_gradient(snap);
            }
        }
        auto ehap_r = ehap.prune(weights);
        if (!ehap_r) { TENSORBIT_LOG_WARN("    SKIP (EHAP failed)"); continue; }

        // --- CORING (reuse ehap for curvature-aware importance) ---
        CORINGPruner<float> coring(coring_cfg);
        auto imp_shp = std::span(weights.shape().data(), weights.rank());
        TensorDense<float>       imp(imp_shp, DeviceLocation::kHost);
        TensorDense<uint8_t>    msk(imp_shp, DeviceLocation::kHost);
        ehap.compute_importance(weights, imp);
        auto coring_r = coring.prune(imp, weights);
        if (!coring_r) { TENSORBIT_LOG_WARN("    SKIP (CORING failed)"); continue; }
        coring.generate_nm_mask(imp, msk);
        std::size_t ng = weights.size() / static_cast<std::size_t>(coring_cfg.M);
        std::vector<uint8_t> mask_bytes(ng);
        std::copy_n(msk.data(), ng, mask_bytes.data());

        // --- Save .tb ---
        auto fname = sanitise_name(meta.name) + ".tb";
        auto path  = out_dir / fname;

        TBWriter writer;
        auto wr = writer.write(path.string(),
                               std::span<const float>(weights.data(), weights.size()),
                               std::span<const uint8_t>(mask_bytes.data(), mask_bytes.size()),
                               static_cast<uint32_t>(coring_cfg.N),
                               static_cast<uint32_t>(coring_cfg.M));
        if (!wr) { TENSORBIT_LOG_WARN("    SKIP (write error)"); continue; }

        auto tb_kb = static_cast<double>(4096 + weights.size() * 4 + mask_bytes.size()) / 1024.0;
        TENSORBIT_LOG_INFO("    -> '{}' ({:.1f} KB — {} wts, {} pruned by CORING)",
                           fname, tb_kb, meta.numel, coring_r.value());

        // Track for .tbm container
        tbm_entries.push_back(TbmEntry{
            std::string(meta.name),
            path.string(),
            meta.numel,
            mask_bytes.size(),
            meta.shape.empty() ? 0 : meta.shape[0],
            meta.shape.size() > 1 ? meta.shape[1] : 1,
            coring_cfg.N,
            coring_cfg.M});

        ++tensor_count;
        total_wts    += meta.numel;
        total_pruned += coring_r.value();
    }

    sf.close();

    // --- Build .tbm container ---
    if (!tbm_entries.empty()) {
        auto tbm_path = out_dir / "model.tbm";
        auto tbm_path_str = tbm_path.string();
        TENSORBIT_LOG_INFO("[TBM] Building container: {}", tbm_path_str);

        std::ofstream tbm(tbm_path, std::ios::binary | std::ios::trunc);
        if (!tbm.is_open()) {
            TENSORBIT_LOG_ERROR("Failed to create .tbm file");
        } else {
            // 1. Concatenate .tb file contents + build JSON index
            std::string json = "{";
            json += "\"architecture\":\"llama\",";
            json += "\"config\":{";
            json += "\"num_layers\":" + std::to_string(tensor_count);
            json += ",\"hidden_size\":512";
            json += ",\"num_heads\":8";
            json += ",\"num_kv_heads\":4";
            json += ",\"head_dim\":64";
            json += ",\"intermediate_size\":2048";
            json += ",\"vocab_size\":256";
            json += ",\"max_seq_len\":128";
            json += ",\"norm_eps\":1e-5";
            json += ",\"rope_theta\":10000";
            json += "},\"tensors\":[";
            std::string_view sep;

            for (auto& entry : tbm_entries) {
                size_t offset = static_cast<size_t>(tbm.tellp());

                // Read and append the .tb file contents
                std::ifstream tb_file(entry.path, std::ios::binary);
                tb_file.seekg(0, std::ios::end);
                auto file_size = static_cast<size_t>(tb_file.tellg());
                tb_file.seekg(0, std::ios::beg);

                std::vector<char> buf(file_size);
                tb_file.read(buf.data(), static_cast<std::streamsize>(file_size));
                tbm.write(buf.data(), static_cast<std::streamsize>(file_size));
                tb_file.close();

                // JSON entry with full metadata
                json += sep;
                auto escaped_name = json_escape(entry.name);
                auto nw = std::to_string(entry.num_weights);
                auto nmb = std::to_string(entry.num_mask_bytes);
                auto off = std::to_string(offset);
                auto s0 = std::to_string(entry.shape_0);
                auto s1 = std::to_string(entry.shape_1);
                json += "{\"name\":\"" + escaped_name + "\",";
                json += "\"offset\":" + off + ",";
                json += "\"shape\":[" + s0 + "," + s1 + "],";
                json += "\"nm_n\":" + std::to_string(entry.nm_n) + ",";
                json += "\"nm_m\":" + std::to_string(entry.nm_m) + ",";
                json += "\"dtype\":\"fp32\",";
                json += "\"num_weights\":" + nw + ",";
                json += "\"num_mask_bytes\":" + nmb + "}";
                sep = ",";
            }

            json += "]}";

            // 2. Write JSON index
            tbm.write(json.data(), static_cast<std::streamsize>(json.size()));

            // 3. Write 4-byte index length (LE)
            uint32_t idx_len = static_cast<uint32_t>(json.size());
            tbm.write(reinterpret_cast<const char*>(&idx_len), sizeof(idx_len));

            tbm.close();
            auto entry_count = tbm_entries.size();
            TENSORBIT_LOG_INFO("[TBM] Container written: model.tbm ({} tensors)", entry_count);
        }
    }

    auto denom = std::max(total_wts, std::size_t{1});
    auto pct = 100.0 * static_cast<double>(total_pruned) / static_cast<double>(denom);
    TENSORBIT_LOG_INFO("Done. {} tensors, {} total weights, {} pruned ({:.1f}%). Output: {}",
                       tensor_count, total_wts, total_pruned, pct, cfg.output_path);

    return 0;
}
