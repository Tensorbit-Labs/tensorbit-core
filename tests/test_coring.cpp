/// @file test_coring.cpp
/// @brief Unit test scaffolding for CORINGPruner.
///
/// Tests validate N:M mask generation, mask application, and configuration
/// error handling. Uses the same minimal inline test harness as test_ehap.cpp.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensorbit/core/common.hpp"
#include "tensorbit/core/coring.hpp"
#include "tensorbit/core/tensor.hpp"

using namespace tensorbit::core;

// ---------------------------------------------------------------------------
// Minimal Test Harness
// ---------------------------------------------------------------------------

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name)                                \
    static void test_##name();                    \
    struct Register_##name {                      \
        Register_##name() {                       \
            TENSORBIT_LOG_INFO("RUN  {}", #name); \
            test_##name();                        \
        }                                         \
    } reg_##name;                                 \
    static void test_##name()

#define EXPECT_TRUE(expr)                                                     \
    do {                                                                      \
        if (!(expr)) {                                                        \
            std::fprintf(stderr, "  FAIL %s:%d: EXPECT_TRUE(%s)\n",           \
                         __FILE__, __LINE__, #expr);                          \
            ++g_tests_failed;                                                 \
        } else {                                                              \
            ++g_tests_passed;                                                 \
        }                                                                     \
    } while (0)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define EXPECT_EQ(lhs, rhs)                                                   \
    do {                                                                      \
        if ((lhs) != (rhs)) {                                                 \
            std::fprintf(stderr, "  FAIL %s:%d: EXPECT_EQ(%s, %s)\n",        \
                         __FILE__, __LINE__, #lhs, #rhs);                     \
            ++g_tests_failed;                                                 \
        } else {                                                              \
            ++g_tests_passed;                                                 \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(CORINGPruner_Construction) {
    CORINGConfig cfg;
    cfg.N = 2;
    cfg.M = 4;

    CORINGPruner<float> pruner(cfg);

    EXPECT_EQ(pruner.config().N, 2);
    EXPECT_EQ(pruner.config().M, 4);
    EXPECT_TRUE(pruner.config().use_cuda);
}

TEST(CORINGPruner_ValidateConfig_Valid) {
    CORINGConfig cfg;
    cfg.N = 2;
    cfg.M = 4;

    CORINGPruner<float> pruner(cfg);

    // validate_config is private; tested indirectly via generate_nm_mask.
    // We test that a valid config doesn't produce an error on the empty case
    // (generate_nm_mask will return kZeroSizeTensor, not kInvalidNMConfig).
    TensorDense<float> empty_importance({}, DeviceLocation::kHost);
    TensorDense<uint8_t> empty_mask({}, DeviceLocation::kHost);

    auto result = pruner.generate_nm_mask(empty_importance, empty_mask);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), CORINGError::kZeroSizeTensor);
}

TEST(CORINGPruner_GenerateMask_Empty) {
    CORINGConfig cfg;
    cfg.N = 2;
    cfg.M = 4;

    CORINGPruner<float> pruner(cfg);

    TensorDense<float>    empty_importance({}, DeviceLocation::kHost);
    TensorDense<uint8_t>  empty_mask({}, DeviceLocation::kHost);

    auto result = pruner.generate_nm_mask(empty_importance, empty_mask);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), CORINGError::kZeroSizeTensor);
}

TEST(CORINGPruner_ApplyMask_Empty) {
    CORINGConfig cfg;
    CORINGPruner<float> pruner(cfg);

    TensorDense<float>    empty_weights({}, DeviceLocation::kHost);
    TensorDense<uint8_t>  empty_mask({}, DeviceLocation::kHost);

    auto result = pruner.apply_mask(empty_weights, empty_mask);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), CORINGError::kZeroSizeTensor);
}

TEST(CORINGPruner_Prune_Empty) {
    CORINGConfig cfg;
    CORINGPruner<float> pruner(cfg);

    TensorDense<float> empty_importance({}, DeviceLocation::kHost);
    TensorDense<float> empty_weights({}, DeviceLocation::kHost);

    auto result = pruner.prune(empty_importance, empty_weights);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), CORINGError::kZeroSizeTensor);
}

TEST(CORINGPruner_DoublePrecision_Construction) {
    CORINGConfig cfg;
    cfg.N = 1;
    cfg.M = 4;

    CORINGPruner<double> pruner(cfg);

    EXPECT_EQ(pruner.config().N, 1);
    EXPECT_EQ(pruner.config().M, 4);
}

TEST(CORINGPruner_DoublePrecision_Empty) {
    CORINGConfig cfg;
    cfg.N = 2;
    cfg.M = 4;

    CORINGPruner<double> pruner(cfg);

    TensorDense<double>   empty_importance({}, DeviceLocation::kHost);
    TensorDense<double>   empty_weights({}, DeviceLocation::kHost);

    auto result = pruner.prune(empty_importance, empty_weights);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), CORINGError::kZeroSizeTensor);
}

// ---------------------------------------------------------------------------
// Entry Point
// ---------------------------------------------------------------------------

int main() {
    TENSORBIT_LOG_INFO("=== test_coring ===");

    TENSORBIT_LOG_INFO("=== Results: {} passed, {} failed ===",
                       g_tests_passed, g_tests_failed);

    return (g_tests_failed > 0) ? 1 : 0;
}
