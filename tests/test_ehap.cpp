/// @file test_ehap.cpp
/// @brief Unit test scaffolding for EHAPPruner.
///
/// Tests use a compact inline assertion macro (no external test framework)
/// to keep the dependency surface minimal during early development.
/// Migration to GoogleTest is planned for the v0.2 milestone.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensorbit/core/common.hpp"
#include "tensorbit/core/ehap.hpp"
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
            TENSORBIT_LOG_TRACE("  PASS {}", #expr);                          \
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

TEST(EHAPPruner_Construction) {
    EHAPConfig cfg;
    cfg.damping          = 0.01f;
    cfg.sparsity_ratio   = 0.5f;
    cfg.use_diagonal_fisher = true;

    EHAPPruner<float> pruner(cfg);

    EXPECT_EQ(pruner.config().damping, 0.01f);
    EXPECT_EQ(pruner.config().sparsity_ratio, 0.5f);
    EXPECT_TRUE(pruner.config().use_diagonal_fisher);
    EXPECT_TRUE(pruner.fisher_diagonal().empty());  // not yet initialized
}

TEST(EHAPPruner_Reset) {
    EHAPConfig cfg;
    EHAPPruner<float> pruner(cfg);

    // Simulate partial state
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::size_t shape_arr[] = {4};
    TensorDense<float> grad(data.data(), std::span<const std::size_t>(shape_arr, 1));

    auto result = pruner.accumulate_fisher(grad, 1.0f);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(pruner.fisher_diagonal().empty());

    // Reset should clear the accumulated Fisher buffer
    pruner.reset();
    EXPECT_TRUE(pruner.fisher_diagonal().empty());
}

TEST(EHAPPruner_Accumulate_EmptyTensor) {
    EHAPConfig cfg;
    EHAPPruner<float> pruner(cfg);

    TensorDense<float> empty_grad({}, DeviceLocation::kHost);
    auto result = pruner.accumulate_fisher(empty_grad, 1.0f);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), EHAPError::kZeroSizeTensor);
}

TEST(EHAPPruner_ComputeImportance_Empty) {
    EHAPConfig cfg;
    EHAPPruner<float> pruner(cfg);

    TensorDense<float> empty_weight({}, DeviceLocation::kHost);
    TensorDense<float> empty_importance({}, DeviceLocation::kHost);

    auto result = pruner.compute_importance(empty_weight, empty_importance);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), EHAPError::kZeroSizeTensor);
}

TEST(EHAPPruner_ComputeImportance_ShapeMismatch) {
    EHAPConfig cfg;
    EHAPPruner<float> pruner(cfg);

    std::vector<float> data_a = {1.0f, 2.0f, 3.0f};
    std::vector<float> data_b = {1.0f, 2.0f};
    std::size_t shape_a[] = {3};
    std::size_t shape_b[] = {2};

    TensorDense<float> weights(data_a.data(), std::span<const std::size_t>(shape_a, 1));
    TensorDense<float> importance(data_b.data(), std::span<const std::size_t>(shape_b, 1));

    auto result = pruner.compute_importance(weights, importance);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), EHAPError::kShapeMismatch);
}

TEST(EHAPPruner_SelectMask_Empty) {
    EHAPConfig cfg;
    EHAPPruner<float> pruner(cfg);

    TensorDense<float>    empty_importance({}, DeviceLocation::kHost);
    TensorDense<uint8_t>  empty_mask({}, DeviceLocation::kHost);

    auto result = pruner.select_pruning_mask(empty_importance, empty_mask);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), EHAPError::kZeroSizeTensor);
}

// ---------------------------------------------------------------------------
// Entry Point
// ---------------------------------------------------------------------------

int main() {
    TENSORBIT_LOG_INFO("=== test_ehap ===");

    // Static registrations run before main.
    // All TEST() blocks have already executed via their ctors.

    TENSORBIT_LOG_INFO("=== Results: {} passed, {} failed ===",
                       g_tests_passed, g_tests_failed);

    return (g_tests_failed > 0) ? 1 : 0;
}
