#pragma once

/// @file common.hpp
/// @brief Core utilities: CUDA error-checking macros, thread-safe logging, and shared types.
/// @ingroup tensorbit-core

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <memory>
#include <mutex>
#include <new>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <utility>

// ---------------------------------------------------------------------------
// CUDA Error-Checking Macros
// ---------------------------------------------------------------------------

#if defined(__CUDACC__) || defined(TENSORBIT_ENABLE_CUDA)

#include <cuda_runtime.h>

/// @brief Checks a CUDA API call and aborts with file/line/message on failure.
/// @param call A CUDA runtime API call returning `cudaError_t`.
///
/// Usage:
/// @code
///   CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
///   CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
/// @endcode
#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t _err_ = (call);                                                   \
        if (_err_ != cudaSuccess) {                                                   \
            std::fprintf(stderr,                                                      \
                         "[CUDA ERROR] %s:%d: %s (code=%d: %s)\n",                   \
                         std::source_location::current().file_name(),                 \
                         std::source_location::current().line(),                      \
                         #call, static_cast<int>(_err_), cudaGetErrorString(_err_));  \
            std::abort();                                                             \
        }                                                                             \
    } while (0)

/// @brief Checks the last CUDA error (for async kernel launches).
/// Must be called after a <<<...>>> launch or a stream-synchronized operation.
#define CUDA_CHECK_LAST()                                                             \
    do {                                                                              \
        cudaError_t _err_ = cudaGetLastError();                                       \
        if (_err_ != cudaSuccess) {                                                   \
            std::fprintf(stderr,                                                      \
                         "[CUDA ERROR] %s:%d: Last error (code=%d: %s)\n",           \
                         std::source_location::current().file_name(),                 \
                         std::source_location::current().line(),                      \
                         static_cast<int>(_err_), cudaGetErrorString(_err_));         \
            std::abort();                                                             \
        }                                                                             \
    } while (0)

/// @brief Synchronizes the default CUDA stream and checks for kernel errors.
#define CUDA_SYNC_CHECK()                                                             \
    do {                                                                              \
        CUDA_CHECK(cudaDeviceSynchronize());                                          \
    } while (0)

#else  // __CUDACC__ not defined

/// @cond Doxygen_Suppress
#define CUDA_CHECK(call)        (void)(call)
#define CUDA_CHECK_LAST()       ((void)0)
#define CUDA_SYNC_CHECK()       ((void)0)
/// @endcond

#endif  // __CUDACC__

// ---------------------------------------------------------------------------
// Generic Error-Checking Macro
// ---------------------------------------------------------------------------

/// @brief Checks a condition and aborts if false.
/// @param cond truth-valued expression
/// @param msg  human-readable message (const char*)
#define TENSORBIT_CHECK(cond, msg)                                                    \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr,                                                      \
                         "[TENSORBIT ERROR] %s:%d: %s (check `%s` failed)\n",        \
                         std::source_location::current().file_name(),                 \
                         std::source_location::current().line(),                      \
                         (msg), #cond);                                               \
            std::abort();                                                             \
        }                                                                             \
    } while (0)

// ---------------------------------------------------------------------------
// Thread-Safe Logger
// ---------------------------------------------------------------------------

namespace tensorbit::core {

/// @brief Log severity levels.
enum class LogLevel : uint8_t {
    kTrace = 0,
    kDebug = 1,
    kInfo  = 2,
    kWarn  = 3,
    kError = 4,
    kFatal = 5,
};

/// @brief Returns a human-readable label for each log level.
[[nodiscard]] constexpr std::string_view to_string(LogLevel level) noexcept {
    using namespace std::string_view_literals;
    switch (level) {
        case LogLevel::kTrace: return "TRACE"sv;
        case LogLevel::kDebug: return "DEBUG"sv;
        case LogLevel::kInfo:  return "INFO"sv;
        case LogLevel::kWarn:  return "WARN"sv;
        case LogLevel::kError: return "ERROR"sv;
        case LogLevel::kFatal: return "FATAL"sv;
    }
    return "UNKNOWN"sv;
}

/// @brief Thread-safe singleton logger.
///
/// Writes timestamped messages to stderr. Use the convenience macros
/// TENSORBIT_LOG_INFO, TENSORBIT_LOG_WARN, etc.
class Logger {
public:
    /// @brief Returns the global logger instance.
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;

    /// @brief Sets the minimum severity level. Messages below this are suppressed.
    void set_level(LogLevel level) { level_ = level; }

    /// @brief Logs a pre-formatted message at the given level.
    /// Formatting is done by the LOG macros via std::vformat.
    void log(LogLevel level, std::string_view msg,
             const std::source_location& loc = std::source_location::current()) {
        if (level < level_) return;
        std::lock_guard lock(mutex_);

        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) %
                    1000;

        std::tm tm_buf{};
        localtime_r(&time, &tm_buf);
        std::array<char, 32> time_buf{};
        std::strftime(time_buf.data(), time_buf.size(),
                      "%Y-%m-%d %H:%M:%S", &tm_buf);

        std::fprintf(stderr, "[%s.%03lld] [%s] %s  (%s:%u)\n",
                     time_buf.data(),
                     static_cast<long long>(ms.count()),
                     to_string(level).data(),
                     msg.data(),
                     loc.file_name(),
                     loc.line());

        if (level == LogLevel::kFatal) {
            std::abort();
        }
    }

private:
    Logger() = default;
    LogLevel   level_{LogLevel::kInfo};
    std::mutex mutex_;
};

}  // namespace tensorbit::core

// ---------------------------------------------------------------------------
// Unexpected<E> — error value wrapper (declared before Result)
// ---------------------------------------------------------------------------

namespace tensorbit::core {

/// @brief Wraps an error value for use with Result.
/// Created by the `unexpected()` factory function.
template<typename E>
class Unexpected {
public:
    template<typename E2 = E>
    explicit Unexpected(E2&& err) noexcept(std::is_nothrow_constructible_v<E, E2>)
        : err_(std::forward<E2>(err)) {}

    [[nodiscard]] E& error() & noexcept { return err_; }
    [[nodiscard]] const E& error() const& noexcept { return err_; }
    [[nodiscard]] E&& error() && noexcept { return std::move(err_); }
    [[nodiscard]] const E&& error() const&& noexcept { return std::move(err_); }

private:
    E err_;
};

template<typename E>
Unexpected(E) -> Unexpected<E>;

/// @brief Creates an Unexpected object (analogous to std::unexpected).
template<typename E>
[[nodiscard]] Unexpected<std::decay_t<E>> unexpected(E&& err) {
    return Unexpected<std::decay_t<E>>(std::forward<E>(err));
}

// ---------------------------------------------------------------------------
// Result<T, E> — C++20-compatible std::expected replacement
// ---------------------------------------------------------------------------

/// @brief A C++20-compatible result type modeling success-or-error semantics.
///
/// Holds either a value of type T (success) or an error of type E (failure).
/// Provides `std::expected`-like API for projects targeting C++20 where
/// `std::expected` is unavailable (it requires C++23).
///
/// @tparam T Success value type. May be `void`.
/// @tparam E Error type. Must be default-constructible.
template<typename T, typename E>
class Result {
public:
    using value_type = T;
    using error_type = E;

    // ---- Constructors ----

    /// @brief Constructs from a success value.
    template<typename U = T>
    /* implicit */ Result(U&& val) noexcept(std::is_nothrow_constructible_v<T, U>)
        requires (!std::is_same_v<std::decay_t<U>, Result>)
        : has_value_(true) {
        ::new (std::addressof(val_)) T(std::forward<U>(val));
    }

    /// @brief Constructs from an Unexpected error.
    template<typename E2>
    /* implicit */ Result(Unexpected<E2>&& unexp) noexcept(
        std::is_nothrow_constructible_v<E, E2>)
        : has_value_(false) {
        ::new (std::addressof(err_)) E(std::move(unexp).error());
    }

    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    Result(Result&& other) noexcept(std::is_nothrow_move_constructible_v<T> &&
                                    std::is_nothrow_move_constructible_v<E>)
        : has_value_(other.has_value_) {
        if (has_value_) {
            ::new (std::addressof(val_)) T(std::move(other.val_));
        } else {
            ::new (std::addressof(err_)) E(std::move(other.err_));
        }
    }

    Result& operator=(Result&& other) noexcept(std::is_nothrow_move_assignable_v<T> &&
                                               std::is_nothrow_move_assignable_v<E>) {
        if (this == &other) return *this;
        destroy();
        has_value_ = other.has_value_;
        if (has_value_) {
            ::new (std::addressof(val_)) T(std::move(other.val_));
        } else {
            ::new (std::addressof(err_)) E(std::move(other.err_));
        }
        return *this;
    }

    ~Result() { destroy(); }

    // ---- Observers ----

    /// @brief Returns true if the result holds a success value.
    [[nodiscard]] bool has_value() const noexcept { return has_value_; }

    /// @brief Returns true if it holds a success value.
    [[nodiscard]] explicit operator bool() const noexcept { return has_value_; }

    /// @brief Accesses the success value. Undefined behavior if !has_value().
    [[nodiscard]] T& value() noexcept { return val_; }
    [[nodiscard]] const T& value() const noexcept { return val_; }

    /// @brief Accesses the success value. Returns `default_value` on failure.
    template<typename U>
    [[nodiscard]] T value_or(U&& default_value) const& {
        return has_value_ ? val_ : static_cast<T>(std::forward<U>(default_value));
    }

    /// @brief Accesses the error value. Undefined behavior if has_value().
    [[nodiscard]] E& error() noexcept { return err_; }
    [[nodiscard]] const E& error() const noexcept { return err_; }

    /// @brief Dereference operators for monadic-like access.
    [[nodiscard]] T& operator*() noexcept { return val_; }
    [[nodiscard]] const T& operator*() const noexcept { return val_; }
    [[nodiscard]] T* operator->() noexcept { return std::addressof(val_); }
    [[nodiscard]] const T* operator->() const noexcept { return std::addressof(val_); }

private:
    void destroy() {
        if (has_value_) {
            val_.~T();
        } else {
            err_.~E();
        }
    }

    union {
        T val_;
        E err_;
    };
    bool has_value_;
};

// void specialization — holds no value, only error state.
template<typename E>
class Result<void, E> {
public:
    using error_type = E;

    Result() noexcept : has_value_(true) {}

    template<typename E2>
    /* implicit */ Result(Unexpected<E2>&& unexp) noexcept(
        std::is_nothrow_constructible_v<E, E2>)
        : has_value_(false) {
        ::new (std::addressof(err_)) E(std::move(unexp).error());
    }

    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    Result(Result&& other) noexcept(std::is_nothrow_move_constructible_v<E>)
        : has_value_(other.has_value_) {
        if (!has_value_) {
            ::new (std::addressof(err_)) E(std::move(other.err_));
        }
    }

    Result& operator=(Result&& other) noexcept(std::is_nothrow_move_assignable_v<E>) {
        if (this == &other) return *this;
        if (!has_value_) err_.~E();
        has_value_ = other.has_value_;
        if (!has_value_) {
            ::new (std::addressof(err_)) E(std::move(other.err_));
        }
        return *this;
    }

    ~Result() { if (!has_value_) err_.~E(); }

    [[nodiscard]] bool has_value() const noexcept { return has_value_; }
    [[nodiscard]] explicit operator bool() const noexcept { return has_value_; }
    [[nodiscard]] E& error() noexcept { return err_; }
    [[nodiscard]] const E& error() const noexcept { return err_; }

private:
    union { E err_; };
    bool has_value_;
};

}  // namespace tensorbit::core

// ---------------------------------------------------------------------------
// Logging Convenience Macros
// ---------------------------------------------------------------------------

#define TENSORBIT_LOG_TRACE(fmt, ...)                                                  \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kTrace,                                          \
        std::vformat((fmt), std::make_format_args(__VA_ARGS__)),                      \
        std::source_location::current())

#define TENSORBIT_LOG_DEBUG(fmt, ...)                                                  \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kDebug,                                          \
        std::vformat((fmt), std::make_format_args(__VA_ARGS__)),                      \
        std::source_location::current())

#define TENSORBIT_LOG_INFO(fmt, ...)                                                   \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kInfo,                                           \
        std::vformat((fmt), std::make_format_args(__VA_ARGS__)),                      \
        std::source_location::current())

#define TENSORBIT_LOG_WARN(fmt, ...)                                                   \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kWarn,                                           \
        std::vformat((fmt), std::make_format_args(__VA_ARGS__)),                      \
        std::source_location::current())

#define TENSORBIT_LOG_ERROR(fmt, ...)                                                  \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kError,                                          \
        std::vformat((fmt), std::make_format_args(__VA_ARGS__)),                      \
        std::source_location::current())

#define TENSORBIT_LOG_FATAL(fmt, ...)                                                  \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kFatal,                                          \
        std::vformat((fmt), std::make_format_args(__VA_ARGS__)),                      \
        std::source_location::current())
