#pragma once

/// @file common.hpp
/// @brief Core utilities: CUDA error-checking macros, thread-safe logging, and shared types.
/// @ingroup tensorbit-core

#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <mutex>
#include <source_location>
#include <string_view>

// ---------------------------------------------------------------------------
// CUDA Error-Checking Macros
// ---------------------------------------------------------------------------

#ifdef __CUDACC__

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

    /// @brief Logs a formatted message at the given level.
    template<typename... Args>
    void log(LogLevel level,
             std::format_string<Args...> fmt,
             Args&&... args,
             const std::source_location& loc = std::source_location::current()) {
        if (level < level_) return;
        std::lock_guard lock(mutex_);

        auto now   = std::chrono::system_clock::now();
        auto time  = std::chrono::system_clock::to_time_t(now);
        auto ms    = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now.time_since_epoch()) %
                     1000;

        std::array<char, 32> time_buf{};
        std::strftime(time_buf.data(), time_buf.size(),
                      "%Y-%m-%d %H:%M:%S", std::localtime(&time));

        std::fprintf(stderr, "[%s.%03lld] [%s] ",
                     time_buf.data(),
                     static_cast<long long>(ms.count()),
                     to_string(level).data());

        std::string msg = std::format(fmt, std::forward<Args>(args)...);
        std::fprintf(stderr, "%s  (%s:%u)\n",
                     msg.c_str(),
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
// Logging Convenience Macros
// ---------------------------------------------------------------------------

#define TENSORBIT_LOG_TRACE(...)                                                      \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kTrace, __VA_ARGS__)

#define TENSORBIT_LOG_DEBUG(...)                                                      \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kDebug, __VA_ARGS__)

#define TENSORBIT_LOG_INFO(...)                                                       \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kInfo, __VA_ARGS__)

#define TENSORBIT_LOG_WARN(...)                                                       \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kWarn, __VA_ARGS__)

#define TENSORBIT_LOG_ERROR(...)                                                      \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kError, __VA_ARGS__)

#define TENSORBIT_LOG_FATAL(...)                                                      \
    ::tensorbit::core::Logger::instance().log(                                        \
        ::tensorbit::core::LogLevel::kFatal, __VA_ARGS__)
