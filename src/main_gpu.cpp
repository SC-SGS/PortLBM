/**
 * @brief       Entry point of the portlbm driver executable.
 *
 * @copyright   Copyright (c) 2025 Marcel Graf
 *              Copyright (c) 2026 Alexander Strack
 */

// INCLUDES ///////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <filesystem>
#include <iostream>
#include <stdexcept>

#ifdef __linux__
#include <unistd.h>  // readlink
#endif

// Library public API
#include <lbm/core/simulation.hpp>
#include <lbm/execution/sycl_algorithm_handler.hpp>
#include <lbm/file_interaction/file_interaction.hpp>

#ifdef WITH_VISUALIZATION
#include <lbm/gui/lbm_gui.hpp>
#endif

#ifdef BENCHMARK_MODE
#include <lbm/benchmark/benchmark.hpp>
#endif

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

/**
 * @brief   Returns the directory that contains the running executable.
 *          Uses /proc/self/exe on Linux for reliability; falls back to
 *          std::filesystem::canonical(argv[0]) on other platforms.
 */
static std::filesystem::path exe_dir(const char* argv0)
{
    namespace fs = std::filesystem;
#ifdef __linux__
    char buf[4096];
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0)
    {
        buf[len] = '\0';
        return fs::path(buf).parent_path();
    }
#endif
    std::error_code ec;
    fs::path p = fs::canonical(argv0, ec);
    if (!ec)
        return p.parent_path();
    return fs::current_path();
}

/**
 * @brief   Returns the project root directory — the parent of the directory
 *          that contains the executable (development layout: exe is in build/).
 *          Falls back to the exe's own directory if settings are found there
 *          instead (in-source or installed layout).
 */
static std::filesystem::path project_root(const char* argv0)
{
    namespace fs = std::filesystem;
    const fs::path dir = exe_dir(argv0);

    // Out-of-source build: <root>/build/portlbm  =>  root = dir/..
    if (fs::exists(dir.parent_path() / "settings" / "settings.json"))
        return dir.parent_path();
    // In-source or installed: <root>/portlbm  =>  root = dir
    if (fs::exists(dir / "settings" / "settings.json"))
        return dir;

    return dir.parent_path();  // best guess; caller will report a clear error
}

/**
 * @brief   Resolves the path to settings.json.
 *          If argv[1] is provided it is used directly; otherwise the file is
 *          located relative to the executable so the binary can be invoked
 *          from any working directory.
 *
 * @throws  std::runtime_error when no settings file can be found.
 */
static std::string resolve_settings(int argc, const char* const* argv)
{
    if (argc > 1)
        return argv[1];

    const std::filesystem::path candidate = project_root(argv[0]) / "settings" / "settings.json";
    if (!std::filesystem::exists(candidate))
        throw std::runtime_error(
            "Cannot locate settings/settings.json relative to the executable.\n"
            "Usage: portlbm [/path/to/settings.json]");
    return candidate.string();
}

// ---------------------------------------------------------------------------
// main() — four variants controlled by preprocessor flags
// ---------------------------------------------------------------------------

#ifdef WITH_VISUALIZATION

#ifdef BENCHMARK_MODE  // WITH_VISUALIZATION && BENCHMARK_MODE

int main(int argc, char* argv[])
{
    try
    {
        const std::string settings_path = resolve_settings(argc, argv);
        const std::filesystem::path root = project_root(argv[0]);

        std::filesystem::create_directories(root / "benchmarks" / "gui");

        lbm::gui::Gui<lbm::execution::SYCLAlgorithmHandler> gui("PortLBM", settings_path);
        gui.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

#else  // WITH_VISUALIZATION && !BENCHMARK_MODE

int main(int argc, char* argv[])
{
    try
    {
        const std::string settings_path = resolve_settings(argc, argv);
        lbm::gui::Gui<lbm::execution::SYCLAlgorithmHandler> gui("PortLBM", settings_path);
        gui.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

#endif

#else  // !WITH_VISUALIZATION

#ifdef BENCHMARK_MODE  // !WITH_VISUALIZATION && BENCHMARK_MODE

int main(int argc, char* argv[])
{
    try
    {
        const std::string settings_path = resolve_settings(argc, argv);
        const std::filesystem::path root = project_root(argv[0]);

        std::filesystem::create_directories(root / "benchmarks" / "phase0");
        std::filesystem::create_directories(root / "benchmarks" / "phase1");
        std::filesystem::create_directories(root / "benchmarks" / "phase2");

        std::cout << "Starting benchmark.\n\n";

        std::shared_ptr<lbm::execution::SYCLAlgorithmHandler> algorithm_handler =
            std::make_shared<lbm::execution::SYCLAlgorithmHandler>(settings_path);

        lbm::benchmark::Benchmark benchmark(algorithm_handler);
        benchmark.phase_0();
        benchmark.phase_1();
        benchmark.phase_2();

        std::cout << "\nDone with all benchmarks. Exiting.\n\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

#else  // !WITH_VISUALIZATION && !BENCHMARK_MODE

int main(int argc, char* argv[])
{
    try
    {
        const std::string settings_path = resolve_settings(argc, argv);
        std::unique_ptr<lbm::execution::SYCLAlgorithmHandler> algorithm_handler =
            std::make_unique<lbm::execution::SYCLAlgorithmHandler>(settings_path);
        algorithm_handler->start();
        algorithm_handler->block_until_finished();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

#endif  // !BENCHMARK_MODE

#endif  // !WITH_VISUALIZATION

// ! main_gpu.cpp
