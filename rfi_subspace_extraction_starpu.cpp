#include "rfi_starpu.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

namespace rfi = rfi_subspace_extraction;
namespace sp = rfi_starpu;

namespace {
using clock_type = std::chrono::steady_clock;
double seconds_between(clock_type::time_point begin, clock_type::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

class StarpuRuntime {
public:
  StarpuRuntime() {
    starpu_conf conf;
    sp::check_starpu(starpu_conf_init(&conf), "initialize configuration");
    conf.ncpus = 0;
    conf.nopencl = 0;
    conf.nhip = 0;
    if (!std::getenv("STARPU_SCHED")) conf.sched_policy_name = "lws";
    sp::check_starpu(starpu_init(&conf), "initialize StarPU");
    const auto workers = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
    if (workers == 0) {
      starpu_shutdown();
      throw std::runtime_error("no CUDA workers");
    }
    std::cout << "RUNTIME backend=cuda cuda_workers=" << workers
              << " task_granularity=whole_tile_stage"
              << " cuda_async=1 cpu_codelets=0 performance_models=0\n";
  }
  ~StarpuRuntime() {
    starpu_task_wait_for_all();
    starpu_shutdown();
  }
  StarpuRuntime(const StarpuRuntime&) = delete;
  StarpuRuntime& operator=(const StarpuRuntime&) = delete;
};
} // namespace

int main(int argc, char **argv) {
  try {
    const rfi::Config config =
        rfi::parse_config(argc, argv, /*celerity_options=*/false);
    sp::validate_starpu_config(config);
    rfi::print_configuration(config, "starpu", "whole_tile");

    const auto total_begin = clock_type::now();
    StarpuRuntime runtime;
    sp::initialize_codelets();
    const auto runtime_init_end = clock_type::now();

    std::vector<std::unique_ptr<sp::TileBuffers>> tiles;
    const std::size_t tile_count = rfi::tile_count(config);
    tiles.reserve(tile_count);
    for (std::size_t time_block = 0; time_block < config.time_blocks;
         ++time_block) {
      for (std::size_t band = 0; band < config.bands; ++band) {
        const std::size_t tile = time_block * config.bands + band;
        tiles.push_back(std::make_unique<sp::TileBuffers>(tile, config));
      }
    }
    const auto data_init_end = clock_type::now();

    rfi::Manifest manifest(config);
    const auto run_begin = clock_type::now();
    sp::submit_batch(tiles, config, manifest);
    sp::check_starpu(starpu_task_wait_for_all(), "wait for batch");
    const auto run_end = clock_type::now();

    if (manifest.submitted() !=
        rfi::kernels_per_tile(config) * tile_count) {
      throw std::runtime_error("submitted kernel count does not match metadata");
    }

    const auto result_begin = clock_type::now();
    const rfi::VerificationResult result = sp::read_results(tiles, config);
    const auto result_end = clock_type::now();

    std::cout << std::fixed << std::setprecision(6)
              << "TIMING runtime_init_sec="
              << seconds_between(total_begin, runtime_init_end) << '\n'
              << "TIMING data_init_sec="
              << seconds_between(runtime_init_end, data_init_end) << '\n'
              << "TIMING run_sec=" << seconds_between(run_begin, run_end)
              << '\n'
              << "TIMING result_sec="
              << seconds_between(result_begin, result_end) << '\n'
              << "TIMING total_sec="
              << seconds_between(total_begin, result_end) << '\n'
              << "RESULT correlation_checksum="
              << result.correlation_checksum
              << " basis_checksum=" << result.basis_checksum
              << " checked_rows=" << result.checked_rows
              << " max_norm_error=" << result.max_norm_error
              << " max_dot=" << result.max_dot
              << " max_mask_error=" << result.max_mask_error
              << " min_regularized_diagonal="
              << result.min_regularized_diagonal
              << " min_active_correlation="
              << result.min_active_correlation
              << " max_active_offdiagonal="
              << result.max_active_offdiagonal
              << " correlation_nonfinite="
              << result.correlation_nonfinite
              << " basis_nonfinite=" << result.basis_nonfinite << '\n'
              << "VERIFY passed=" << (result.passed ? 1 : 0)
              << " full=" << (config.verify_full ? 1 : 0) << '\n';
    sp::print_worker_stats();
    return (config.verify || config.verify_full) && !result.passed ? 2 : 0;
  } catch (const std::exception &exception) {
    std::cerr << "ERROR " << exception.what() << '\n';
    return 2;
  }
}
