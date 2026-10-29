#include "rfi_starpu.hpp"
#include <array>
#include <atomic>
#include <fstream>
#include <map>
#include <time.h>

namespace rfi_starpu {
namespace {
constexpr std::size_t nstages = static_cast<std::size_t>(Stage::Count);
const char* names[] = {"mean", "stddev", "normalize", "correlation_row",
    "correlation_pair", "correlation_summary", "reverse_norm",
    "reverse_normalize", "reverse_panel", "reverse_update", "final_summary", "tile"};
const char* categories[] = {"preprocessing", "preprocessing", "preprocessing",
    "correlation", "correlation", "summary", "mgs", "mgs", "mgs", "mgs", "summary", "tile"};
struct TaskRecord {
  std::size_t id, tile;
  Stage stage;
  KernelArgs args;
  starpu_task* task;
  starpu_profiling_task_info info{};
  unsigned reads = 0, writes = 0, annotations = 0;
  double predicted_us = 0;
  bool complete = false;
};
std::vector<std::unique_ptr<TaskRecord>> records;
std::array<std::atomic<std::size_t>, STARPU_NMAXWORKERS> kernel_counts{};
struct BusRecord {
  int id, src, dst;
  long long execution_bytes = 0, result_bytes = 0;
  int execution_transfers = 0, result_transfers = 0;
};
std::vector<BusRecord> buses;
std::vector<starpu_profiling_worker_info> workers;
double batch_begin = 0, batch_end = 0;

// Time the selected policy's callbacks with per-thread CPU clocks. Nested
// callbacks are counted once; lock sleep time is excluded by the clock.
enum Hook { Push, Pop, Submit, Pre, Post, Notify, Schedule, Simulate, HookCount };
const char* hook_names[] = {"push", "pop", "submit", "pre_exec", "post_exec", "notify", "do_schedule", "simulate_push"};
std::array<std::atomic<unsigned long long>, HookCount> hook_ns{}, hook_calls{};
std::atomic<std::size_t> scheduler_seen{0};
std::array<unsigned long long, HookCount> batch_hook_ns{}, batch_hook_calls{};
std::size_t batch_scheduler_seen = 0;
starpu_sched_policy original{}, measured{};
thread_local unsigned hook_depth = 0;
unsigned long long cpu_ns() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<unsigned long long>(t.tv_sec) * 1000000000ull + t.tv_nsec;
}
class HookTimer {
  Hook hook_;
  bool outer_;
  unsigned long long begin_;
public:
  explicit HookTimer(Hook h) : hook_(h), outer_(hook_depth++ == 0),
      begin_(outer_ ? cpu_ns() : 0) {}
  ~HookTimer() {
    if (outer_) {
      hook_ns[hook_].fetch_add(cpu_ns() - begin_, std::memory_order_relaxed);
      hook_calls[hook_].fetch_add(1, std::memory_order_relaxed);
    }
    --hook_depth;
  }
};
int timed_push(starpu_task* t) { HookTimer timer(Push); return original.push_task(t); }
starpu_task* timed_pop(unsigned ctx) { HookTimer timer(Pop); return original.pop_task(ctx); }
void timed_submit(starpu_task* t) {
  HookTimer timer(Submit);
  scheduler_seen.fetch_add(1, std::memory_order_relaxed);
  if (original.submit_hook) original.submit_hook(t);
}
void timed_pre(starpu_task* t, unsigned ctx) { HookTimer timer(Pre); original.pre_exec_hook(t, ctx); }
void timed_post(starpu_task* t, unsigned ctx) { HookTimer timer(Post); original.post_exec_hook(t, ctx); }
void timed_notify(starpu_task* t, int w, int p, unsigned ctx) {
  HookTimer timer(Notify); original.push_task_notify(t, w, p, ctx);
}
void timed_schedule(unsigned ctx) { HookTimer timer(Schedule); original.do_schedule(ctx); }
double timed_simulate(starpu_task* t) { HookTimer timer(Simulate); return original.simulate_push_task(t); }

double seconds(const timespec& t) { return double(t.tv_sec) + double(t.tv_nsec) * 1e-9; }
double interval(const timespec& begin, const timespec& end) {
  const double a = seconds(begin), b = seconds(end);
  return a > 0 && b >= a ? b - a : std::numeric_limits<double>::quiet_NaN();
}
void completed(void* arg) {
  auto& record = *static_cast<TaskRecord*>(arg);
  if (record.task->profiling_info) record.info = *record.task->profiling_info;
  record.predicted_us = record.task->predicted;
  record.complete = true;
}
std::ofstream output(const std::string& prefix, const char* suffix) {
  std::ofstream file(prefix + suffix);
  if (!file) throw std::runtime_error("open metrics file " + prefix + suffix);
  file << std::setprecision(12);
  return file;
}
struct Samples {
  std::vector<double> values;
  double sum = 0;
  void add(double value) { if (std::isfinite(value)) { values.push_back(value); sum += value; } }
  double mean() const { return values.empty() ? NAN : sum / values.size(); }
  double p95() {
    if (values.empty()) return NAN;
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(values.size() * 0.95)) - 1];
  }
};
struct Device {
  std::size_t tasks = 0, kernels = 0;
  std::vector<std::pair<double, double>> spans;
};
double busy_time(std::vector<std::pair<double, double>> spans) {
  std::sort(spans.begin(), spans.end());
  double total = 0, end = batch_begin;
  for (const auto& span : spans) {
    const double a = std::max({span.first, end, batch_begin});
    const double b = std::min(span.second, batch_end);
    if (b > a) total += b - a;
    end = std::max(end, b);
  }
  return total;
}
void collect_buses(bool result_phase) {
  for (auto& bus : buses) {
    starpu_profiling_bus_info info{};
    check_starpu(starpu_bus_get_profiling_info(bus.id, &info), "read bus counters");
    if (result_phase) {
      bus.result_bytes = info.transferred_bytes;
      bus.result_transfers = info.transfer_count;
    } else {
      bus.execution_bytes = info.transferred_bytes;
      bus.execution_transfers = info.transfer_count;
    }
  }
}
} // namespace

const char* stage_name(Stage stage) { return names[static_cast<std::size_t>(stage)]; }

void configure_scheduler(starpu_conf& conf) {
  const char* env = std::getenv("STARPU_SCHED");
  const std::string name = env ? env : "lws";
  bool found = false;
  for (auto** p = starpu_sched_get_predefined_policies(); *p; ++p) {
    if ((*p)->policy_name && name == (*p)->policy_name) { original = **p; found = true; break; }
  }
  if (!found) throw std::invalid_argument("scheduler unavailable: " + name);
  measured = original;
  if (original.push_task) measured.push_task = timed_push;
  if (original.pop_task) measured.pop_task = timed_pop;
  measured.submit_hook = timed_submit;
  if (original.pre_exec_hook) measured.pre_exec_hook = timed_pre;
  if (original.post_exec_hook) measured.post_exec_hook = timed_post;
  if (original.push_task_notify) measured.push_task_notify = timed_notify;
  if (original.do_schedule) measured.do_schedule = timed_schedule;
  if (original.simulate_push_task) measured.simulate_push_task = timed_simulate;
  // Pass the selected policy through the pointer API so its callbacks can be timed.
  conf.sched_policy_name = nullptr;
  conf.sched_policy = &measured;
}

void record_cuda_kernel() {
  kernel_counts[starpu_worker_get_id()].fetch_add(1, std::memory_order_relaxed);
}
void track_task(starpu_task* task, std::size_t tile, Stage stage, const KernelArgs& args) {
  auto record = std::make_unique<TaskRecord>();
  record->id = records.size();
  record->tile = tile;
  record->stage = stage;
  record->args = args;
  record->task = task;
  record->annotations = STARPU_TASK_GET_NBUFFERS(task);
  for (unsigned i = 0; i < record->annotations; ++i) {
    const auto mode = STARPU_TASK_GET_MODE(task, i);
    record->reads += (mode & STARPU_R) != 0;
    record->writes += (mode & STARPU_W) != 0;
  }
  task->callback_func = completed;
  task->callback_arg = record.get();
  records.push_back(std::move(record));
}
void begin_measurement() {
  const int status = starpu_profiling_status_set(STARPU_PROFILING_ENABLE);
  if (status < 0) check_starpu(status, "enable task profiling");
  for (auto& c : hook_ns) c.store(0);
  for (auto& c : hook_calls) c.store(0);
  scheduler_seen.store(0);
  records.clear();
  for (auto& c : kernel_counts) c.store(0);
  buses.clear();
  for (int id = 0; id < starpu_bus_get_count(); ++id) {
    BusRecord bus{};
    bus.id = id;
    bus.src = starpu_bus_get_src(id);
    bus.dst = starpu_bus_get_dst(id);
    buses.push_back(bus);
  }
  workers.resize(starpu_worker_get_count());
  batch_begin = starpu_timing_now() * 1e-6;
}
void end_execution() {
  batch_end = starpu_timing_now() * 1e-6;
  for (unsigned i = 0; i < HookCount; ++i) {
    batch_hook_ns[i] = hook_ns[i].load();
    batch_hook_calls[i] = hook_calls[i].load();
  }
  batch_scheduler_seen = scheduler_seen.load();
  for (unsigned id = 0; id < workers.size(); ++id)
    check_starpu(starpu_profiling_worker_get_info(id, &workers[id]), "read worker profiling");
  collect_buses(false);
}
void end_measurement() { collect_buses(true); }

void write_metrics(const rfi::Config& config, double runtime_init, double data_init,
                   double run, double result, double e2e, double total) {
  auto tasks = output(config.metrics_prefix, "_tasks.csv");
  tasks << "task_id,tile,category,stage,pass,panel_begin,panel_end,row_begin,row_count,worker,cuda_device,submit_sec,ready_sec,push_end_sec,pop_end_sec,acquire_start_sec,acquire_end_sec,start_sec,end_sec,dependency_wait_sec,scheduling_latency_sec,queue_residence_sec,data_ready_wait_sec,codelet_sec,predicted_us,handle_annotations,read_annotations,write_annotations\n";
  Samples scheduling, dependency, queue, data_wait, execution;
  std::array<Samples, nstages> stage_exec;
  std::array<std::size_t, nstages> stage_tasks{};
  std::map<int, Device> devices;
  for (unsigned worker = 0; worker < workers.size(); ++worker) {
    if (starpu_worker_get_type(worker) == STARPU_CUDA_WORKER)
      devices[starpu_worker_get_devid(worker)].kernels += kernel_counts[worker].load();
  }
  std::size_t annotations = 0, reads = 0, writes = 0, kernels = 0, predicted = 0;
  for (const auto& ptr : records) {
    const auto& r = *ptr;
    if (!r.complete || seconds(r.info.end_time) == 0)
      throw std::runtime_error("missing task profiling record");
    const auto& p = r.info;
    const auto s = static_cast<std::size_t>(r.stage);
    const int device = starpu_worker_get_devid(p.workerid);
    const double dep = interval(p.submit_time, p.push_start_time);
    const double sched = interval(p.push_start_time, p.start_time);
    const double queued = interval(p.push_end_time, p.pop_end_time);
    const double wait = interval(p.acquire_data_start_time, p.acquire_data_end_time);
    const double exec = interval(p.start_time, p.end_time);
    dependency.add(dep); scheduling.add(sched); queue.add(queued); data_wait.add(wait); execution.add(exec);
    stage_exec[s].add(exec); ++stage_tasks[s];
    ++devices[device].tasks;
    devices[device].spans.emplace_back(seconds(p.start_time), seconds(p.end_time));
    annotations += r.annotations; reads += r.reads; writes += r.writes;
    predicted += std::isfinite(r.predicted_us) && r.predicted_us > 0;
    tasks << r.id << ',' << r.tile << ',' << categories[s] << ',' << names[s] << ','
          << r.args.mgs_pass << ',' << r.args.panel_begin << ',' << r.args.panel_end << ','
          << r.args.row_begin << ',' << r.args.row_count << ',' << p.workerid << ',' << device << ','
          << seconds(p.submit_time) << ',' << seconds(p.push_start_time) << ',' << seconds(p.push_end_time) << ','
          << seconds(p.pop_end_time) << ',' << seconds(p.acquire_data_start_time) << ','
          << seconds(p.acquire_data_end_time) << ',' << seconds(p.start_time) << ',' << seconds(p.end_time) << ','
          << dep << ',' << sched << ',' << queued << ',' << wait << ',' << exec << ',' << r.predicted_us << ','
          << r.annotations << ',' << r.reads << ',' << r.writes << '\n';
  }
  auto device_file = output(config.metrics_prefix, "_devices.csv");
  device_file << "cuda_device,application_tasks,cuda_kernels,codelet_busy_sec,codelet_idle_fraction\n";
  double total_idle = 0;
  for (auto& item : devices) {
    const double busy = busy_time(item.second.spans);
    const double idle = 1.0 - busy / (batch_end - batch_begin);
    total_idle += idle;
    kernels += item.second.kernels;
    device_file << item.first << ',' << item.second.tasks << ',' << item.second.kernels << ',' << busy << ',' << idle << '\n';
    std::cout << "DEVICE cuda_device=" << item.first << " tasks=" << item.second.tasks
              << " kernels=" << item.second.kernels << " codelet_idle_fraction=" << idle << '\n';
  }
  auto worker_file = output(config.metrics_prefix, "_workers.csv");
  worker_file << "worker,cuda_device,memory_node,runtime_executed_tasks,total_sec,executing_sec,waiting_sec,sleeping_sec,scheduling_wall_sec\n";
  for (unsigned w = 0; w < workers.size(); ++w) {
    if (starpu_worker_get_type(w) != STARPU_CUDA_WORKER) continue;
    const auto& p = workers[w];
    worker_file << w << ',' << starpu_worker_get_devid(w) << ',' << starpu_worker_get_memory_node(w) << ',' << p.executed_tasks << ','
                << seconds(p.total_time) << ',' << seconds(p.executing_time) << ',' << seconds(p.waiting_time) << ','
                << seconds(p.sleeping_time) << ',' << seconds(p.scheduling_time) << '\n';
  }
  auto bus_file = output(config.metrics_prefix, "_buses.csv");
  bus_file << "bus,src_memory_node,dst_memory_node,execution_bytes,result_bytes,execution_transfers,result_transfers\n";
  long long execution_bytes = 0, result_bytes = 0;
  for (const auto& bus : buses) {
    execution_bytes += bus.execution_bytes; result_bytes += bus.result_bytes;
    bus_file << bus.id << ',' << bus.src << ',' << bus.dst << ',' << bus.execution_bytes << ',' << bus.result_bytes
             << ',' << bus.execution_transfers << ',' << bus.result_transfers << '\n';
  }
  auto scheduler_file = output(config.metrics_prefix, "_scheduler.csv");
  scheduler_file << "callback,outer_calls,thread_cpu_sec\n";
  double scheduler_cpu = 0;
  for (unsigned i = 0; i < HookCount; ++i) {
    const double cpu = batch_hook_ns[i] * 1e-9;
    scheduler_cpu += cpu;
    scheduler_file << hook_names[i] << ',' << batch_hook_calls[i] << ',' << cpu << '\n';
  }
  auto stages = output(config.metrics_prefix, "_stages.csv");
  stages << "category,stage,application_tasks,execution_sum_sec,execution_mean_sec,execution_p95_sec\n";
  for (std::size_t i = 0; i < nstages; ++i) if (stage_tasks[i])
    stages << categories[i] << ',' << names[i] << ',' << stage_tasks[i] << ',' << stage_exec[i].sum << ','
           << stage_exec[i].mean() << ',' << stage_exec[i].p95() << '\n';
  const std::size_t tiles = rfi::tile_count(config);
  const bool partitioned = config.granularity == "partitioned" && config.mode != rfi::Mode::OrthogonalizeOnly;
  const auto expected_kernels = tiles * rfi::kernels_per_tile(config) + (partitioned ? tiles * (config.partitions - 1) : 0);
  if (records.size() != expected_tasks(config) || kernels != expected_kernels)
    throw std::runtime_error("task/kernel accounting mismatch");
  auto summary = output(config.metrics_prefix, "_run.csv");
  summary << "metric,value\n";
  summary << "granularity," << config.granularity << "\nscheduler," << original.policy_name
          << "\ntiles," << tiles << "\nantennas," << config.antennas << "\nsnapshots," << config.snapshots
          << "\nlags," << config.correlation_lags << "\nrank," << config.subspace_rank
          << "\nmode," << rfi::mode_name(config.mode)
          << "\ncorrelation_layout," << rfi::correlation_layout_name(config.correlation_layout)
          << "\ncovariance_order," << rfi::correlation_order_name(config.correlation_order)
          << "\ngs_layout," << rfi::gs_layout_name(config.gs_layout)
          << "\ngs_kernel," << rfi::gs_kernel_name(config.gs_kernel)
          << "\ncompact_gs," << config.compact_gs << "\nverify_full," << config.verify_full
          << "\nmanual_partition_axis," << (partitioned ? "work_rows" : "none")
          << "\npanel_size," << config.panel_size << "\nmgs_passes," << config.mgs_passes
          << "\npartitions," << (partitioned ? config.partitions : 0)
          << "\napplication_tasks," << records.size() << "\ncuda_kernels," << kernels
          << "\nscheduler_seen_tasks," << batch_scheduler_seen
          << "\ntasks_with_scheduler_prediction," << predicted
          << "\nregistered_parent_handles," << 9 * tiles
          << "\npartition_definitions," << (partitioned ? 2 * tiles : 0)
          << "\npartition_child_handles," << (partitioned ? 2 * tiles * config.partitions : 0)
          << "\nhandle_annotations," << annotations << "\nread_annotations," << reads << "\nwrite_annotations," << writes
          << "\nexplicit_worker_placements,0\npartition_gather_node," << (partitioned ? "STARPU_MAIN_RAM" : "none")
          << "\nruntime_init_sec," << runtime_init << "\ndata_init_sec," << data_init
          << "\nrun_sec," << run << "\nresult_sec," << result << "\ne2e_sec," << e2e
          << "\ntotal_sec," << total << "\ncodelet_execution_sum_sec," << execution.sum
          << "\ncodelet_execution_mean_sec," << execution.mean() << "\ncodelet_execution_p95_sec," << execution.p95()
          << "\nscheduling_latency_mean_sec," << scheduling.mean() << "\nscheduling_latency_p95_sec," << scheduling.p95()
          << "\ndependency_wait_mean_sec," << dependency.mean() << "\nqueue_residence_mean_sec," << queue.mean()
          << "\ndata_ready_wait_sum_sec," << data_wait.sum << "\ndata_ready_wait_mean_sec," << data_wait.mean()
          << "\ndata_ready_wait_samples," << data_wait.values.size()
          << "\nmean_gpu_codelet_idle_fraction," << total_idle / devices.size()
          << "\nexecution_transfer_bytes," << execution_bytes << "\nresult_transfer_bytes," << result_bytes
          << "\ne2e_transfer_bytes," << execution_bytes + result_bytes
          << "\nscheduler_callback_cpu_sec," << scheduler_cpu << '\n';
  std::cout << "METRICS granularity=" << config.granularity << " scheduler=" << original.policy_name
            << " application_tasks=" << records.size() << " cuda_kernels=" << kernels
            << " e2e_sec=" << e2e << " scheduler_callback_cpu_sec=" << scheduler_cpu
            << " transfer_bytes=" << execution_bytes + result_bytes
            << " prefix=" << config.metrics_prefix << '\n';
}
} // namespace rfi_starpu
