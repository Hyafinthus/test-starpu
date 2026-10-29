# 编译与运行

在已启用四张 A6000 的 Linux StarPU Docker 容器中，进入本仓库目录：

```bash
git pull
bash build.sh

export STARPU_NCUDA=4
export STARPU_NCPU=0
export STARPU_NOPENCL=0
export STARPU_NHIP=0
export STARPU_NWORKER_PER_CUDA=1
mkdir -p logs
```

构建使用 C++14、`g++`、`nvcc`、`pkg-config starpu-1.4` 和 Python 3。CUDA 10.1 默认生成 `compute_75` PTX，由驱动在 A6000 上 JIT。CUDA 11.1 及以上可用 `CUDA_ARCH=86 bash build.sh`。

默认参数：64 tiles（64 bands × 1 time block），每 tile 4096 antennas、2048 snapshots、24 lags，rank 1024，一次 reverse-MGS；`--panel 1 --gs-kernel blocked --gs-layout row --correlation-layout pair --covariance-order forward`。

运行三个版本：

```bash
STARPU_SCHED=lws ./build/rfi_subspace_extraction_starpu \
  --granularity tile --metrics-prefix logs/tile-lws

STARPU_SCHED=lws ./build/rfi_subspace_extraction_starpu \
  --granularity stage --metrics-prefix logs/stage-lws

STARPU_SCHED=lws ./build/rfi_subspace_extraction_starpu \
  --granularity partitioned --partitions 4 --metrics-prefix logs/partitioned-lws
```

`--granularity` 默认 `stage`。将 `STARPU_SCHED` 改成 `dmda` 或 `dmdas` 即可切换策略。`VERIFY passed=1` 表示摘要校验通过。`--verify-full 1` 会取回整个 basis 做完整校验，其传输和耗时计入 result/E2E。

完整参数：

```bash
./build/rfi_subspace_extraction_starpu --help
```

# 批量实验与模型校准

```bash
python3 run_benchmarks.py --output logs/a6000-4gpu \
  --calibration-runs 1 --repeats 3
```

依次执行 `tile/stage/partitioned × lws/dmda/dmdas`。每组在独立模型目录中做一次完整校准运行，再正式运行三次。校准用 `STARPU_CALIBRATE=1`；正式运行用 `0`，每次复制同一份校准后的模型。结果目录每次使用新名称。

只跑 Stage，或改变运行规模：

```bash
python3 run_benchmarks.py --output logs/stage-policies \
  --versions stage --policies lws dmda dmdas --calibration-runs 2 --repeats 3

python3 run_benchmarks.py --output logs/custom \
  --versions tile stage --policies lws dmda dmdas -- \
  --bands 64 --time-blocks 1 --antennas 4096 --snapshots 2048 \
  --correlation-lags 24 --rank 1024 --mgs-passes 1 --panel 1
```

输出：

- `runs.csv`：全部运行的指标、校验值、阶段（calibrate/measure）、进程总耗时。
- `calibration.csv`：各组校准运行次数、校准 E2E 累计时间、初始化时间、进程累计时间。这里的校准成本是获得模型所执行的完整训练运行成本，包含 RFI 计算。
- `environment.json`：GPU、驱动、CUDA、StarPU 版本和启动参数。
- 各组目录：原始日志、逐任务 CSV、设备/传输统计、校准模型及各次运行模型。

`model_warning_lines` 记录日志中的未充分校准提示数量，`tasks_with_scheduler_prediction` 记录任务获得正值时间预测的数量；后者用于检查 `dmda/dmdas` 的预测覆盖。增加 `--calibration-runs` 可增加模型样本。[StarPU 官方校准说明](https://starpu.gitlabpages.inria.fr/quickGuide.html)；模型目录通过 [`STARPU_PERF_MODEL_DIR`](https://starpu.gitlabpages.inria.fr/tutorials/docker/) 设置。

# 三种任务粒度

下表是上述默认规模、`panel=1` 时的应用 task 数。task 数与实际 CUDA kernel 启动数分别记录。

| 版本 | 调度边界 | 每 tile tasks | 64 tiles tasks | CUDA kernels |
| --- | --- | ---: | ---: | ---: |
| StarPU-Tile | 一个 codelet 内完成整个 tile | 1 | 64 | 131392 |
| StarPU-Stage | 每个数值 kernel 一个 codelet task | 2053 | 131392 | 131392 |
| StarPU-Partitioned | Stage + correlation 输出行四分块 | 2056 | 131584 | 131584 |

Tile 的 codelet 在同一个 StarPU CUDA stream 上依次启动原有 kernel，适合比较 tile placement。Stage 保留 preprocessing、correlation、MGS 和 summary 内部的 kernel 边界，允许在这些边界上重新选择 GPU。

Stage 每个 tile 的任务组成：preprocessing 3 个，correlation 1 个，correlation summary 1 个，MGS panel 1024 个、update 1023 个，final summary 1 个。MGS 面板按 rank 从后向前处理，每个面板完成后更新其前缀。

Partitioned 使用 `starpu_matrix_filter_vertical_block` 对 basis 和 correlation snapshot 按工作行分块。各块读取完整 normalized samples 和 lag offsets，分别写入两个矩阵的子句柄；计算仍执行相同的逐对、逐 lag、逐 sample 累加。分块后显式在 `STARPU_MAIN_RAM` 合并，再进入 MGS/summary；这部分往返传输计入指标。当前分块范围是 correlation，MGS 沿用 Stage 的任务粒度。

GPU 分配由所选调度策略决定，应用没有设置 task-to-worker 绑定。`lws` 使用工作窃取；`dmda/dmdas` 使用任务性能模型和数据传输信息。应用为所有 codelets 配置 `STARPU_HISTORY_BASED` 模型，模型 key 包含矩阵尺寸、lags、rank、面板区间、分块行区间和布局。

# 数据依赖与实现结构

每个 tile 注册九个父句柄：samples、mean、stddev、basis、correlation snapshot、lag offsets、correlation row summaries、tile summary、MGS norm。

| Codelet | 句柄访问模式 |
| --- | --- |
| Mean | samples R；mean W |
| Stddev | samples R；mean R；stddev W |
| Normalize | samples RW；mean R；stddev R |
| Correlation | samples R；lags R；basis W；snapshot W |
| Correlation summary | snapshot R；row summaries W |
| MGS panel / update | basis RW |
| Classic MGS norm / normalize | basis R + norm W / basis RW + norm R |
| Final summary | basis R；row summaries R；tile summary W |

StarPU 根据同一句柄的 R/W/RW 访问顺序建立 RAW、WAR、WAW 依赖。不同 tile 使用独立句柄。Correlation summary 与 MGS 在 correlation 之后分叉，final summary 等待两者；最终各 tile 摘要由宿主读取并汇总校验。Partitioned 用 `partition_plan/partition_submit/unpartition_submit` 维护父子句柄依赖。

| 文件 | 对应原 SYCL 单文件中的职责 |
| --- | --- |
| `rfi_subspace_extraction_common.hpp` | 参数、输入生成、计算量统计、结果校验 |
| `rfi_subspace_extraction_types.hpp` | 摘要数据结构 |
| `rfi_kernels.hpp` | 原数值计算体；保留 float 计算及 double 摘要累加 |
| `rfi_codelets.hpp` | KernelArgs 与 codelet 标识 |
| `rfi_cuda.cu` | CUDA kernel launch、StarPU stream、Tile 内部流水线 |
| `rfi_tasks.cpp` | 数据注册、R/W/RW 声明、task 提交、分块、性能模型 |
| `rfi_subspace_extraction_starpu.cpp` | 初始化、计时边界、等待、取回结果 |
| `rfi_metrics.cpp` | 任务时间线、调度 CPU 计时、bus/device 统计、CSV |
| `run_benchmarks.py` | 三种粒度 × 三种策略、校准与重复运行 |

执行后端是 CUDA：StarPU 调用 CUDA codelet，codelet 在 `starpu_cuda_get_local_stream()` 上启动复用的数值 kernel，并通过 `STARPU_CUDA_ASYNC` 通知运行时跟踪异步完成。

# 指标口径

每次运行按 `--metrics-prefix` 生成七个 CSV。所有粒度均启用同一套 profiling。

| 文件/字段 | 含义 |
| --- | --- |
| `_run.csv` / `e2e_sec` | 从开始提交 batch 到宿主读取摘要并完成校验；包含初次 H2D、调度、计算、D2H 和结果处理 |
| `run_sec` | 开始提交到 `starpu_task_wait_for_all` 返回 |
| `runtime_init_sec`、`data_init_sec`、`total_sec` | StarPU 初始化、输入生成/句柄注册、初始化起至结果完成的总时间 |
| `_tasks.csv` / `scheduling_latency_sec` | task ready（push_start）至 codelet start；包含排队和数据等待 |
| `dependency_wait_sec` | submit 至 ready，反映前驱依赖等待 |
| `queue_residence_sec` | push_end 至 pop_end |
| `data_ready_wait_sec` | acquire_data_start 至 acquire_data_end，反映 worker 获取可用输入的等待 |
| `codelet_sec` | StarPU profiling 的 end_time − start_time，记录异步 CUDA codelet 执行区间 |
| `_devices.csv` / `codelet_idle_fraction` | `1 − 该 GPU 上 codelet 区间并集 / batch 时间`；表示没有应用 codelet 执行的时间比例 |
| `_workers.csv` | StarPU worker 的 GPU/memory-node 对应关系、执行、等待、休眠、调度墙钟时间与运行时执行任务数 |
| `_buses.csv` | StarPU 各 memory-node bus 的实际传输 bytes/次数，区分 execution 和 result 阶段 |
| `_scheduler.csv` / `thread_cpu_sec` | 原调度策略 push/pop/submit/pre/post 等回调内的线程 CPU 时间；包含返回空任务的 pop，嵌套回调只计一次 |
| `_stages.csv` | 每类 codelet 的 task 数、执行时间总和、平均值、P95 |

`_tasks.csv` 同时保留原始时间戳、tile、pass、panel、分块行区间、worker/GPU、预测时间和访问声明数。时间单位为秒，`predicted_us` 为微秒。缺失的时间区间写 `nan`；data-ready 的有效样本数另行记录。调度等待、数据等待和执行时间按其区间定义使用，跨 GPU 的累计时间可大于 E2E。

GPU 的 `codelet_idle_fraction` 描述任务时间线；硬件 SM utilization 属于另一测量口径。Bus bytes 按物理传输段累计，经主存中转的 GPU-to-GPU 数据会分别计入两段。Scheduler CPU 指标覆盖策略回调；StarPU 的依赖维护、数据管理和模型更新在各自运行时路径执行。

`_run.csv` 还输出应用 task 数、CUDA kernel 数、scheduler 观察到的提交数、父/子句柄数、partition 定义数、R/W/RW 声明数和显式 worker 绑定数。默认 Tile/Stage 均有 576 个父句柄；四分块版增加 128 个 partition 定义、512 个子句柄。`application_tasks` 由应用完成回调计数；StarPU 内部分区/数据管理任务通过运行时统计另列。

手动决策包括：选择 Tile/Stage/Partitioned、MGS panel 大小、correlation row/pair 布局；Partitioned 额外指定分块数，使用连续工作行划分和主存合并。代码没有逐 task 指定 GPU。
