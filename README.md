# 编译运行

在已启用四张 A6000 GPU 的 Linux StarPU Docker 容器中，从仓库根目录执行：

```bash
git pull
cd starpu
bash build.sh
```

构建使用 C++14：`g++` 编译宿主代码，`nvcc` 编译 CUDA kernel 并链接 `starpu-1.4`，生成 `build/rfi_subspace_extraction_starpu`。

默认适配 CUDA 10.1，生成 `compute_75` PTX，由 NVIDIA 驱动在 A6000 上 JIT 编译。使用 CUDA 11.1 及以上时，可执行 `CUDA_ARCH=86 bash build.sh` 生成 A6000 的原生 `sm_86` 代码。

四卡运行：

```bash
export STARPU_NCUDA=4
export STARPU_NCPU=0
export STARPU_NOPENCL=0
export STARPU_NHIP=0
export STARPU_NWORKER_PER_CUDA=1
export STARPU_SCHED=lws

mkdir -p logs
set -o pipefail
./build/rfi_subspace_extraction_starpu \
  --antennas 768 --snapshots 2048 \
  --bands 8 --time-blocks 1 --correlation-lags 32 \
  --rank 512 --panel 1 --mgs-passes 2 \
  --mode full --gs-kernel blocked --gs-layout row \
  --correlation-layout pair --covariance-order forward \
  --compact-gs 0 --verify 1 --verify-full 0 \
  --print-manifest 0 \
  2>&1 | tee logs/rfi_4gpu.txt
```

`TIMING run_sec` 是 batch 耗时，`VERIFY passed=1` 表示校验通过，`WORKER` 行记录各张 GPU 的任务数。

单卡运行时将 `STARPU_NCUDA` 改成 `1`。全部应用参数可通过以下命令查看：

```bash
./build/rfi_subspace_extraction_starpu --help
```
