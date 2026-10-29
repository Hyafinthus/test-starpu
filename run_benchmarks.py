#!/usr/bin/env python3
"""Run the RFI granularity/scheduler matrix inside the Linux GPU container."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def write_rows(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./build/rfi_subspace_extraction_starpu")
    parser.add_argument("--output", default="logs/bench-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--versions", nargs="+", choices=["tile", "stage", "partitioned"], default=["tile", "stage", "partitioned"])
    parser.add_argument("--policies", nargs="+", choices=["lws", "dmda", "dmdas"], default=["lws", "dmda", "dmdas"])
    parser.add_argument("--calibration-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("app_args", nargs=argparse.REMAINDER, help="RFI arguments after --")
    args = parser.parse_args()
    if min(args.calibration_runs, args.repeats, args.gpus, args.partitions) < 1:
        parser.error("run counts, GPU count and partition count must be positive")
    extra = args.app_args[1:] if args.app_args[:1] == ["--"] else args.app_args
    if any(token.split("=", 1)[0] in ("--granularity", "--partitions", "--metrics-prefix") for token in extra):
        parser.error("granularity, partitions and output prefix are controlled by the runner")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = str(Path(args.binary).resolve())
    env = os.environ.copy()
    env.update(STARPU_NCUDA=str(args.gpus), STARPU_NCPU="0", STARPU_NOPENCL="0",
               STARPU_NHIP="0", STARPU_NWORKER_PER_CUDA="1", STARPU_FXT_TRACE="0")
    metadata = {"runner": vars(args), "binary": binary,
                "environment": {k: v for k, v in env.items() if k.startswith(("STARPU_", "CUDA_"))}}
    for name, command in [("gpus", ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv"]),
                          ("cuda", ["nvcc", "--version"]),
                          ("starpu", ["pkg-config", "--modversion", env.get("STARPU_PKG", "starpu-1.4")])]:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        metadata[name] = proc.stdout
    (root / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    runs, calibrations = [], []

    def execute(version, policy, phase, index, directory, model_dir):
        prefix = directory / (phase + "-" + str(index))
        run_env = env.copy()
        run_env.update(STARPU_SCHED=policy, STARPU_PERF_MODEL_DIR=str(model_dir),
                       STARPU_CALIBRATE="1" if phase == "calibrate" else "0")
        command = [binary, "--granularity", version, "--partitions", str(args.partitions),
                   "--metrics-prefix", str(prefix)] + extra
        print("{} {} {} {}".format(version, policy, phase, index), flush=True)
        started = time.perf_counter()
        with prefix.with_suffix(".log").open("w") as log:
            log.write("command: " + json.dumps(command) + "\n")
            log.write("model_dir: " + str(model_dir) + "\n")
            log.flush()
            proc = subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT)
        wall = time.perf_counter() - started
        row = {"version": version, "policy": policy, "phase": phase, "index": index,
               "process_wall_sec": wall, "returncode": proc.returncode, "prefix": str(prefix)}
        run_file = Path(str(prefix) + "_run.csv")
        if run_file.exists():
            with run_file.open(newline="") as stream:
                row.update({item["metric"]: item["value"] for item in csv.DictReader(stream)})
        text = prefix.with_suffix(".log").read_text(errors="replace")
        for line in text.splitlines():
            if line.startswith("RESULT ") or line.startswith("VERIFY "):
                for field in line.split()[1:]:
                    key, value = field.split("=", 1)
                    row[("verify_" if line.startswith("VERIFY ") else "") + key] = value
        row["model_warning_lines"] = sum("not calibrated" in line.lower() for line in text.splitlines())
        runs.append(row)
        write_rows(root / "runs.csv", runs)
        if proc.returncode:
            raise SystemExit("run failed: " + str(prefix.with_suffix(".log")))
        return row

    for version in args.versions:
        for policy in args.policies:
            directory = root / (version + "-" + policy)
            directory.mkdir()
            trained = directory / "trained-models"
            trained.mkdir()
            training = [execute(version, policy, "calibrate", i, directory, trained)
                        for i in range(1, args.calibration_runs + 1)]
            calibration = {"version": version, "policy": policy,
                           "calibration_runs": args.calibration_runs,
                           "calibration_process_wall_sec": sum(float(row["process_wall_sec"]) for row in training),
                           "calibration_e2e_sec": sum(float(row["e2e_sec"]) for row in training),
                           "calibration_runtime_init_sec": sum(float(row["runtime_init_sec"]) for row in training),
                           "last_calibration_warning_lines": training[-1]["model_warning_lines"]}
            calibrations.append(calibration)
            write_rows(root / "calibration.csv", calibrations)
            for i in range(1, args.repeats + 1):
                # Each measured repetition starts from the same trained history.
                model_dir = directory / ("models-measure-" + str(i))
                shutil.copytree(str(trained), str(model_dir))
                execute(version, policy, "measure", i, directory, model_dir)
    print("Results: " + str(root / "runs.csv"), flush=True)


if __name__ == "__main__":
    main()
