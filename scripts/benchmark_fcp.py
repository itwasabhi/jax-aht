#!/usr/bin/env python3
"""Side-by-side benchmark comparing Direct FCP (teammate_generation/fcp.py)
vs. Staged Pipeline FCP (teammate_generation/pipelines/fcp_pipeline.py).

Both methods are configured identically, inheriting all task-specific
hyperparameters (timesteps, envs, learning rates, epochs, etc.) directly
from `algorithm/fcp/{task}`.

Example usage with uv:
    # Benchmark using the task's default FCP config:
    uv run python scripts/benchmark_fcp.py --task lbf/lbf_7x7_nolevels

    # Benchmark with custom pop size, seeds, and shortened timesteps:
    uv run python scripts/benchmark_fcp.py --task lbf/lbf_7x7_nolevels --partner_pop_size 5 --num_seeds 2 --timesteps 50000

    # Benchmark only one method:
    uv run python scripts/benchmark_fcp.py --task lbf/lbf_7x7_nolevels --method pipeline
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from teammate_generation.pipelines.utils import resolve_config


def get_dir_size_mb(path):
    """Calculate total size of directory in megabytes."""
    if not os.path.exists(path):
        return 0.0
    total_bytes = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_bytes += os.path.getsize(fp)
            except OSError:
                pass
    return round(total_bytes / (1024 * 1024), 2)


def get_proc_peak_rss_mb(pid):
    """Read peak resident memory (VmHWM) in MB from Linux /proc status if available."""
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    return round(int(parts[1]) / 1024, 1)
    except Exception:
        pass
    return None


def run_command_monitored(cmd, desc):
    """Run a command while measuring elapsed time and tracking peak memory."""
    print(f"\n{'='*80}\n[START] {desc}\nCMD: {' '.join(cmd)}\n{'='*80}", flush=True)
    start_time = time.time()
    peak_rss = None

    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    try:
        while proc.poll() is None:
            time.sleep(0.5)
            rss = get_proc_peak_rss_mb(proc.pid)
            if rss is not None:
                peak_rss = max(peak_rss or 0.0, rss)
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise

    exit_code = proc.wait()
    duration = time.time() - start_time
    status = "SUCCESS" if exit_code == 0 else f"FAILED (code {exit_code})"
    print(
        f"\n[DONE] {desc} -> {status} in {duration:.2f}s "
        f"(Peak RSS: {f'{peak_rss:.1f} MB' if peak_rss else 'N/A'})\n",
        flush=True,
    )
    return {
        "status": status,
        "duration_sec": round(duration, 2),
        "peak_rss_mb": peak_rss,
        "exit_code": exit_code,
    }


def benchmark_direct_fcp(cfg_params, run_dir):
    """Run teammate_generation/run.py with algorithm=fcp (in-memory nested vmap)."""
    os.makedirs(run_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "teammate_generation/run.py",
        f"task={cfg_params['task']}",
        f"algorithm=fcp/{cfg_params['task']}",
        "train_ego=false",
        "run_heldout_eval=false",
        f"algorithm.PARTNER_POP_SIZE={cfg_params['pop_size']}",
        f"algorithm.NUM_SEEDS={cfg_params['num_seeds']}",
        f"algorithm.TOTAL_TIMESTEPS={cfg_params['timesteps']}",
        f"algorithm.NUM_ENVS={cfg_params['num_envs']}",
        f"algorithm.NUM_CHECKPOINTS={cfg_params['num_checkpoints']}",
        "logger.mode=disabled",
        "logger.log_train_out=false",
        "local_logger.save_train_out=true",
        f"hydra.run.dir={run_dir}",
    ]
    res = run_command_monitored(
        cmd,
        desc=(
            f"Method 1: Direct FCP (vmapped - {cfg_params['num_seeds']} seeds x "
            f"{cfg_params['pop_size']} agents = {cfg_params['num_seeds'] * cfg_params['pop_size']} total)"
        ),
    )
    checkpoint_dir = os.path.join(run_dir, "saved_train_run")
    res["checkpoint_dir"] = checkpoint_dir
    res["checkpoint_size_mb"] = get_dir_size_mb(checkpoint_dir)
    return res


def benchmark_pipeline_fcp(cfg_params, run_label):
    """Run teammate_generation/pipelines/fcp_pipeline.py with --stages pool."""
    extra_pool = (
        f"algorithm.TOTAL_TIMESTEPS={cfg_params['timesteps']} "
        f"algorithm.NUM_ENVS={cfg_params['num_envs']} "
        f"algorithm.NUM_CHECKPOINTS={cfg_params['num_checkpoints']}"
    )
    cmd = [
        sys.executable,
        "teammate_generation/pipelines/fcp_pipeline.py",
        "--task",
        cfg_params["task"],
        "--label",
        run_label,
        "--stages",
        "pool",
        "--partner_pop_size",
        str(cfg_params["pop_size"]),
        "--num_seeds",
        str(cfg_params["num_seeds"]),
        "--num_checkpoints",
        str(cfg_params["num_checkpoints"]),
        "--logger_mode",
        "disabled",
        "--extra_pool",
        extra_pool,
    ]
    res = run_command_monitored(
        cmd,
        desc=(
            f"Method 2: Staged Pipeline FCP (pipeline - {cfg_params['num_seeds']} seeds x "
            f"{cfg_params['pop_size']} agents = {cfg_params['num_seeds'] * cfg_params['pop_size']} total)"
        ),
    )

    base_results = os.path.join(
        REPO_ROOT, "results", cfg_params["task"], "fcp_pipeline", run_label
    )
    latest_run_dir = None
    if os.path.exists(base_results):
        subdirs = sorted(
            [
                os.path.join(base_results, d)
                for d in os.listdir(base_results)
                if os.path.isdir(os.path.join(base_results, d))
            ]
        )
        if subdirs:
            latest_run_dir = subdirs[-1]

    res["output_dir"] = latest_run_dir or base_results
    res["checkpoint_size_mb"] = (
        get_dir_size_mb(latest_run_dir) if latest_run_dir else 0.0
    )
    return res


def format_table(results_dict, cfg_params):
    """Format a clean side-by-side comparison table."""
    total_steps = (
        cfg_params["num_seeds"] * cfg_params["pop_size"] * cfg_params["timesteps"]
    )
    header = (
        f"{'Metric':<32} | {'Direct FCP (vmapped)':<24} | {'Pipeline FCP (staged)':<26}\n"
        f"{'-'*32}-+-{'-'*24}-+-{'-'*26}"
    )
    rows = []

    m1 = results_dict.get("direct", {})
    m2 = results_dict.get("pipeline", {})

    def get_val(m, k, fmt=lambda x: str(x), default="N/A"):
        v = m.get(k)
        return fmt(v) if v is not None else default

    t1 = m1.get("duration_sec")
    t2 = m2.get("duration_sec")

    rows.append(
        f"{'Status':<32} | {get_val(m1, 'status'):<24} | {get_val(m2, 'status'):<26}"
    )
    rows.append(
        f"{'Total Time':<32} | {f'{t1:.2f} s' if t1 else 'N/A':<24} | {f'{t2:.2f} s' if t2 else 'N/A':<26}"
    )

    fps1 = (total_steps / t1) if (t1 and t1 > 0) else None
    fps2 = (total_steps / t2) if (t2 and t2 > 0) else None
    rows.append(
        f"{'Throughput (env steps/sec)':<32} | {f'{fps1:,.1f}' if fps1 else 'N/A':<24} | {f'{fps2:,.1f}' if fps2 else 'N/A':<26}"
    )

    rows.append(
        f"{'Peak RSS Memory':<32} | {get_val(m1, 'peak_rss_mb', lambda x: f'{x:.1f} MB'):<24} | {get_val(m2, 'peak_rss_mb', lambda x: f'{x:.1f} MB'):<26}"
    )
    rows.append(
        f"{'Checkpoint Size on Disk':<32} | {get_val(m1, 'checkpoint_size_mb', lambda x: f'{x:.1f} MB'):<24} | {get_val(m2, 'checkpoint_size_mb', lambda x: f'{x:.1f} MB'):<26}"
    )
    rows.append(
        f"{'Execution Architecture':<32} | {'Nested jax.vmap':<24} | {'Sequential Subprocesses':<26}"
    )
    rows.append(
        f"{'Ego Retrain Reusability':<32} | {'Coupled (re-trains pool)':<24} | {'Decoupled (--pool_path)':<26}"
    )

    if t1 and t2 and t1 > 0 and t2 > 0:
        faster_str = (
            f"Pipeline is {t1/t2:.2f}x faster"
            if t2 < t1
            else f"Direct is {t2/t1:.2f}x faster"
        )
        rows.append(
            f"{'Relative Speed':<32} | {faster_str:<53}"
        )

    return "\n".join([header, *rows])


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side benchmark of Direct FCP (vmapped) vs. Pipeline FCP (staged)."
    )
    parser.add_argument(
        "--task",
        default="lbf/lbf_7x7_nolevels",
        help="Task name (default: lbf/lbf_7x7_nolevels)",
    )
    parser.add_argument(
        "--partner_pop_size",
        type=int,
        default=None,
        help="Number of partner agents per pool (default: task fcp config value)",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Number of pools / seeds to train (default: task fcp config value)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Timesteps per partner agent (default: task fcp config value)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of vectorized environments (default: task fcp config value)",
    )
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=None,
        help="Checkpoints to save per agent (default: task fcp config value)",
    )
    parser.add_argument(
        "--method",
        choices=["both", "direct", "pipeline"],
        default="both",
        help="Which method to benchmark (default: both)",
    )
    args = parser.parse_args()

    # Resolve base FCP config for the specified task to get like-for-like defaults
    try:
        base_cfg = resolve_config(
            "teammate_generation/run.py",
            [
                f"task={args.task}",
                f"algorithm=fcp/{args.task}",
                "train_ego=false",
                "run_heldout_eval=false",
            ],
        )
        algo = base_cfg.get("algorithm", {})
    except Exception as e:
        print(f"Warning: Could not resolve base FCP config for task '{args.task}': {e}")
        algo = {}

    pop_size = args.partner_pop_size or algo.get("PARTNER_POP_SIZE", 5)
    num_seeds = args.num_seeds or algo.get("NUM_SEEDS", 2)
    timesteps = (
        args.timesteps
        if args.timesteps is not None
        else int(algo.get("TOTAL_TIMESTEPS", 50000))
    )
    num_envs = args.num_envs or algo.get("NUM_ENVS", 8)
    num_checkpoints = args.num_checkpoints or algo.get("NUM_CHECKPOINTS", 2)

    cfg_params = {
        "task": args.task,
        "pop_size": pop_size,
        "num_seeds": num_seeds,
        "timesteps": timesteps,
        "num_envs": num_envs,
        "num_checkpoints": num_checkpoints,
        "learning_rate": algo.get("LR", "default"),
        "update_epochs": algo.get("UPDATE_EPOCHS", "default"),
        "minibatches": algo.get("NUM_MINIBATCHES", "default"),
        "clip_eps": algo.get("CLIP_EPS", "default"),
        "ent_coef": algo.get("ENT_COEF", "default"),
        "activation": algo.get("ACTIVATION", "default"),
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bench_dir = os.path.join(REPO_ROOT, "results", "benchmark_fcp", timestamp)
    os.makedirs(bench_dir, exist_ok=True)

    print(
        f"\n{'#'*80}\n"
        f"# FCP BENCHMARK: Direct (vmapped) vs. Pipeline (staged)\n"
        f"# Task: {args.task}\n"
        f"# Setup: {num_seeds} pool(s) x {pop_size} agents = {num_seeds * pop_size} total partner agents\n"
        f"# Timesteps per agent: {timesteps:,} (envs: {num_envs}, checkpoints: {num_checkpoints})\n"
        f"# Shared Hyperparams (from algorithm/fcp/{args.task}):\n"
        f"#   LR: {cfg_params['learning_rate']} | Epochs: {cfg_params['update_epochs']} | "
        f"Minibatches: {cfg_params['minibatches']} | Clip: {cfg_params['clip_eps']} | "
        f"Ent: {cfg_params['ent_coef']}\n"
        f"# Total environment steps: {num_seeds * pop_size * timesteps:,}\n"
        f"{'#'*80}\n"
    )

    results = {}

    if args.method in ("both", "direct"):
        direct_run_dir = os.path.join(bench_dir, "direct_fcp")
        results["direct"] = benchmark_direct_fcp(cfg_params, direct_run_dir)

    if args.method in ("both", "pipeline"):
        pipeline_label = f"bench_{timestamp}"
        results["pipeline"] = benchmark_pipeline_fcp(cfg_params, pipeline_label)

    print("\n" + "#" * 80)
    print("# BENCHMARK RESULTS")
    print("#" * 80)
    print(format_table(results, cfg_params))
    print("#" * 80 + "\n")

    summary_path = os.path.join(bench_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"config": cfg_params, "results": results}, f, indent=2)
    print(f"Benchmark summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
