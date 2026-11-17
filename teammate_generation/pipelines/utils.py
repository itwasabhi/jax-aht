"""Shared helpers for staged pipelines.

A pipeline is a sequence of ordinary training scripts (marl/run.py,
teammate_generation/run.py, ego_agent_training/run.py), each run as its own
subprocess so that every stage gets its own Hydra output dir, wandb run and
saved checkpoint.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_config(script, overrides):
    """Return the composed Hydra config of `script` without running it."""
    cmd = [sys.executable, script, *overrides, "--cfg", "job", "--resolve"]
    out = subprocess.run(
        cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    return yaml.safe_load(out)


def hydra_value(v):
    """Format a python value as a Hydra override literal."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{hydra_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(hydra_value(x) for x in v) + "]"
    if isinstance(v, str):
        return "'" + v.replace("'", "\\'") + "'"
    return repr(v)


def _override_key(o):
    return o.split("=", 1)[0].lstrip("+~")


def dedupe_overrides(overrides):
    """Keep the last override for each key, so user extras replace generated ones."""
    last = {_override_key(o): i for i, o in enumerate(overrides)}
    forced = {_override_key(o) for o in overrides if o.startswith("++")}
    out = []
    for i, o in enumerate(overrides):
        k = _override_key(o)
        if last[k] != i:
            continue
        if k in forced and not o.startswith(("+", "~")):
            o = "++" + o  # keep upsert semantics for keys absent from the base config
        out.append(o)
    return out


class Pipeline:
    def __init__(self, args, alg):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = os.path.join(
            REPO_ROOT, "results", args.task, f"{alg}_pipeline", args.label, self.timestamp
        )
        os.makedirs(self.dir, exist_ok=True)
        self.tag = f"{alg}_pipeline_{args.label}_{self.timestamp}"
        self.records = []

    def stage_dir(self, *parts):
        return os.path.join(self.dir, *parts)

    def run_stage(self, name, script, overrides, run_dir=None):
        """Run one stage; returns its output dir. Stops the pipeline on failure."""
        run_dir = run_dir or self.stage_dir(name)
        all_overrides = dedupe_overrides([
            f"task={self.args.task}",
            f"label={self.args.label}",
            f"logger.mode={self.args.logger_mode}",
            f"hydra.run.dir={run_dir}",
            f"++logger.group={self.tag}",
            "local_logger.save_train_out=true",
            *overrides,
            *getattr(self.args, f"extra_{name.split('/')[0]}", "").split(),
        ])
        cmd = [sys.executable, script, *all_overrides]
        print(f"\n=== [{name}] {' '.join(cmd)}\n", flush=True)
        record = {"stage": name, "cmd": cmd, "output_dir": run_dir, "status": "running"}
        self.records.append(record)
        self._write()
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError:
            record["status"] = "failed"
            self._write()
            raise
        record["status"] = "success"
        self._write()
        return run_dir

    def _write(self):
        with open(os.path.join(self.dir, "pipeline.json"), "w") as f:
            json.dump(self.records, f, indent=2)


def common_argparser(stages):
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="e.g. lbf/lbf_7x7_nolevels")
    p.add_argument("--label", default="default_label")
    p.add_argument("--logger_mode", default="offline")
    p.add_argument("--num_seeds", type=int, default=None, help="override NUM_SEEDS")
    p.add_argument(
        "--stages", default=",".join(stages), help=f"comma separated subset of {stages}"
    )
    for s in stages:
        p.add_argument(f"--extra_{s}", default="", help=f"hydra overrides for {s}")
    return p


def ego_stage(pipe, name, task, partner_name, partner_spec, ego_train_algorithm, run_dir):
    """Run ppo_ego against a saved partner population."""
    ego_hparams = {
        k: v
        for k, v in ego_train_algorithm.items()
        if k not in ("ENV_NAME", "ENV_KWARGS", "ROLLOUT_LENGTH", "NUM_EGO_TRAIN_SEEDS")
    }
    overrides = [
        f"algorithm=ppo_ego/{task}",
        "~algorithm.partner_agent",
        f"+algorithm.partner_agent.{partner_name}={hydra_value(partner_spec)}",
        *[f"++algorithm.{k}={hydra_value(v)}" for k, v in ego_hparams.items()],
    ]
    return pipe.run_stage(name, "ego_agent_training/run.py", overrides, run_dir)
