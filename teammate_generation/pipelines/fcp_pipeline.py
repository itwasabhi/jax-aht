"""Staged FCP: IPPO partner pool (marl/run.py) -> PPO ego.

Usage:
    uv run teammate_generation/pipelines/fcp_pipeline.py --task lbf/lbf_7x7_nolevels
"""

import os

from teammate_generation.pipelines.utils import (
    Pipeline,
    common_argparser,
    ego_stage,
    resolve_config,
)

STAGES = ["pool", "ego"]


def main():
    p = common_argparser(STAGES)
    p.add_argument("--pool_path", help="existing pool checkpoint; skips the pool stage")
    p.add_argument("--partner_pop_size", type=int, help="IPPO agents per pool (default: fcp config)")
    p.add_argument("--num_checkpoints", type=int, help="checkpoints per agent (default: fcp config)")
    args = p.parse_args()
    stages = args.stages.split(",")
    task = args.task

    fcp_overrides = [f"algorithm=fcp/{task}", "train_ego=false", "run_heldout_eval=false"]
    if args.num_seeds is not None:
        fcp_overrides.append(f"algorithm.NUM_SEEDS={args.num_seeds}")
    cfg = resolve_config("teammate_generation/run.py", [f"task={task}", *fcp_overrides])
    algo = cfg["algorithm"]
    # --extra_pool targets marl/run.py (ippo config), so pool shape comes from flags/fcp config
    pop_size = args.partner_pop_size or algo["PARTNER_POP_SIZE"]
    num_ckpts = args.num_checkpoints or algo["NUM_CHECKPOINTS"]
    pool_seeds = algo["NUM_SEEDS"]

    pipe = Pipeline(args, "fcp")
    pool_paths = [args.pool_path] if args.pool_path else []

    # One pool per FCP seed; a pool is PARTNER_POP_SIZE IPPO seeds.
    if "pool" in stages and not pool_paths:
        for s in range(pool_seeds):
            name = "pool" if pool_seeds == 1 else f"pool/seed_{s}"
            d = pipe.run_stage(
                name,
                "marl/run.py",
                [
                    f"algorithm=ippo/{task}",
                    f"algorithm.NUM_SEEDS={pop_size}",
                    f"algorithm.NUM_CHECKPOINTS={num_ckpts}",
                    f"algorithm.TRAIN_SEED={algo['TRAIN_SEED'] + s}",
                ],
                pipe.stage_dir(name),
            )
            pool_paths.append(os.path.join(d, "saved_train_run"))

    if "ego" in stages:
        assert pool_paths, "ego stage needs a pool checkpoint (--pool_path)"
        for s, path in enumerate(pool_paths):
            spec = {
                "path": path,
                "actor_type": algo.get("ACTOR_TYPE", "mlp"),
                "ckpt_key": "checkpoints",
                "idx_list": None,
            }
            name = "ego" if len(pool_paths) == 1 else f"ego/seed_{s}"
            ego_stage(pipe, name, task, "fcp", spec, algo["ego_train_algorithm"], pipe.stage_dir(name))


if __name__ == "__main__":
    main()
