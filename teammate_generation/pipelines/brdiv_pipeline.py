"""Staged BRDiv / LBRDiv: partner population -> PPO ego.

Usage:
    uv run teammate_generation/pipelines/brdiv_pipeline.py --alg brdiv --task lbf/lbf_7x7_nolevels
"""

import os

from teammate_generation.pipelines.utils import (
    Pipeline,
    common_argparser,
    ego_stage,
    resolve_config,
)

STAGES = ["population", "ego"]


def main():
    p = common_argparser(STAGES)
    p.add_argument("--alg", choices=["brdiv", "lbrdiv"], default="brdiv")
    p.add_argument("--population_path", help="existing population checkpoint; skips training")
    args = p.parse_args()
    stages = args.stages.split(",")
    task = args.task

    overrides = [f"algorithm={args.alg}/{task}", "train_ego=false", "run_heldout_eval=false"]
    if args.num_seeds is not None:
        overrides.append(f"algorithm.NUM_SEEDS={args.num_seeds}")
    cfg = resolve_config("teammate_generation/run.py", [f"task={task}", *overrides, *args.extra_population.split()])
    algo = cfg["algorithm"]
    pop_size, num_seeds = algo["PARTNER_POP_SIZE"], algo["NUM_SEEDS"]

    pipe = Pipeline(args, args.alg)
    pop_path = args.population_path
    if "population" in stages and not pop_path:
        d = pipe.run_stage("population", "teammate_generation/run.py", overrides)
        pop_path = os.path.join(d, "saved_train_run")

    if "ego" in stages:
        assert pop_path, "ego stage needs a population checkpoint (--population_path)"
        for s in range(num_seeds):
            spec = {
                "path": pop_path,
                "actor_type": "actor_with_conditional_critic",
                "ckpt_key": "final_params_conf",
                "POP_SIZE": pop_size,
                "idx_list": [[s, i] for i in range(pop_size)],
            }
            name = "ego" if num_seeds == 1 else f"ego/seed_{s}"
            ego_stage(pipe, name, task, args.alg, spec, algo["ego_train_algorithm"], pipe.stage_dir(name))


if __name__ == "__main__":
    main()
