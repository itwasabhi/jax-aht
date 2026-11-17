"""Staged CoMeDi: PPO seed (marl/run.py) -> CoMeDi population -> PPO ego.

Usage:
    uv run teammate_generation/pipelines/comedi_pipeline.py --task lbf/lbf_7x7_nolevels
"""

import os

from teammate_generation.pipelines.utils import (
    Pipeline,
    common_argparser,
    ego_stage,
    resolve_config,
)

STAGES = ["seed", "comedi", "ego"]


def main():
    p = common_argparser(STAGES)
    p.add_argument("--seed_path", help="existing seed checkpoint; skips the seed stage")
    p.add_argument("--comedi_path", help="existing comedi checkpoint; skips comedi stage")
    args = p.parse_args()
    stages = args.stages.split(",")
    task = args.task

    comedi_overrides = [
        f"algorithm=comedi/{task}",
        "train_ego=false",
        "run_heldout_eval=false",
    ]
    if args.num_seeds is not None:
        comedi_overrides.append(f"algorithm.NUM_SEEDS={args.num_seeds}")
    cfg = resolve_config("teammate_generation/run.py", [f"task={task}", *comedi_overrides, *args.extra_comedi.split()])
    algo = cfg["algorithm"]
    pop_size, num_seeds = algo["PARTNER_POP_SIZE"], algo["NUM_SEEDS"]

    pipe = Pipeline(args, "comedi")
    seed_path, comedi_path = args.seed_path, args.comedi_path

    if "seed" in stages and not seed_path and not comedi_path:
        d = pipe.run_stage(
            "seed",
            "marl/run.py",
            [
                f"algorithm=ippo/{task}",
                "algorithm.ACTOR_TYPE=pseudo_actor_with_conditional_critic",
                f"++algorithm.POP_SIZE={pop_size}",
                f"algorithm.NUM_SEEDS={num_seeds}",
            ],
        )
        seed_path = os.path.join(d, "saved_train_run")

    if "comedi" in stages and not comedi_path:
        assert seed_path, "comedi stage needs a seed checkpoint (--seed_path)"
        d = pipe.run_stage(
            "comedi",
            "teammate_generation/run.py",
            [*comedi_overrides, f"algorithm.seed_agent.path={seed_path}"],
        )
        comedi_path = os.path.join(d, "saved_train_run")

    if "ego" in stages:
        assert comedi_path, "ego stage needs a comedi checkpoint (--comedi_path)"
        for s in range(num_seeds):
            spec = {
                "path": comedi_path,
                "actor_type": "actor_with_conditional_critic",
                "ckpt_key": "final_params_conf",
                "POP_SIZE": pop_size,
                "idx_list": [[s, i] for i in range(pop_size)],
            }
            name = "ego" if num_seeds == 1 else f"ego/seed_{s}"
            ego_stage(pipe, name, task, "comedi", spec, algo["ego_train_algorithm"], pipe.stage_dir(name))


if __name__ == "__main__":
    main()
