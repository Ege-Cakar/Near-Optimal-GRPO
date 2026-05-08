from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grpo_mario_theory.bc_pretrain import run_bc_suite
from grpo_mario_theory.plotting import plot_bc
from grpo_mario_theory.utils import ensure_results, load_config, platformer_kwargs, resolve_device, resolve_profile, save_run_config, section


def run(config: str, seed: int, profile: bool | str, outdir: str, device: str = "cpu", env_backend: str | None = None):
    cfg = load_config(config)
    cfg["seed"] = seed
    cfg["profile"] = resolve_profile(profile)
    if env_backend:
        cfg["platformer"]["backend"] = env_backend
    paths = ensure_results(outdir)
    save_run_config(cfg, outdir)
    s = section(cfg, "bc", profile)
    psec = section(cfg, "platformer", profile)
    df = run_bc_suite(
        s["Ms"],
        platformer_kwargs(cfg, profile),
        seed,
        int(psec["train_level_seed_start"]),
        int(psec["eval_level_seed_start"]),
        paths["datasets"],
        paths["checkpoints"],
        paths["csv"],
        int(s["epochs"]),
        int(s["batch_size"]),
        float(s["lr"]),
        int(s["planner_horizon"]),
        int(s["planner_beam"]),
        int(s["eval_rollouts"]),
        s.get("max_expert_attempts"),
        device=resolve_device(device),
    )
    plot_bc(df, paths["figures"])
    return df


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--medium", action="store_true")
    p.add_argument("--outdir", default="results")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    p.add_argument("--env-backend", choices=["libre", "infinite_tux", "mario_ai_private", "gym_super_mario_bros"])
    args = p.parse_args(argv)
    if args.quick and args.medium:
        raise SystemExit("Use either --quick or --medium, not both.")
    run(args.config, args.seed, "medium" if args.medium else args.quick, args.outdir, args.device, args.env_backend)


if __name__ == "__main__":
    main()
