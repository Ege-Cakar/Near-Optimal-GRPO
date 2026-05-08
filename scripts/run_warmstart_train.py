from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grpo_mario_theory.plotting import plot_warmstart
from grpo_mario_theory.utils import ensure_results, load_config, platformer_kwargs, resolve_device, resolve_profile, save_run_config, section
from grpo_mario_theory.warmstart_train import run_warmstart_training


def run(config: str, seed: int, profile: bool | str, outdir: str, device: str = "cpu", env_backend: str | None = None):
    cfg = load_config(config)
    cfg["seed"] = seed
    cfg["profile"] = resolve_profile(profile)
    if env_backend:
        cfg["platformer"]["backend"] = env_backend
    paths = ensure_results(outdir)
    save_run_config(cfg, outdir)
    s = section(cfg, "warmstart", profile)
    psec = section(cfg, "platformer", profile)
    hist, targets = run_warmstart_training(
        platformer_kwargs(cfg, profile),
        paths["checkpoints"],
        paths["csv"],
        seed,
        int(psec["train_level_seed_start"]),
        int(psec["eval_level_seed_start"]),
        int(s["train_levels"]),
        int(s["eval_levels"]),
        int(s["max_scan"]),
        int(s["iterations"]),
        int(s["rollouts_per_iter"]),
        int(s["epochs_per_iter"]),
        int(s["batch_size"]),
        float(s["lr"]),
        s["target_p0s"],
        s.get("require_target_p0"),
        int(s["planner_horizon"]),
        int(s["planner_beam"]),
        float(s["rollin_expert_start"]),
        float(s["rollin_expert_end"]),
        s["logit_scales"],
        int(s["max_dataset"]),
        int(s.get("rl_rollouts_per_iter", 0)),
        int(s.get("rl_updates_per_iter", 0)),
        float(s.get("rl_lr", 1e-3)),
        float(s.get("rl_gamma", 0.99)),
        float(s.get("entropy_coef", 0.01)),
        resolve_device(device),
    )
    plot_warmstart(hist, paths["figures"])
    print(targets.to_string(index=False))
    return hist, targets


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
