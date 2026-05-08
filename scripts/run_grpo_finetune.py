from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grpo_mario_theory.grpo_finetune import run_grpo_suite
from grpo_mario_theory.plotting import plot_grpo
from grpo_mario_theory.utils import ensure_results, load_config, platformer_kwargs, resolve_device, resolve_profile, save_run_config, section
from grpo_mario_theory.warmstart_train import load_warmstart_specs


def run(config: str, seed: int, profile: bool | str, outdir: str, device: str = "cpu", env_backend: str | None = None, num_workers: int | None = None):
    cfg = load_config(config)
    cfg["seed"] = seed
    cfg["profile"] = resolve_profile(profile)
    if env_backend:
        cfg["platformer"]["backend"] = env_backend
    paths = ensure_results(outdir)
    save_run_config(cfg, outdir)
    s = section(cfg, "grpo", profile)
    psec = section(cfg, "platformer", profile)
    specs = load_warmstart_specs(paths["csv"] / "warmstart_targets.csv") if s.get("use_warmstart_targets", False) else []
    if s.get("use_warmstart_targets", False) and not specs:
        raise FileNotFoundError("Missing measured warm-start targets; run scripts/run_representation_pretrain.py first.")
    df = run_grpo_suite(
        platformer_kwargs(cfg, profile),
        paths["checkpoints"],
        paths["csv"],
        s["Ms"],
        s["eps_smooths"],
        s["Gs"],
        s["betas"],
        int(s["seeds"]),
        int(s["iterations"]),
        int(s["prompts"]),
        int(s["eval_rollouts"]),
        int(psec["train_level_seed_start"]) + 50_000,
        int(psec["eval_level_seed_start"]) + 50_000,
        float(s["lr"]),
        int(s["update_epochs"]),
        float(s["beta_kl"]),
        seed,
        device=resolve_device(device),
        num_workers=int(num_workers if num_workers is not None else s.get("num_workers", 1)),
        checkpoint_specs=specs or None,
    )
    plot_grpo(df, paths["figures"])
    return df


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--medium", action="store_true")
    p.add_argument("--outdir", default="results")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    p.add_argument("--num-workers", type=int, default=None, help="Worker processes for GRPO rollout/evaluation collection.")
    p.add_argument("--env-backend", choices=["libre", "infinite_tux", "mario_ai_private", "gym_super_mario_bros"])
    args = p.parse_args(argv)
    if args.quick and args.medium:
        raise SystemExit("Use either --quick or --medium, not both.")
    run(args.config, args.seed, "medium" if args.medium else args.quick, args.outdir, args.device, args.env_backend, args.num_workers)


if __name__ == "__main__":
    main()
