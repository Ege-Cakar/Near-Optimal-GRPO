from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grpo_mario_theory.grpo_finetune import run_grpo_suite
from grpo_mario_theory.plotting import plot_grpo
from grpo_mario_theory.utils import ensure_results, load_config, platformer_kwargs, resolve_device, save_run_config, section


def run(config: str, seed: int, quick: bool, outdir: str, device: str = "cpu", env_backend: str | None = None):
    cfg = load_config(config)
    cfg["seed"] = seed
    if env_backend:
        cfg["platformer"]["backend"] = env_backend
    paths = ensure_results(outdir)
    save_run_config(cfg, outdir)
    s = section(cfg, "grpo", quick)
    psec = section(cfg, "platformer", quick)
    df = run_grpo_suite(
        platformer_kwargs(cfg, quick),
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
    )
    plot_grpo(df, paths["figures"])
    return df


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--outdir", default="results")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    p.add_argument("--env-backend", choices=["libre", "infinite_tux", "mario_ai_private"])
    args = p.parse_args(argv)
    run(args.config, args.seed, args.quick, args.outdir, args.device, args.env_backend)


if __name__ == "__main__":
    main()
