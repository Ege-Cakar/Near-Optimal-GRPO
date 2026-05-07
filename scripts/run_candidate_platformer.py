from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grpo_mario_theory.candidate_experiment import run_candidate_suite
from grpo_mario_theory.plotting import plot_candidate
from grpo_mario_theory.utils import ensure_results, load_config, platformer_kwargs, save_run_config, section


def run(config: str, seed: int, quick: bool, outdir: str, env_backend: str | None = None):
    cfg = load_config(config)
    cfg["seed"] = seed
    if env_backend:
        cfg["platformer"]["backend"] = env_backend
    paths = ensure_results(outdir)
    save_run_config(cfg, outdir)
    s = section(cfg, "candidate", quick)
    bc = section(cfg, "bc", quick)
    curves = run_candidate_suite(
        platformer_kwargs(cfg, quick),
        s["betas"],
        s["eps_smooths"],
        s["candidate_bank_sizes"],
        s["target_p0s"],
        int(s["max_iters"]),
        int(s["level_seed"]),
        s["mutation_rates"],
        paths["csv"],
        seed,
        int(bc["planner_horizon"]),
        int(bc["planner_beam"]),
    )
    plot_candidate(curves, paths["figures"])
    return curves


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--outdir", default="results")
    p.add_argument("--env-backend", choices=["libre", "infinite_tux", "mario_ai_private"])
    args = p.parse_args(argv)
    run(args.config, args.seed, args.quick, args.outdir, args.env_backend)


if __name__ == "__main__":
    main()
