from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grpo_mario_theory.plotting import plot_population
from grpo_mario_theory.recurrence import hitting_times, run_population_grid
from grpo_mario_theory.utils import ensure_results, load_config, resolve_profile, save_run_config, section, tau_grid


def run(config: str, seed: int, profile: bool | str, outdir: str):
    cfg = load_config(config)
    cfg["seed"] = seed
    cfg["profile"] = resolve_profile(profile)
    paths = ensure_results(outdir)
    save_run_config(cfg, outdir)
    s = section(cfg, "population", profile)
    curves = run_population_grid(s["betas"], s["q0s"], s["eps_smooths"], int(s["max_iters"]))
    hits = hitting_times(curves, tau_grid(int(s.get("tau_min_exp", -40)), int(s["tau_points"])))
    curves.to_csv(paths["csv"] / "population_recurrence.csv", index=False)
    hits.to_csv(paths["csv"] / "population_hitting_times.csv", index=False)
    plot_population(curves, hits, paths["figures"])
    return curves, hits


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--medium", action="store_true")
    p.add_argument("--outdir", default="results")
    args = p.parse_args(argv)
    if args.quick and args.medium:
        raise SystemExit("Use either --quick or --medium, not both.")
    run(args.config, args.seed, "medium" if args.medium else args.quick, args.outdir)


if __name__ == "__main__":
    main()
