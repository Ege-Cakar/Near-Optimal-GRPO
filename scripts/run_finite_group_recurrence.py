from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grpo_mario_theory.finite_group import finite_group_summary, hitting_distribution, run_finite_group_grid
from grpo_mario_theory.plotting import plot_finite_group
from grpo_mario_theory.utils import ensure_results, load_config, save_run_config, section


def run(config: str, seed: int, quick: bool, outdir: str):
    cfg = load_config(config)
    cfg["seed"] = seed
    paths = ensure_results(outdir)
    save_run_config(cfg, outdir)
    s = section(cfg, "finite_group", quick)
    df = run_finite_group_grid(int(s["seeds"]), s["Gs"], s["eps_smooths"], s["q0s"], float(s["beta"]), int(s["max_iters"]), seed)
    summary = finite_group_summary(df)
    df.to_csv(paths["csv"] / "finite_group_recurrence.csv", index=False)
    summary.to_csv(paths["csv"] / "finite_group_summary.csv", index=False)
    hitting_distribution(df).to_csv(paths["csv"] / "finite_group_hitting_times.csv", index=False)
    plot_finite_group(summary, paths["figures"])
    return df, summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--outdir", default="results")
    args = p.parse_args(argv)
    run(args.config, args.seed, args.quick, args.outdir)


if __name__ == "__main__":
    main()
