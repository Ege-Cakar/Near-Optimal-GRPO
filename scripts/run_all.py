from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grpo_mario_theory.utils import ensure_results, final_summary, load_config, run_sanity_checks, save_run_config

import run_bc_pretrain
import run_candidate_platformer
import run_finite_group_recurrence
import run_grpo_finetune
import run_population_recurrence


def run(config: str, seed: int, quick: bool, outdir: str, device: str = "cpu", env_backend: str | None = None):
    cfg = load_config(config)
    cfg["seed"] = seed
    if env_backend:
        cfg["platformer"]["backend"] = env_backend
    paths = ensure_results(outdir)
    save_run_config(cfg, outdir)
    print("Running sanity checks")
    run_sanity_checks()
    print("Running population recurrence")
    run_population_recurrence.run(config, seed, quick, outdir)
    print("Running finite-group recurrence")
    run_finite_group_recurrence.run(config, seed, quick, outdir)
    print("Running candidate platformer")
    run_candidate_platformer.run(config, seed, quick, outdir, env_backend)
    print("Running behavior cloning warm-start")
    run_bc_pretrain.run(config, seed, quick, outdir, device, env_backend)
    print("Running binary Mirror-GRPO fine-tuning")
    run_grpo_finetune.run(config, seed, quick, outdir, device, env_backend)
    summary = final_summary(paths["csv"])
    if not summary.empty:
        print("\nFinal summary table")
        print(summary.to_string(index=False, max_colwidth=50))
    print(f"\nWrote figures to {Path(outdir) / 'figures'}")
    print(f"Wrote CSVs to {Path(outdir) / 'csv'}")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--outdir", default="results")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    p.add_argument("--env-backend", choices=["libre", "infinite_tux", "mario_ai_private"])
    args = p.parse_args(argv)
    if args.quick and args.full:
        raise SystemExit("Use either --quick or --full, not both.")
    run(args.config, args.seed, quick=args.quick or not args.full, outdir=args.outdir, device=args.device, env_backend=args.env_backend)


if __name__ == "__main__":
    main()
