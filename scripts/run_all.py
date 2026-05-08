from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grpo_mario_theory.utils import ensure_results, final_summary, load_config, resolve_profile, run_sanity_checks, save_run_config, section

import run_bc_pretrain
import run_candidate_platformer
import run_finite_group_recurrence
import run_grpo_finetune
import run_population_recurrence
import run_representation_pretrain
import run_warmstart_train


def run(config: str, seed: int, profile: bool | str, outdir: str, device: str = "cpu", env_backend: str | None = None, num_workers: int | None = None):
    cfg = load_config(config)
    cfg["seed"] = seed
    cfg["profile"] = resolve_profile(profile)
    if env_backend:
        cfg["platformer"]["backend"] = env_backend
    paths = ensure_results(outdir)
    save_run_config(cfg, outdir)
    print("Running sanity checks")
    run_sanity_checks()
    print("Running population recurrence")
    run_population_recurrence.run(config, seed, profile, outdir)
    print("Running finite-group recurrence")
    run_finite_group_recurrence.run(config, seed, profile, outdir)
    print("Running candidate platformer")
    run_candidate_platformer.run(config, seed, profile, outdir, env_backend)
    rep_enabled = section(cfg, "representation", profile).get("enabled", True)
    warm_enabled = section(cfg, "warmstart", profile).get("enabled", True)
    if rep_enabled:
        print("Training neural representation and measured-p0 linear heads")
        run_representation_pretrain.run(config, seed, profile, outdir, device, env_backend)
    elif warm_enabled:
        print("Training measured-p0 warm-start checkpoints")
        run_warmstart_train.run(config, seed, profile, outdir, device, env_backend)
    else:
        print("Running behavior cloning warm-start")
        run_bc_pretrain.run(config, seed, profile, outdir, device, env_backend)
    print("Running binary Mirror-GRPO fine-tuning")
    run_grpo_finetune.run(config, seed, profile, outdir, device, env_backend, num_workers)
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
    p.add_argument("--medium", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--outdir", default="results")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    p.add_argument("--num-workers", type=int, default=None, help="Worker processes for GRPO rollout/evaluation collection.")
    p.add_argument("--env-backend", choices=["libre", "infinite_tux", "mario_ai_private", "gym_super_mario_bros"])
    args = p.parse_args(argv)
    if sum([args.quick, args.medium, args.full]) > 1:
        raise SystemExit("Use only one of --quick, --medium, or --full.")
    profile = "medium" if args.medium else ("full" if args.full else "quick")
    run(args.config, args.seed, profile, args.outdir, args.device, args.env_backend, args.num_workers)


if __name__ == "__main__":
    main()
