from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.train_success import run_suite
from benchmarks.utils import resolve_device, save_run_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="benchmarks/config.yaml")
    p.add_argument("--outdir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--medium", action="store_true")
    p.add_argument("--env", action="append", dest="envs", help="Benchmark env to run; repeat or omit for config envs.")
    args = p.parse_args()
    if args.quick and args.medium:
        p.error("Use at most one of --quick or --medium.")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    outdir = Path(args.outdir or cfg.get("outdir", "results/benchmarks"))
    outdir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, outdir)

    profile = "quick" if args.quick else ("medium" if args.medium else "full")
    envs = args.envs or (cfg.get("medium_envs", cfg["envs"]) if profile == "medium" else cfg["envs"])
    run_suite(cfg, envs, outdir, profile=profile, seed=int(cfg.get("seed", 0)), device=resolve_device(args.device))


if __name__ == "__main__":
    main()
