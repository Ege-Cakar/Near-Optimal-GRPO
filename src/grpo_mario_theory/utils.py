from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    """Resolve cpu/cuda/mps/auto and fail clearly when an accelerator is unavailable."""
    device = device.lower()
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested --device mps, but torch.backends.mps.is_available() is false.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but torch.cuda.is_available() is false.")
    if device != "cpu" and not (device.startswith("cuda") or device == "mps"):
        raise ValueError(f"Unsupported device: {device}")
    return device


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_profile(profile: bool | str) -> str:
    profile = "quick" if profile is True else ("full" if profile is False else str(profile))
    if profile not in ("quick", "medium", "full"):
        raise ValueError(f"Unknown profile: {profile}")
    return profile


def section(cfg: dict[str, Any], name: str, profile: bool | str) -> dict[str, Any]:
    """Return a config section with quick/medium overrides applied."""
    profile = resolve_profile(profile)
    out = {k: v for k, v in cfg[name].items() if k not in ("quick", "medium")}
    if profile in ("quick", "medium"):
        out.update(cfg[name].get(profile, {}))
    return out


def ensure_results(outdir: str | Path) -> dict[str, Path]:
    root = Path(outdir)
    paths = {
        "root": root,
        "figures": root / "figures",
        "csv": root / "csv",
        "checkpoints": root / "checkpoints",
        "datasets": root / "datasets",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_run_config(cfg: dict[str, Any], outdir: str | Path) -> None:
    with open(Path(outdir) / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def platformer_kwargs(cfg: dict[str, Any], profile: bool | str) -> dict[str, Any]:
    s = section(cfg, "platformer", profile)
    keys = [
        "backend",
        "height",
        "length",
        "difficulty",
        "max_steps",
        "obs_h",
        "obs_w",
        "infinite_tux_dir",
        "mario_ai_dir",
        "mario_ai_level_set",
        "gym_mario_env_id",
        "gym_mario_stages",
        "gym_mario_movement",
        "java_cmd",
        "javac_cmd",
        "level_type",
    ]
    return {k: s[k] for k in keys if k in s}


def stable_sigmoid_pair(E: float) -> tuple[float, float]:
    """Return p=sigmoid(E), q=1-p without avoidable cancellation."""
    if E >= 0:
        z = np.exp(-E) if E < 745 else 0.0
        p = 1.0 / (1.0 + z)
        q = z / (1.0 + z)
    else:
        z = np.exp(E)
        p = z / (1.0 + z)
        q = 1.0 / (1.0 + z)
    return float(p), float(q)


def logit_from_q(q: float) -> float:
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError(f"q0 must be in (0,1), got {q}")
    return float(np.log1p(-q) - np.log(q))


def tau_grid(tau_min_exp: int = -40, points: int = 80) -> np.ndarray:
    return np.logspace(-2, tau_min_exp, int(points))


def binomial_ci(p: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    half = 1.96 * np.sqrt(max(p * (1.0 - p), 0.0) / n)
    return max(0.0, p - half), min(1.0, p + half)


def save_dual(fig, path_no_ext: str | Path) -> None:
    path = Path(path_no_ext)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def final_summary(csv_dir: str | Path) -> pd.DataFrame:
    """Build a compact summary from experiment CSVs when present."""
    csv_dir = Path(csv_dir)
    rows: list[dict[str, Any]] = []

    pop = csv_dir / "population_recurrence.csv"
    if pop.exists():
        df = pd.read_csv(pop)
        for keys, g in df.groupby(["beta", "q0", "eps_smooth"], sort=False):
            rows.append(_summary_row("population", _setting(("beta", "q0", "eps"), keys), g, "n", np.nan))

    fg = csv_dir / "finite_group_recurrence.csv"
    if fg.exists():
        df = pd.read_csv(fg)
        for keys, g in df.groupby(["G", "eps_smooth", "q0"], sort=False):
            med = g.groupby("n", as_index=False).agg(
                p=("p", "median"), q=("q", "median"), degenerate=("degenerate", "mean")
            )
            rows.append(_summary_row("finite_group", _setting(("G", "eps", "q0"), keys), med, "n", med["degenerate"].mean()))

    cand = csv_dir / "candidate_platformer_curves.csv"
    if cand.exists():
        df = pd.read_csv(cand)
        for keys, g in df.groupby(["K", "beta", "eps_smooth", "target_p0"], sort=False):
            rows.append(_summary_row("candidate", _setting(("K", "beta", "eps", "target_p0"), keys), g, "n", np.nan))

    grpo = csv_dir / "grpo_eval.csv"
    if grpo.exists():
        df = pd.read_csv(grpo)
        for keys, g in df.groupby(["M", "eps_smooth", "G", "beta", "seed"], sort=False):
            rows.append(_summary_row("grpo", _setting(("M", "eps", "G", "beta", "seed"), keys), g, "iteration", g["skipped_group_fraction"].mean()))

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(csv_dir / "final_summary.csv", index=False)
    return out


def _summary_row(experiment: str, setting: str, g: pd.DataFrame, n_col: str, skipped: float) -> dict[str, Any]:
    first, last = g.iloc[0], g.iloc[-1]
    q_col = "q" if "q" in g else "q_eval"
    p_col = "p" if "p" in g else "success_rate"
    return {
        "experiment": experiment,
        "setting": setting,
        "initial_p": first.get(p_col, np.nan),
        "final_p": last.get(p_col, np.nan),
        "initial_q": first.get(q_col, np.nan),
        "final_q": max(0.0, float(last.get(q_col, np.nan))),
        "hitting_time_tau_1e-2": _hit(g, q_col, n_col, 1e-2),
        "hitting_time_tau_1e-4": _hit(g, q_col, n_col, 1e-4),
        "skipped_group_fraction": skipped,
    }


def _hit(g: pd.DataFrame, q_col: str, n_col: str, tau: float) -> float:
    hit = g.loc[g[q_col] <= tau, n_col]
    return float(hit.iloc[0]) if len(hit) else np.nan


def _setting(names: tuple[str, ...], values: tuple[Any, ...]) -> str:
    return ", ".join(f"{k}={float(v):g}" if isinstance(v, (float, np.floating)) else f"{k}={v}" for k, v in zip(names, values))


def run_sanity_checks() -> None:
    """Small assertions covering the required numerical and environment invariants."""
    from .bc_pretrain import tiny_bc_loss_check
    from .candidate_experiment import candidate_matches_scalar_check
    from .libre_platformer import LibrePlatformer
    from .recurrence import population_recurrence

    df = population_recurrence(beta=1.0, q0=0.2, eps_smooth=0.0, max_iters=8)
    assert np.allclose(df["p"] + df["q"], 1.0)
    assert np.all(np.diff(df["q"]) <= 1e-15)
    assert candidate_matches_scalar_check()
    env = LibrePlatformer(length=32, difficulty=0.2, max_steps=80)
    obs = env.reset(seed=0, level_seed=1)
    obs2, reward, done, info = env.step(1)
    assert obs.shape == obs2.shape and reward in (0.0, 1.0) and isinstance(done, bool) and "distance" in info
    assert tiny_bc_loss_check()
