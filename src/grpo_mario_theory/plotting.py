from __future__ import annotations

import os
import tempfile
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "grpo_mario_theory_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
if not os.access(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")), os.W_OK):
    os.environ["XDG_CACHE_HOME"] = str(_cache_root)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import save_dual


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def plot_population(curves: pd.DataFrame, hits: pd.DataFrame, fig_dir: str | Path) -> None:
    style()
    fig_dir = Path(fig_dir)
    g = _pick(curves, beta=1.0, q0=0.2)
    if g.empty:
        g = curves
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for eps, h in g.groupby("eps_smooth", sort=True):
        ax.plot(h["n"], np.maximum(h["q"], 1e-300), label=f"eps={eps:g}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("failure probability q_n")
    ax.set_title("Population Recurrence")
    ax.legend(ncols=2, fontsize=8)
    save_dual(fig, fig_dir / "population_q_vs_iter")
    plt.close(fig)

    h = _pick(hits, beta=1.0, q0=0.2)
    if h.empty:
        h = hits
    for xcol, name, xlabel in [
        ("loglog_inv_tau", "population_hitting_loglog", "log log(1/tau)"),
        ("log_inv_tau", "population_hitting_log", "log(1/tau)"),
    ]:
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        for eps, z in h.groupby("eps_smooth", sort=True):
            z = z.dropna(subset=["T"])
            ax.plot(z[xcol], z["T"], marker="o", ms=2.5, lw=1.2, label=f"eps={eps:g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("hitting time T(tau)")
        ax.set_title("Population Hitting Time")
        ax.legend(ncols=2, fontsize=8)
        save_dual(fig, fig_dir / name)
        plt.close(fig)


def plot_finite_group(summary: pd.DataFrame, fig_dir: str | Path) -> None:
    style()
    fig_dir = Path(fig_dir)
    g = _pick(summary, q0=0.2, eps_smooth=0.0)
    if g.empty:
        g = _pick(summary, q0=summary["q0"].iloc[0], eps_smooth=summary["eps_smooth"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for G, h in g.groupby("G", sort=True):
        ax.plot(h["n"], np.maximum(h["q_median"], 1e-300), label=f"G={G}")
        ax.fill_between(h["n"], np.maximum(h["q_p10"], 1e-300), np.maximum(h["q_p90"], 1e-300), alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("median q_n")
    ax.set_title("Finite-Group Recurrence")
    ax.legend(ncols=2, fontsize=8)
    save_dual(fig, fig_dir / "finite_group_q_vs_iter")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for G, h in g.groupby("G", sort=True):
        ax.plot(h["n"], h["degenerate_fraction"], label=f"G={G}")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("fraction skipped")
    ax.set_title("Degenerate Finite Groups")
    ax.legend(ncols=2, fontsize=8)
    save_dual(fig, fig_dir / "finite_group_degenerate_fraction")
    plt.close(fig)


def plot_candidate(curves: pd.DataFrame, fig_dir: str | Path) -> None:
    style()
    if curves.empty:
        return
    fig_dir = Path(fig_dir)
    K = sorted(curves["K"].unique())[-1]
    g = _pick(curves, K=K, beta=1.0, target_p0=0.2)
    if g.empty:
        g = curves[curves["K"] == K]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for eps, h in g.groupby("eps_smooth", sort=True):
        ax.plot(h["n"], np.maximum(h["q"], 1e-300), label=f"cand eps={eps:g}")
        ax.plot(h["n"], np.maximum(h["scalar_q"], 1e-300), ls="--", lw=1.0, color=ax.lines[-1].get_color(), alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("failure mass q_n")
    ax.set_title("Candidate Reweighting vs Scalar Recurrence")
    ax.legend(ncols=2, fontsize=8)
    save_dual(fig, fig_dir / "candidate_platformer_q_vs_iter")
    plt.close(fig)


def plot_bc(eval_df: pd.DataFrame, fig_dir: str | Path) -> None:
    style()
    if eval_df.empty:
        return
    g = eval_df.groupby("M", as_index=False).agg(p0=("success", "mean"), n=("success", "size"))
    g["lo"] = g["p0"] - 1.96 * np.sqrt(g["p0"] * (1 - g["p0"]) / g["n"])
    g["hi"] = g["p0"] + 1.96 * np.sqrt(g["p0"] * (1 - g["p0"]) / g["n"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.errorbar(
        g["M"],
        g["p0"],
        yerr=[g["p0"] - g["lo"].clip(lower=0), g["hi"].clip(upper=1) - g["p0"]],
        marker="o",
        capsize=3,
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("behavior-cloning samples M")
    ax.set_ylabel("measured warm-start p0")
    ax.set_title("Warm-Start Success Probability")
    save_dual(fig, Path(fig_dir) / "bc_warmstart_p0_vs_M")
    plt.close(fig)


def plot_grpo(df: pd.DataFrame, fig_dir: str | Path) -> None:
    style()
    if df.empty:
        return
    fig_dir = Path(fig_dir)
    _plot_grpo_family(
        _filter_preferred(df, eps_smooth=1e-4, G=max(df["G"]), beta=1.0),
        "M",
        "grpo_success_vs_iter_by_M",
        "GRPO by Warm Start",
        fig_dir,
    )
    _plot_grpo_family(
        _filter_preferred(df, M=max(df["M"]), G=max(df["G"]), beta=1.0),
        "eps_smooth",
        "grpo_success_vs_iter_by_eps",
        "GRPO by Smoothing",
        fig_dir,
    )
    _plot_grpo_family(
        _filter_preferred(df, M=max(df["M"]), eps_smooth=1e-4, beta=1.0),
        "G",
        "grpo_group_size_effect",
        "GRPO by Group Size",
        fig_dir,
    )


def _plot_grpo_family(df: pd.DataFrame, key: str, name: str, title: str, fig_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for value, h in df.groupby(key, sort=True):
        s = h.groupby("iteration", as_index=False).agg(success_rate=("success_rate", "mean"))
        ax.plot(s["iteration"], s["success_rate"], marker="o", ms=2.5, label=f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}")
    ax.set_xlabel("GRPO iteration")
    ax.set_ylabel("held-out success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(ncols=2, fontsize=8)
    save_dual(fig, fig_dir / name)
    plt.close(fig)


def _pick(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    out = df
    for k, v in kwargs.items():
        if k in out and v in set(out[k].unique()):
            out = out[out[k] == v]
    return out


def _filter_preferred(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    out = df
    for k, v in kwargs.items():
        if k not in out:
            continue
        values = np.asarray(sorted(out[k].unique()))
        chosen = v if v in set(values) else values[np.argmin(np.abs(values.astype(float) - float(v)))]
        out = out[out[k] == chosen]
    return out
