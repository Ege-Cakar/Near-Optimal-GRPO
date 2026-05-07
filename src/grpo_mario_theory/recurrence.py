from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import logit_from_q, stable_sigmoid_pair


@dataclass(frozen=True)
class RecurrenceSetting:
    beta: float
    q0: float
    eps_smooth: float
    max_iters: int


def population_recurrence(beta: float, q0: float, eps_smooth: float, max_iters: int) -> pd.DataFrame:
    """Exact binary population mirror recurrence in log-odds."""
    E = logit_from_q(q0)
    rows = []
    for n in range(max_iters + 1):
        p, q = stable_sigmoid_pair(E)
        rows.append({"beta": beta, "q0": q0, "eps_smooth": eps_smooth, "n": n, "E": E, "p": p, "q": q})
        if eps_smooth == 0.0 and (p * q <= 0.0 or q < 1e-300):
            break
        denom = np.sqrt(p * q + eps_smooth)
        if denom == 0.0 or not np.isfinite(denom):
            break
        E += 1.0 / (beta * denom)
    return pd.DataFrame(rows)


def run_population_grid(
    betas: list[float],
    q0s: list[float],
    eps_smooths: list[float],
    max_iters: int,
) -> pd.DataFrame:
    frames = [
        population_recurrence(float(beta), float(q0), float(eps), int(max_iters))
        for beta in betas
        for q0 in q0s
        for eps in eps_smooths
    ]
    return pd.concat(frames, ignore_index=True)


def hitting_times(curves: pd.DataFrame, taus: np.ndarray) -> pd.DataFrame:
    rows = []
    for keys, g in curves.groupby(["beta", "q0", "eps_smooth"], sort=False):
        q = g["q"].to_numpy()
        n = g["n"].to_numpy()
        for tau in taus:
            hit = n[q <= tau]
            rows.append(
                {
                    "beta": keys[0],
                    "q0": keys[1],
                    "eps_smooth": keys[2],
                    "tau": tau,
                    "loglog_inv_tau": np.log(np.log(1.0 / tau)),
                    "log_inv_tau": np.log(1.0 / tau),
                    "T": float(hit[0]) if len(hit) else np.nan,
                }
            )
    return pd.DataFrame(rows)
