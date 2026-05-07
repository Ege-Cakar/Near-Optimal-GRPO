from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import logit_from_q, stable_sigmoid_pair


def finite_group_trajectory(
    seed: int,
    G: int,
    eps_smooth: float,
    beta: float,
    q0: float,
    max_iters: int,
) -> pd.DataFrame:
    """Stochastic finite-group analogue of the scalar recurrence."""
    rng = np.random.default_rng(seed)
    E = logit_from_q(q0)
    rows = []
    for n in range(max_iters + 1):
        p, q = stable_sigmoid_pair(E)
        K = int(rng.binomial(G, p))
        p_hat = K / G
        degenerate = eps_smooth == 0.0 and K in (0, G)
        sigma_hat = float(np.sqrt(p_hat * (1.0 - p_hat) + eps_smooth))
        rows.append(
            {
                "seed": seed,
                "G": G,
                "eps_smooth": eps_smooth,
                "beta": beta,
                "q0": q0,
                "n": n,
                "E": E,
                "p": p,
                "q": q,
                "K": K,
                "p_hat": p_hat,
                "sigma_hat": sigma_hat,
                "degenerate": degenerate,
            }
        )
        if not degenerate and sigma_hat > 0.0:
            E += 1.0 / (beta * sigma_hat)
    return pd.DataFrame(rows)


def run_finite_group_grid(
    seeds: int,
    Gs: list[int],
    eps_smooths: list[float],
    q0s: list[float],
    beta: float,
    max_iters: int,
    base_seed: int = 0,
) -> pd.DataFrame:
    frames = []
    run_id = 0
    for G in Gs:
        for eps in eps_smooths:
            for q0 in q0s:
                for s in range(seeds):
                    frames.append(finite_group_trajectory(base_seed + run_id, int(G), float(eps), beta, float(q0), max_iters))
                    run_id += 1
    return pd.concat(frames, ignore_index=True)


def finite_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["G", "eps_smooth", "q0", "n"], as_index=False)
        .agg(
            q_median=("q", "median"),
            q_p10=("q", lambda x: np.quantile(x, 0.10)),
            q_p90=("q", lambda x: np.quantile(x, 0.90)),
            degenerate_fraction=("degenerate", "mean"),
        )
    )


def hitting_distribution(df: pd.DataFrame, taus: tuple[float, ...] = (1e-2, 1e-4)) -> pd.DataFrame:
    rows = []
    group_cols = ["G", "eps_smooth", "q0", "seed"]
    for keys, g in df.groupby(group_cols, sort=False):
        for tau in taus:
            hit = g.loc[g["q"] <= tau, "n"]
            rows.append(dict(zip(group_cols, keys), tau=tau, T=float(hit.iloc[0]) if len(hit) else np.nan))
    return pd.DataFrame(rows)
