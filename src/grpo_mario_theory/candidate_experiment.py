from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .envs import expert_action, make_env
from .libre_platformer import N_ACTIONS
from .recurrence import population_recurrence


def expert_sequence(env_kwargs: dict, level_seed: int, seed: int, horizon: int = 18, beam: int = 24) -> list[int]:
    env = make_env(env_kwargs)
    env.reset(seed=seed, level_seed=level_seed)
    actions = []
    done = False
    while not done:
        a = expert_action(env, env_kwargs, horizon, beam)
        actions.append(a)
        _, _, done, _ = env.step(a)
    close = getattr(env, "close", None)
    if close:
        close()
    return actions


def evaluate_sequence(env_kwargs: dict, level_seed: int, seed: int, actions: list[int]) -> dict:
    env = make_env(env_kwargs)
    env.reset(seed=seed, level_seed=level_seed)
    info = getattr(env, "last_info", {})
    for a in actions[: env.max_steps]:
        _, _, done, info = env.step(a)
        if done:
            break
    else:
        info = env.info() if hasattr(env, "info") else info
    close = getattr(env, "close", None)
    if close:
        close()
    return {"success": int(info["success"]), "distance": float(info["distance"]), "death": bool(info["death"])}


def generate_candidate_bank(
    K: int,
    env_kwargs: dict,
    level_seed: int,
    seed: int,
    mutation_rates: list[float],
    planner_horizon: int = 18,
    planner_beam: int = 24,
) -> pd.DataFrame:
    """Create a reproducible bank of successful and failed candidate action sequences."""
    rng = np.random.default_rng(seed)
    expert = expert_sequence(env_kwargs, level_seed, seed, planner_horizon, planner_beam)
    max_steps = env_kwargs["max_steps"]
    expert = (expert + [0] * max_steps)[:max_steps]
    expert_result = evaluate_sequence(env_kwargs, level_seed, seed, expert)
    if env_kwargs.get("backend") == "gym_super_mario_bros" and not expert_result["success"]:
        raise RuntimeError(
            "Gym Super Mario Bros candidate experiment needs a successful warm-start candidate, "
            "but the current scripted expert failed. Use a checkpoint/candidate bank with measured p0 > 0, "
            "or run the publishable LibrePlatformer / Mario AI Framework backend."
        )
    candidates: list[list[int]] = [expert, [0] * max_steps, [5] * max_steps]
    while len(candidates) < K:
        if rng.random() < 0.7:
            seq = expert.copy()
            rate = float(rng.choice(mutation_rates))
            mask = rng.random(max_steps) < rate
            random_actions = rng.integers(0, N_ACTIONS, size=max_steps)
            seq = [int(random_actions[i]) if mask[i] else int(a) for i, a in enumerate(seq)]
        else:
            seq = rng.integers(0, N_ACTIONS, size=max_steps).astype(int).tolist()
        candidates.append(seq)
    rows = [
        {
            "candidate_id": 0,
            "success": expert_result["success"],
            "distance": expert_result["distance"],
            "death": expert_result["death"],
            "actions": " ".join(map(str, expert)),
        }
    ]
    for i, seq in tqdm(enumerate(candidates[1:K], start=1), total=max(K - 1, 0), desc=f"candidate eval K={K}", leave=False):
        result = evaluate_sequence(env_kwargs, level_seed, seed + i, seq)
        rows.append({"candidate_id": i, "success": result["success"], "distance": result["distance"], "death": result["death"], "actions": " ".join(map(str, seq))})
    bank = pd.DataFrame(rows)
    if bank["success"].sum() == 0:
        raise RuntimeError("Candidate bank has no successful trajectory; reduce level difficulty or increase planner horizon.")
    if bank["success"].sum() == len(bank):
        raise RuntimeError("Candidate bank has no failed trajectory; increase mutation/random candidates.")
    return bank


def initial_weights(success: np.ndarray, target_p0: float) -> np.ndarray:
    s = success.astype(bool)
    if s.sum() == 0 or (~s).sum() == 0:
        raise ValueError("Need at least one success and one failure candidate.")
    w = np.zeros_like(success, dtype=float)
    w[s] = target_p0 / s.sum()
    w[~s] = (1.0 - target_p0) / (~s).sum()
    return w


def mirror_candidate_curves(
    success: np.ndarray,
    init_w: np.ndarray,
    beta: float,
    eps_smooth: float,
    max_iters: int,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    r = success.astype(float)
    logw = np.log(init_w)
    curves, weights = [], []
    for n in range(max_iters + 1):
        w = _softmax_logw(logw)
        p = float(np.clip(np.dot(w, r), 0.0, 1.0))
        q = float(max(0.0, 1.0 - p))
        curves.append({"n": n, "p": p, "q": q, "degenerate": eps_smooth == 0.0 and (p <= 0.0 or p >= 1.0)})
        weights.append(w)
        if eps_smooth == 0.0 and (p <= 0.0 or p >= 1.0):
            break
        sigma = np.sqrt(p * q + eps_smooth)
        if sigma == 0.0:
            break
        logw = logw + r / (beta * sigma)
    return pd.DataFrame(curves), weights


def run_candidate_suite(
    env_kwargs: dict,
    betas: list[float],
    eps_smooths: list[float],
    candidate_bank_sizes: list[int],
    target_p0s: list[float],
    max_iters: int,
    level_seed: int,
    mutation_rates: list[float],
    csv_dir: str | Path,
    seed: int,
    planner_horizon: int = 18,
    planner_beam: int = 24,
) -> pd.DataFrame:
    csv_dir = Path(csv_dir)
    weights_path = csv_dir / "candidate_platformer_weights.csv"
    curve_rows = []
    with open(weights_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["K", "beta", "eps_smooth", "target_p0", "candidate_id", "success", "initial_weight", "n", "weight", "p", "q"],
        )
        writer.writeheader()
        for K in tqdm(candidate_bank_sizes, desc="candidate bank sizes"):
            bank = generate_candidate_bank(K, env_kwargs, level_seed, seed + K, mutation_rates, planner_horizon, planner_beam)
            bank.to_csv(csv_dir / f"candidate_bank_K{K}.csv", index=False)
            success = bank["success"].to_numpy(dtype=int)
            for target_p0 in target_p0s:
                init_w = initial_weights(success, float(target_p0))
                for beta in betas:
                    for eps in eps_smooths:
                        curves, weights = mirror_candidate_curves(success, init_w, float(beta), float(eps), max_iters)
                        scalar_eps = population_recurrence(float(beta), 1.0 - float(target_p0), float(eps), max_iters)[["n", "p", "q"]]
                        scalar_map = scalar_eps.set_index("n")
                        for _, row in curves.iterrows():
                            n = int(row["n"])
                            curve_rows.append(
                                {
                                    "K": K,
                                    "beta": beta,
                                    "eps_smooth": eps,
                                    "target_p0": target_p0,
                                    "n": n,
                                    "p": row["p"],
                                    "q": row["q"],
                                    "scalar_p": scalar_map.loc[n, "p"] if n in scalar_map.index else np.nan,
                                    "scalar_q": scalar_map.loc[n, "q"] if n in scalar_map.index else np.nan,
                                    "degenerate": row["degenerate"],
                                }
                            )
                            for cid, w in enumerate(weights[n]):
                                writer.writerow(
                                    {
                                        "K": K,
                                        "beta": beta,
                                        "eps_smooth": eps,
                                        "target_p0": target_p0,
                                        "candidate_id": cid,
                                        "success": int(success[cid]),
                                        "initial_weight": init_w[cid],
                                        "n": n,
                                        "weight": float(w),
                                        "p": row["p"],
                                        "q": row["q"],
                                    }
                                )
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(csv_dir / "candidate_platformer_curves.csv", index=False)
    return curves


def candidate_matches_scalar_check() -> bool:
    success = np.array([1, 0, 0, 0, 0], dtype=int)
    init_w = initial_weights(success, 0.2)
    cand, _ = mirror_candidate_curves(success, init_w, beta=1.0, eps_smooth=1e-8, max_iters=8)
    scalar = population_recurrence(beta=1.0, q0=0.8, eps_smooth=1e-8, max_iters=8)
    return np.allclose(cand["q"].to_numpy(), scalar["q"].to_numpy(), rtol=1e-10, atol=1e-12)


def _softmax_logw(logw: np.ndarray) -> np.ndarray:
    z = logw - np.max(logw)
    w = np.exp(z)
    return w / w.sum()
