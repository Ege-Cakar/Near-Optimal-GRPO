from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical
from tqdm import tqdm

from .envs import expert_action, make_env
from .libre_platformer import N_ACTIONS
from .policy import LinearPolicy, clone_policy, rollout_policy, save_policy
from .utils import seed_all


def run_warmstart_training(
    env_kwargs: dict,
    checkpoint_dir: str | Path,
    csv_dir: str | Path,
    seed: int,
    train_level_seed_start: int,
    eval_level_seed_start: int,
    train_levels: int,
    eval_levels: int,
    max_scan: int,
    iterations: int,
    rollouts_per_iter: int,
    epochs_per_iter: int,
    batch_size: int,
    lr: float,
    target_p0s: list[float],
    require_target_p0: float | None,
    planner_horizon: int,
    planner_beam: int,
    rollin_expert_start: float,
    rollin_expert_end: float,
    logit_scales: list[float],
    max_dataset: int,
    rl_rollouts_per_iter: int = 0,
    rl_updates_per_iter: int = 0,
    rl_lr: float = 1e-3,
    rl_gamma: float = 0.99,
    entropy_coef: float = 0.01,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a linear policy until measured held-out p0 reaches requested targets."""
    seed_all(seed)
    checkpoint_dir, csv_dir = Path(checkpoint_dir), Path(csv_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = _expert_solved_levels(env_kwargs, train_level_seed_start, train_levels, max_scan, seed, planner_horizon, planner_beam, "train")
    eval_seeds = _expert_solved_levels(env_kwargs, eval_level_seed_start, eval_levels, max_scan, seed + 10_000, planner_horizon, planner_beam, "eval")
    pd.DataFrame(
        [{"split": "train", "level_seed": s} for s in train_seeds] + [{"split": "eval", "level_seed": s} for s in eval_seeds]
    ).to_csv(csv_dir / "warmstart_level_sets.csv", index=False)

    env = make_env(env_kwargs)
    policy = LinearPolicy(env.obs_dim).to(device)
    obs_rows, action_rows, history = [], [], []
    rng = np.random.default_rng(seed)
    try:
        for it in tqdm(range(iterations + 1), desc="warm-start training"):
            if it == 0:
                obs, actions = _collect_expert(env, env_kwargs, train_seeds, seed, planner_horizon, planner_beam)
            else:
                t = (it - 1) / max(1, iterations - 1)
                expert_prob = (1 - t) * rollin_expert_start + t * rollin_expert_end
                levels = rng.choice(train_seeds, size=min(rollouts_per_iter, len(train_seeds)), replace=False).tolist()
                obs, actions = _collect_dagger(env, env_kwargs, policy, levels, seed + 1000 * it, expert_prob, planner_horizon, planner_beam, device)
            obs_rows.append(obs)
            action_rows.append(actions)
            x, y = _bounded_dataset(obs_rows, action_rows, max_dataset, rng)
            _train(policy, x, y, epochs_per_iter, batch_size, lr, device)
            if rl_rollouts_per_iter and rl_updates_per_iter:
                levels = rng.choice(train_seeds, size=min(rl_rollouts_per_iter, len(train_seeds)), replace=False).tolist()
                _train_shaped_rl(env, policy, levels, seed + 20_000 + it, rl_updates_per_iter, rl_lr, rl_gamma, entropy_coef, device)
            row = _best_scaled_eval(env_kwargs, policy, eval_seeds, seed + 50_000 + it, logit_scales, checkpoint_dir, it, device)
            row.update({"iteration": it, "train_pairs": len(x), "new_pairs": len(actions)})
            history.append(row)
            if require_target_p0 is not None and row["p0"] >= require_target_p0:
                break
    finally:
        close = getattr(env, "close", None)
        if close:
            close()

    hist = pd.DataFrame(history)
    hist.to_csv(csv_dir / "warmstart_eval.csv", index=False)
    targets = _write_target_checkpoints(hist, target_p0s, checkpoint_dir, csv_dir)
    if require_target_p0 is not None and hist["p0"].max() < require_target_p0:
        raise RuntimeError(f"Warm-start training reached max p0={hist['p0'].max():.3f}, below required {require_target_p0:.3f}.")
    return hist, targets


def load_warmstart_specs(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df[df["met"]].to_dict("records")


def _expert_solved_levels(env_kwargs, start, count, max_scan, seed, horizon, beam, split) -> list[int]:
    env = make_env(env_kwargs)
    levels = []
    try:
        for j in tqdm(range(max_scan), desc=f"select {split} levels"):
            level_seed = int(start + j)
            _obs = env.reset(seed=seed + j, level_seed=level_seed)
            done = False
            while not done:
                a = expert_action(env, env_kwargs, horizon, beam)
                _obs, _, done, info = env.step(a)
            if info["success"]:
                levels.append(level_seed)
                if len(levels) >= count:
                    return levels
    finally:
        close = getattr(env, "close", None)
        if close:
            close()
    raise RuntimeError(f"Only found {len(levels)} expert-solved {split} levels; requested {count}.")


def _collect_expert(env, env_kwargs, levels, seed, horizon, beam):
    obs_rows, actions = [], []
    for i, level_seed in enumerate(levels):
        obs = env.reset(seed=seed + i, level_seed=level_seed)
        done = False
        while not done:
            a = expert_action(env, env_kwargs, horizon, beam)
            obs_rows.append(obs)
            actions.append(a)
            obs, _, done, _info = env.step(a)
    return np.asarray(obs_rows, np.float32), np.asarray(actions, np.int64)


def _collect_dagger(env, env_kwargs, policy, levels, seed, expert_prob, horizon, beam, device):
    rng = np.random.default_rng(seed)
    obs_rows, labels = [], []
    for i, level_seed in enumerate(levels):
        obs = env.reset(seed=seed + i, level_seed=level_seed)
        done = False
        while not done:
            label = expert_action(env, env_kwargs, horizon, beam)
            action = label if rng.random() < expert_prob else policy.act(obs, deterministic=False, device=device)[0]
            obs_rows.append(obs)
            labels.append(label)
            obs, _, done, _info = env.step(action)
    return np.asarray(obs_rows, np.float32), np.asarray(labels, np.int64)


def _train(policy, obs, actions, epochs, batch_size, lr, device):
    ds = TensorDataset(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    counts = np.bincount(actions, minlength=N_ACTIONS).clip(min=1)
    weights = torch.as_tensor(len(actions) / (N_ACTIONS * counts), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.nn.functional.cross_entropy(policy(xb), yb, weight=weights)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()


def _train_shaped_rl(env, policy, levels, seed, updates, lr, gamma, entropy_coef, device):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for u in range(updates):
        obs_rows, action_rows, return_rows = [], [], []
        for i, level_seed in enumerate(levels):
            obs = env.reset(seed=seed + 1000 * u + i, level_seed=level_seed)
            done, rewards, traj_obs, traj_actions, last_dist = False, [], [], [], env.info().get("distance", 0.0)
            while not done:
                action = policy.act(obs, deterministic=False, device=device)[0]
                traj_obs.append(obs)
                traj_actions.append(action)
                obs, _, done, info = env.step(action)
                dist = info["distance"]
                r = 0.2 * (dist - last_dist) - 0.002
                r += 10.0 if info["success"] else (-2.0 if info["death"] else (-0.5 if info["timeout"] else 0.0))
                rewards.append(float(r))
                last_dist = dist
            obs_rows.extend(traj_obs)
            action_rows.extend(traj_actions)
            return_rows.extend(_discounted(rewards, gamma))
        returns = torch.as_tensor(return_rows, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std().clamp_min(1e-6))
        obs_t = torch.as_tensor(np.asarray(obs_rows, np.float32), dtype=torch.float32, device=device)
        act_t = torch.as_tensor(action_rows, dtype=torch.long, device=device)
        dist = Categorical(logits=policy(obs_t))
        loss = -(returns * dist.log_prob(act_t)).mean() - entropy_coef * dist.entropy().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        opt.step()


def _discounted(rewards, gamma):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return list(reversed(out))


def _best_scaled_eval(env_kwargs, policy, eval_seeds, seed, scales, checkpoint_dir, iteration, device):
    rows = []
    for scale in scales:
        p = clone_policy(policy).to(device)
        with torch.no_grad():
            p.linear.weight.mul_(float(scale))
        metrics, _ = evaluate_policy_on_levels(env_kwargs, p, eval_seeds, seed, deterministic=False, device=device)
        rows.append((metrics["success_rate"], float(scale), metrics, p))
    p0, scale, metrics, best_policy = max(rows, key=lambda x: (x[0], x[1]))
    path = Path(checkpoint_dir) / f"warmstart_iter{iteration:03d}_p{p0:.3f}.pt"
    save_policy(best_policy, path, {"iteration": iteration, "p0": p0, "logit_scale": scale, "eval_metrics": metrics})
    return {"p0": p0, "logit_scale": scale, "checkpoint_path": str(path), **metrics}


def evaluate_policy_on_levels(env_kwargs, policy, levels, seed, deterministic, device):
    env = make_env(env_kwargs)
    rows = []
    try:
        for i, level_seed in enumerate(levels):
            tr = rollout_policy(env, policy, level_seed, seed + i, deterministic, device)
            rows.append({"level_seed": level_seed, **tr})
    finally:
        close = getattr(env, "close", None)
        if close:
            close()
    s = np.array([r["success"] for r in rows], dtype=float)
    return {
        "success_rate": float(s.mean()),
        "q_eval": float(1.0 - s.mean()),
        "avg_distance": float(np.mean([r["distance"] for r in rows])),
        "death_rate": float(np.mean([r["death"] for r in rows])),
        "timeout_rate": float(np.mean([r["timeout"] for r in rows])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in rows])),
    }, rows


def _bounded_dataset(obs_rows, action_rows, max_dataset, rng):
    obs, actions = np.concatenate(obs_rows), np.concatenate(action_rows)
    if len(actions) > max_dataset:
        idx = rng.choice(len(actions), size=max_dataset, replace=False)
        obs, actions = obs[idx], actions[idx]
    return obs.astype(np.float32), actions.astype(np.int64)


def _write_target_checkpoints(hist, targets, checkpoint_dir, csv_dir):
    rows = []
    for target in targets:
        ok = hist[hist["p0"] >= float(target)]
        pick = ok.sort_values("p0").iloc[0] if len(ok) else hist.sort_values("p0").iloc[-1]
        label = f"p{int(round(1000 * float(target))):04d}"
        dst = Path(checkpoint_dir) / f"warmstart_{label}.pt"
        shutil.copyfile(pick["checkpoint_path"], dst)
        rows.append(
            {
                "label": label,
                "target_p0": float(target),
                "achieved_p0": float(pick["p0"]),
                "met": bool(pick["p0"] >= float(target)),
                "iteration": int(pick["iteration"]),
                "checkpoint_path": str(dst),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(Path(csv_dir) / "warmstart_targets.csv", index=False)
    return out
