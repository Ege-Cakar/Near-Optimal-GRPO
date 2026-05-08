from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .envs import expert_action, make_env
from .policy import LinearPolicy, evaluate_policy, save_policy
from .utils import seed_all


def collect_expert_dataset(
    M: int,
    env_kwargs: dict,
    seed: int,
    level_seed_start: int,
    dataset_dir: str | Path,
    planner_horizon: int,
    planner_beam: int,
    max_expert_attempts: int | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Collect M state-action pairs from the beam-search expert, with npz caching."""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(env_kwargs)
    tag = f"{env_kwargs.get('backend', 'libre')}_M{M}_seed{seed}_L{env_kwargs['length']}_D{int(1000 * env_kwargs['difficulty'])}_O{env.obs_dim}"
    path = dataset_dir / f"expert_{tag}.npz"
    traj_path = dataset_dir / f"expert_traj_{tag}.csv"
    if path.exists():
        data = np.load(path)
        traj = pd.read_csv(traj_path) if traj_path.exists() else pd.DataFrame()
        close = getattr(env, "close", None)
        if close:
            close()
        return data["obs"], data["actions"], traj
    if M == 0:
        obs = np.empty((0, env.obs_dim), dtype=np.float32)
        actions = np.empty((0,), dtype=np.int64)
        np.savez_compressed(path, obs=obs, actions=actions)
        close = getattr(env, "close", None)
        if close:
            close()
        return obs, actions, pd.DataFrame()

    seed_all(seed)
    obs_rows, action_rows, traj_rows = [], [], []
    pbar = tqdm(total=M, desc=f"expert M={M}", disable=not show_progress)
    level_idx, attempts, successes = 0, 0, 0
    attempt_limit = int(max_expert_attempts) if max_expert_attempts else max(1000, 10 * M)
    while len(action_rows) < M and attempts < attempt_limit:
        level_seed = level_seed_start + level_idx
        obs = env.reset(seed=seed + level_idx, level_seed=level_seed)
        done = False
        traj_obs, traj_actions = [], []
        while not done and len(action_rows) < M:
            a = expert_action(env, env_kwargs, planner_horizon, planner_beam)
            traj_obs.append(obs)
            traj_actions.append(a)
            obs, _, done, info = env.step(a)
        needed = M - len(action_rows)
        added = 0
        if env.info()["success"]:
            successes += 1
            take = min(needed, len(traj_actions))
            obs_rows.extend(traj_obs[:take])
            action_rows.extend(traj_actions[:take])
            pbar.update(take)
            added = take
        traj_rows.append({"level_seed": level_seed, "pairs": added, "used_for_bc": added > 0, "actions": " ".join(map(str, traj_actions)), **env.info()})
        pbar.set_postfix(attempts=attempts + 1, successes=successes, pairs=len(action_rows))
        level_idx += 1
        attempts += 1
    pbar.close()
    if len(action_rows) < M:
        raise RuntimeError(
            f"Only collected {len(action_rows)} successful expert pairs out of requested M={M} after {attempts} episodes. "
            "This backend needs a stronger warm-start/expert or a stage subset where the expert has nonzero success."
        )
    obs = np.asarray(obs_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.int64)
    traj = pd.DataFrame(traj_rows)
    np.savez_compressed(path, obs=obs, actions=actions)
    traj.to_csv(traj_path, index=False)
    close = getattr(env, "close", None)
    if close:
        close()
    return obs, actions, traj


def train_bc_policy(
    obs: np.ndarray,
    actions: np.ndarray,
    obs_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str = "cpu",
) -> tuple[LinearPolicy, list[float]]:
    seed_all(seed)
    policy = LinearPolicy(obs_dim).to(device)
    if len(actions) == 0:
        return policy, []
    ds = TensorDataset(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.nn.functional.cross_entropy(policy(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            count += len(yb)
        losses.append(total / max(count, 1))
    return policy, losses


def run_bc_suite(
    Ms: list[int],
    env_kwargs: dict,
    seed: int,
    level_seed_start: int,
    eval_level_seed_start: int,
    dataset_dir: str | Path,
    checkpoint_dir: str | Path,
    csv_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    planner_horizon: int,
    planner_beam: int,
    eval_rollouts: int,
    max_expert_attempts: int | None = None,
    show_progress: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    rows, loss_rows = [], []
    tmp_env = make_env(env_kwargs)
    obs_dim = tmp_env.obs_dim
    close = getattr(tmp_env, "close", None)
    if close:
        close()
    for M in tqdm(Ms, desc="BC checkpoints", disable=not show_progress):
        obs, actions, _ = collect_expert_dataset(M, env_kwargs, seed, level_seed_start, dataset_dir, planner_horizon, planner_beam, max_expert_attempts, show_progress)
        policy, losses = train_bc_policy(obs, actions, obs_dim, epochs, batch_size, lr, seed + M, device)
        ckpt_path = Path(checkpoint_dir) / f"bc_M{M}.pt"
        save_policy(policy, ckpt_path, {"M": M, "bc_losses": losses})
        metrics, eval_rows = evaluate_policy(env_kwargs, policy, eval_level_seed_start, eval_rollouts, seed + M, deterministic=False, device=device)
        for i, r in enumerate(eval_rows):
            rows.append({"M": M, "seed": seed, "level_seed": eval_level_seed_start + i, "return": r["success"], **r})
        for epoch, loss in enumerate(losses):
            loss_rows.append({"M": M, "epoch": epoch, "loss": loss})
        print(f"BC M={M}: measured p0={metrics['success_rate']:.3f}, avg_distance={metrics['avg_distance']:.2f}")
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(Path(csv_dir) / "bc_eval.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(Path(csv_dir) / "bc_losses.csv", index=False)
    return eval_df


def tiny_bc_loss_check() -> bool:
    seed_all(123)
    obs_dim = 32
    obs = torch.randn(96, obs_dim)
    y = (obs[:, :3].argmax(dim=1) + 1).long()
    policy = LinearPolicy(obs_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=5e-3)
    first = torch.nn.functional.cross_entropy(policy(obs), y).item()
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(policy(obs), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = torch.nn.functional.cross_entropy(policy(obs), y).item()
    return last < first
