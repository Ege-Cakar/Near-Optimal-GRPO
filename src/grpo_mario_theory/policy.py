from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .envs import make_env
from .libre_platformer import N_ACTIONS


class LinearPolicy(nn.Module):
    """Linear softmax policy with logits phi(x)^T Omega and phi(x)=[x,1]."""

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.linear = nn.Linear(self.obs_dim + 1, n_actions, bias=False)

    def phi(self, obs: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(*obs.shape[:-1], 1, dtype=obs.dtype, device=obs.device)
        return torch.cat([obs, ones], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.phi(obs))

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False, device: str = "cpu") -> tuple[int, float]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self(x)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())


def clone_policy(policy: LinearPolicy) -> LinearPolicy:
    return deepcopy(policy).eval()


def save_policy(policy: LinearPolicy, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "obs_dim": policy.obs_dim, "policy": "linear_phi_bias", "meta": meta or {}}, path)


def load_policy(path: str | Path, device: str = "cpu") -> LinearPolicy:
    ckpt = torch.load(path, map_location=device)
    policy = LinearPolicy(int(ckpt["obs_dim"]))
    policy.load_state_dict(ckpt["state_dict"])
    return policy.to(device)


def rollout_policy(
    env,
    policy: LinearPolicy,
    level_seed: int,
    seed: int,
    deterministic: bool = False,
    device: str = "cpu",
) -> dict:
    obs = env.reset(seed=seed, level_seed=level_seed)
    obses, actions, logps = [], [], []
    done = False
    while not done:
        a, logp = policy.act(obs, deterministic=deterministic, device=device)
        obses.append(obs)
        actions.append(a)
        logps.append(logp)
        obs, _, done, info = env.step(a)
    return {
        "obs": np.asarray(obses, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "logps": np.asarray(logps, dtype=np.float32),
        "success": float(info["success"]),
        "distance": float(info["distance"]),
        "death": float(info["death"]),
        "timeout": float(info["timeout"]),
        "time_to_goal": float(info["time_to_goal"]) if info["success"] else np.nan,
        "episode_length": len(actions),
    }


def evaluate_policy(
    env_kwargs: dict,
    policy: LinearPolicy,
    level_seed_start: int,
    n_rollouts: int,
    seed: int,
    deterministic: bool = False,
    device: str = "cpu",
) -> tuple[dict, list[dict]]:
    env = make_env(env_kwargs)
    rows = []
    for i in range(n_rollouts):
        tr = rollout_policy(env, policy, level_seed_start + i, seed + 100_000 + i, deterministic, device)
        rows.append({k: tr[k] for k in ("success", "distance", "death", "timeout", "time_to_goal", "episode_length")})
    close = getattr(env, "close", None)
    if close:
        close()
    s = np.array([r["success"] for r in rows], dtype=float)
    return {
        "success_rate": float(s.mean()) if len(s) else np.nan,
        "q_eval": float(1.0 - s.mean()) if len(s) else np.nan,
        "avg_distance": float(np.mean([r["distance"] for r in rows])) if rows else np.nan,
        "death_rate": float(np.mean([r["death"] for r in rows])) if rows else np.nan,
        "timeout_rate": float(np.mean([r["timeout"] for r in rows])) if rows else np.nan,
        "mean_episode_length": float(np.mean([r["episode_length"] for r in rows])) if rows else np.nan,
    }, rows
