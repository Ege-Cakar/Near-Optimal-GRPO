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
        self.n_actions = int(n_actions)
        self.linear = nn.Linear(self.obs_dim + 1, n_actions, bias=False)
        self.policy_kind = "linear_phi_bias"

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


class NeuralFeaturePolicy(nn.Module):
    """Softmax(phi_psi(x)^T Omega), with trainable/freezeable neural phi_psi."""

    def __init__(self, obs_dim: int, feature_dim: int = 128, hidden_dim: int = 256, n_actions: int = N_ACTIONS):
        super().__init__()
        self.obs_dim, self.feature_dim, self.hidden_dim, self.n_actions = int(obs_dim), int(feature_dim), int(hidden_dim), int(n_actions)
        self.backbone = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(self.feature_dim + 1, self.n_actions, bias=False)
        self.policy_kind = "neural_feature_head"

    def phi(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.backbone(obs)
        ones = torch.ones(*z.shape[:-1], 1, dtype=z.dtype, device=z.device)
        return torch.cat([z, ones], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.phi(obs))

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False, device: str = "cpu") -> tuple[int, float]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self(x)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())


def clone_policy(policy: nn.Module) -> nn.Module:
    return deepcopy(policy).eval()


def policy_payload(policy: nn.Module) -> dict:
    return {
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "obs_dim": int(policy.obs_dim),
        "policy": getattr(policy, "policy_kind", "linear_phi_bias"),
        "feature_dim": int(getattr(policy, "feature_dim", 0)),
        "hidden_dim": int(getattr(policy, "hidden_dim", 0)),
        "n_actions": int(getattr(policy, "n_actions", N_ACTIONS)),
    }


def policy_from_payload(ckpt: dict, device: str = "cpu", freeze_backbone: bool = False) -> nn.Module:
    kind = ckpt.get("policy", "linear_phi_bias")
    if kind == "neural_feature_head":
        policy = NeuralFeaturePolicy(int(ckpt["obs_dim"]), int(ckpt["feature_dim"]), int(ckpt["hidden_dim"]), int(ckpt.get("n_actions", N_ACTIONS)))
        if freeze_backbone:
            policy.freeze_backbone()
    elif kind == "linear_phi_bias":
        policy = LinearPolicy(int(ckpt["obs_dim"]), int(ckpt.get("n_actions", N_ACTIONS)))
    else:
        raise ValueError(f"Unknown policy kind: {kind}")
    policy.load_state_dict(ckpt["state_dict"])
    return policy.to(device)


def save_policy(policy: nn.Module, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = policy_payload(policy)
    payload["meta"] = meta or {}
    torch.save(payload, path)


def load_policy(path: str | Path, device: str = "cpu", freeze_backbone: bool = False) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    return policy_from_payload(ckpt, device=device, freeze_backbone=freeze_backbone)


def rollout_policy(
    env,
    policy: nn.Module,
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
    policy: nn.Module,
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
