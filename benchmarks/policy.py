from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class CNNFeaturePolicy(nn.Module):
    """Softmax(phi_psi(image, aux)^T Omega), with a freezeable CNN feature map."""

    def __init__(self, obs_shape: tuple[int, int, int], feature_dim: int, hidden_dim: int, n_actions: int, aux_dim: int = 0):
        super().__init__()
        self.obs_shape = tuple(map(int, obs_shape))
        self.image_dim, self.aux_dim = int(np.prod(self.obs_shape)), int(aux_dim)
        self.obs_dim = self.image_dim + self.aux_dim
        self.feature_dim, self.hidden_dim, self.n_actions = int(feature_dim), int(hidden_dim), int(n_actions)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.obs_shape[0], 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(32 * 4 * 4 + self.aux_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(self.feature_dim + 1, self.n_actions, bias=False)

    def phi(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.reshape(-1, self.obs_dim)
        z = self.mlp(torch.cat([self.cnn(x[:, : self.image_dim].reshape(-1, *self.obs_shape)), x[:, self.image_dim :]], dim=-1))
        return torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.phi(obs))

    def freeze_backbone(self) -> None:
        for p in list(self.cnn.parameters()) + list(self.mlp.parameters()):
            p.requires_grad_(False)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False, device: str = "cpu") -> tuple[int, float]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = Categorical(logits=self(x))
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())


def clone_policy(policy: nn.Module) -> nn.Module:
    return deepcopy(policy).eval()


def save_policy(policy: CNNFeaturePolicy, path: str | Path, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "obs_shape": policy.obs_shape,
            "aux_dim": policy.aux_dim,
            "feature_dim": policy.feature_dim,
            "hidden_dim": policy.hidden_dim,
            "n_actions": policy.n_actions,
            "meta": meta or {},
        },
        path,
    )


def load_policy(path: str | Path, device: str = "cpu", freeze_backbone: bool = False) -> CNNFeaturePolicy:
    ckpt = torch.load(path, map_location=device)
    policy = CNNFeaturePolicy(tuple(ckpt["obs_shape"]), int(ckpt["feature_dim"]), int(ckpt["hidden_dim"]), int(ckpt["n_actions"]), int(ckpt["aux_dim"]))
    policy.load_state_dict(ckpt["state_dict"])
    if freeze_backbone:
        policy.freeze_backbone()
    return policy.to(device)
