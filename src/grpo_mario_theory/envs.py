from __future__ import annotations

import numpy as np

from .infinite_tux_env import InfiniteTuxEnv
from .libre_platformer import LibrePlatformer, plan_action
from .gym_mario_env import GymSuperMarioBrosEnv
from .mario_ai_env import MarioAIPrivateEnv


def make_env(env_kwargs: dict):
    backend = env_kwargs.get("backend", "libre")
    kwargs = {k: v for k, v in env_kwargs.items() if k not in {"backend"}}
    if backend == "libre":
        allowed = {"height", "length", "difficulty", "max_steps", "obs_h", "obs_w"}
        return LibrePlatformer(**{k: v for k, v in kwargs.items() if k in allowed})
    if backend == "infinite_tux":
        return InfiniteTuxEnv(**kwargs)
    if backend == "mario_ai_private":
        return MarioAIPrivateEnv(**kwargs)
    if backend == "gym_super_mario_bros":
        return GymSuperMarioBrosEnv(**kwargs)
    raise ValueError(f"Unknown environment backend: {backend}")


def expert_action(env, env_kwargs: dict, planner_horizon: int, planner_beam: int) -> int:
    if env_kwargs.get("backend", "libre") == "libre":
        return plan_action(env, planner_horizon, planner_beam)
    if hasattr(env, "expert_action"):
        return env.expert_action()
    return obs_heuristic_action(env.last_obs, env.obs_h, env.obs_w)


def obs_heuristic_action(obs: np.ndarray | None, obs_h: int, obs_w: int) -> int:
    if obs is None:
        return 3
    grid = obs[: 4 * obs_h * obs_w].reshape(4, obs_h, obs_w)
    mid_y, start_x, end_x = obs_h // 2, obs_w // 2, min(obs_w, obs_w // 2 + 5)
    ahead_solid = grid[1, max(0, mid_y - 2) : min(obs_h, mid_y + 2), start_x:end_x].max() > 0
    ahead_hazard = grid[2, :, start_x:end_x].max() > 0
    floor_ahead = grid[1, min(obs_h - 1, mid_y + 3), start_x:end_x].mean()
    return 4 if ahead_solid or ahead_hazard or floor_ahead < 0.4 else 3
