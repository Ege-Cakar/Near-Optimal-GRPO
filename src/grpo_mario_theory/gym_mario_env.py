from __future__ import annotations

import numpy as np

from .libre_platformer import ACTION_NAMES


class GymSuperMarioBrosEnv:
    """Private wrapper for Kautenja/gym-super-mario-bros RandomStages."""

    def __init__(
        self,
        max_steps: int = 1200,
        obs_h: int = 9,
        obs_w: int = 13,
        gym_mario_env_id: str = "SuperMarioBrosRandomStages-v0",
        gym_mario_stages: list[str] | None = None,
        gym_mario_movement: str = "SIMPLE_MOVEMENT",
        **_ignored,
    ):
        try:
            import gym_super_mario_bros
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
            from nes_py.wrappers import JoypadSpace
        except ImportError as e:
            raise RuntimeError("Install the optional backend with `uv sync --extra gym-mario`.") from e
        if gym_mario_movement != "SIMPLE_MOVEMENT":
            raise ValueError("This linear-policy suite expects SIMPLE_MOVEMENT's 7-action space.")
        kwargs = {"stages": gym_mario_stages} if gym_mario_stages else {}
        self.env = JoypadSpace(gym_super_mario_bros.make(gym_mario_env_id, **kwargs), SIMPLE_MOVEMENT)
        self.max_steps, self.obs_h, self.obs_w = int(max_steps), int(obs_h), int(obs_w)
        self.elapsed, self.last_raw_obs, self.last_obs, self.last_info = 0, None, None, {}

    @property
    def obs_dim(self) -> int:
        return 4 * self.obs_h * self.obs_w + 8

    def reset(self, seed: int | None = None, level_seed: int | None = None, checkpoint=None) -> np.ndarray:
        if checkpoint is not None:
            raise NotImplementedError("GymSuperMarioBrosEnv does not support state snapshots.")
        seed = int(seed if level_seed is None else level_seed)
        try:
            out = self.env.reset(seed=seed)
        except TypeError:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            out = self.env.reset()
        raw, info = out if isinstance(out, tuple) and len(out) == 2 else (out, {})
        self.elapsed, self.last_raw_obs = 0, raw
        self.last_info = self._info(info, done=False)
        self.last_obs = self._features(raw, self.last_info)
        return self.last_obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        out = self.env.step(int(action))
        if len(out) == 5:
            raw, _reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            raw, _reward, done, info = out
        self.elapsed += 1
        done = bool(done) or self.elapsed >= self.max_steps
        self.last_raw_obs = raw
        self.last_info = self._info(info, done)
        self.last_obs = self._features(raw, self.last_info)
        return self.last_obs, float(self.last_info["success"] and done), done, self.last_info

    def expert_action(self) -> int:
        if self.elapsed % 28 in range(18, 24):
            return 4
        if self.elapsed > 20 and self.last_info.get("x_delta", 1.0) < 0.5:
            return 4
        return 3

    def info(self) -> dict:
        return dict(self.last_info)

    def close(self) -> None:
        self.env.close()

    def render_ascii(self) -> str:
        return "<gym-super-mario-bros uses NES frames; ASCII rendering is not implemented.>"

    def _info(self, info: dict, done: bool) -> dict:
        x = float(info.get("x_pos", 0.0))
        last_x = float(self.last_info.get("x_pos_raw", x))
        success = bool(info.get("flag_get", False))
        timeout = bool(done and not success and self.elapsed >= self.max_steps)
        return {
            "success": success,
            "death": bool(done and not success and not timeout),
            "timeout": timeout,
            "distance": x,
            "time_to_goal": float(self.elapsed) if success else np.nan,
            "elapsed": self.elapsed,
            "x_pos_raw": x,
            "x_delta": x - last_x,
            "world": int(info.get("world", 0) or 0),
            "stage": int(info.get("stage", 0) or 0),
            "life": int(info.get("life", 0) or 0),
        }

    def _features(self, raw: np.ndarray, info: dict) -> np.ndarray:
        gray = raw.astype(np.float32).mean(axis=2) / 255.0
        rows = np.array_split(gray, self.obs_h, axis=0)
        small = np.array([[c.mean() for c in np.array_split(r, self.obs_w, axis=1)] for r in rows], dtype=np.float32)
        bins = np.stack([small < 0.25, (small >= 0.25) & (small < 0.5), (small >= 0.5) & (small < 0.75), small >= 0.75])
        extras = np.array(
            [
                info["distance"] / 4000.0,
                info["x_delta"] / 16.0,
                self.elapsed / max(1, self.max_steps),
                info["world"] / 8.0,
                info["stage"] / 4.0,
                info["life"] / 3.0,
                float(info["death"]),
                float(info["timeout"]),
            ],
            dtype=np.float32,
        )
        return np.concatenate([bins.astype(np.float32).ravel(), extras])


assert len(ACTION_NAMES) == 7
