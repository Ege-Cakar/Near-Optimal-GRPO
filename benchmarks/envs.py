from __future__ import annotations

import numpy as np


DIRS = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)], dtype=int)


def make_task(name: str, cfg: dict):
    spec = cfg["tasks"][name]
    kwargs = {k: v for k, v in spec.items() if k != "kind"}
    if spec["kind"] == "procgen":
        return ProcgenTask(name, **kwargs)
    if spec["kind"] == "minigrid":
        return MiniGridTask(name, **kwargs)
    if spec["kind"] == "gymnasium":
        return GymnasiumTask(name, **kwargs)
    raise ValueError(f"Unknown benchmark kind: {spec['kind']}")


class ProcgenTask:
    def __init__(self, name: str, env_name: str, distribution_mode: str = "easy", max_steps: int = 1000, scripted_expert: bool = True):
        self.name, self.env_name, self.distribution_mode, self.max_steps = name, env_name, distribution_mode, int(max_steps)
        self.scripted_expert = bool(scripted_expert)
        self.env, self.level_seed, self.elapsed, self.prev_action, self.success = None, None, 0, -1, False

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (3, 64, 64)

    @property
    def aux_dim(self) -> int:
        return self.n_actions

    @property
    def obs_dim(self) -> int:
        return int(np.prod(self.obs_shape)) + self.aux_dim

    @property
    def n_actions(self) -> int:
        if self.env is None:
            self._ensure_env(0)
        return int(self.env.action_space.n)

    def reset(self, seed: int, level_seed: int):
        self._ensure_env(level_seed)
        try:
            out = self.env.reset(seed=int(seed))
        except TypeError:
            out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        self.elapsed, self.prev_action, self.success = 0, -1, False
        return self._obs(obs)

    def step(self, action: int):
        out = self.env.step(int(action))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = out
        self.elapsed += 1
        self.prev_action = int(action)
        self.success = self.success or float(reward) > 0
        done = bool(done) or self.elapsed >= self.max_steps
        return self._obs(obs), float(reward), done, self.info()

    def info(self) -> dict:
        return {"success": self.success, "distance": float(self.success), "death": False, "timeout": self.elapsed >= self.max_steps and not self.success, "time_to_goal": self.elapsed if self.success else np.nan}

    def expert_action(self):
        if not self.scripted_expert or self.env_name not in {"coinrun", "jumper"}:
            return None
        right, right_up = self._combo_action(("RIGHT",)), self._combo_action(("RIGHT", "UP"))
        if right is None:
            return None
        jump = self.elapsed % (10 if self.env_name == "coinrun" else 6) < 2
        return right_up if jump and right_up is not None else right

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def _ensure_env(self, level_seed: int):
        if self.env is not None and self.level_seed == int(level_seed):
            return
        self.close()
        try:
            import gym
            import procgen  # noqa: F401
        except ImportError as e:
            raise RuntimeError("Install with `uv sync --extra benchmarks --extra procgen` to use Procgen.") from e
        self.level_seed = int(level_seed)
        self.env = gym.make(
            f"procgen:procgen-{self.env_name}-v0",
            start_level=self.level_seed,
            num_levels=1,
            distribution_mode=self.distribution_mode,
        )

    def _obs(self, obs):
        rgb = obs.get("rgb", obs) if isinstance(obs, dict) else obs
        x = np.asarray(rgb, dtype=np.float32) / 255.0
        if x.shape[:2] != (64, 64):
            x = _resize_mean(x, 64, 64)
        image = np.transpose(x, (2, 0, 1)).ravel()
        aux = np.zeros(self.n_actions, dtype=np.float32)
        if 0 <= self.prev_action < self.n_actions:
            aux[self.prev_action] = 1.0
        return np.concatenate([image, aux]).astype(np.float32)

    def _combo_action(self, buttons: tuple[str, ...]):
        combos = getattr(getattr(getattr(self.env, "unwrapped", self.env), "env", self.env), "combos", None)
        combos = combos or getattr(getattr(getattr(getattr(self.env, "unwrapped", self.env), "env", self.env), "env", self.env), "combos", None)
        if combos is None:
            return None
        target = tuple(buttons)
        for i, combo in enumerate(combos):
            if tuple(combo) == target:
                return i
        return None


class MiniGridTask:
    def __init__(self, name: str, env_id: str = "MiniGrid-DoorKey-8x8-v0", max_steps: int = 256, fully_observable: bool = True, obs_size: int = 8):
        self.name, self.env_id, self.max_steps = name, env_id, int(max_steps)
        self.fully_observable, self.obs_size = bool(fully_observable), int(obs_size)
        self.env, self.elapsed, self.prev_action, self.success = None, 0, -1, False

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (3, self.obs_size, self.obs_size) if self.fully_observable else (3, 7, 7)

    @property
    def aux_dim(self) -> int:
        return 6 + self.n_actions

    @property
    def obs_dim(self) -> int:
        return int(np.prod(self.obs_shape)) + self.aux_dim

    @property
    def n_actions(self) -> int:
        self._ensure_env()
        return int(self.env.action_space.n)

    def reset(self, seed: int, level_seed: int):
        self._ensure_env()
        obs, _info = self.env.reset(seed=int(level_seed))
        self.elapsed, self.prev_action, self.success = 0, -1, False
        return self._obs(obs)

    def step(self, action: int):
        obs, reward, terminated, truncated, _info = self.env.step(int(action))
        self.elapsed += 1
        self.prev_action = int(action)
        self.success = self.success or float(reward) > 0
        done = bool(terminated or truncated) or self.elapsed >= self.max_steps
        return self._obs(obs), float(reward), done, self.info()

    def info(self) -> dict:
        return {"success": self.success, "distance": float(self.success), "death": False, "timeout": self.elapsed >= self.max_steps and not self.success, "time_to_goal": self.elapsed if self.success else np.nan}

    def expert_action(self):
        plan = _minigrid_plan(self.env.unwrapped)
        return plan[0] if plan else 2

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def _ensure_env(self):
        if self.env is not None:
            return
        try:
            import gymnasium as gym
            import minigrid  # noqa: F401
            from minigrid.wrappers import FullyObsWrapper
        except ImportError as e:
            raise RuntimeError("Install with `uv sync --extra benchmarks` to use MiniGrid.") from e
        env = gym.make(self.env_id, max_steps=self.max_steps)
        self.env = FullyObsWrapper(env) if self.fully_observable else env

    def _obs(self, obs):
        image = np.asarray(obs["image"], dtype=np.float32)
        scales = np.array([10.0, 6.0, 3.0], dtype=np.float32)
        image = np.transpose(image / scales, (2, 0, 1)).ravel()
        direction = np.zeros(4, dtype=np.float32)
        direction[int(obs["direction"])] = 1.0
        state = np.array([float(self._has_key()), float(self._door_open())], dtype=np.float32)
        prev = np.zeros(self.n_actions, dtype=np.float32)
        if 0 <= self.prev_action < self.n_actions:
            prev[self.prev_action] = 1.0
        return np.concatenate([image, direction, state, prev]).astype(np.float32)

    def _has_key(self) -> bool:
        return bool(getattr(self.env.unwrapped, "carrying", None) is not None and self.env.unwrapped.carrying.type == "key")

    def _door_open(self) -> bool:
        door = _find(self.env.unwrapped.grid, "door")
        return _door_open(self.env.unwrapped.grid, door) if door is not None else False


class GymnasiumTask:
    def __init__(self, name: str, env_id: str, max_steps: int, success: str, expert: str | None = None, reward_shift: float = 0.0):
        self.name, self.env_id, self.max_steps, self.success_rule = name, env_id, int(max_steps), success
        self.expert, self.reward_shift = expert, float(reward_shift)
        self.env, self.elapsed, self.prev_action, self.success, self.raw_obs = None, 0, -1, False, None

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (self._obs_base_dim(), 1, 1)

    @property
    def aux_dim(self) -> int:
        return self.n_actions

    @property
    def obs_dim(self) -> int:
        return int(np.prod(self.obs_shape)) + self.aux_dim

    @property
    def n_actions(self) -> int:
        self._ensure_env()
        return int(self.env.action_space.n)

    def reset(self, seed: int, level_seed: int):
        self._ensure_env()
        self.raw_obs, _info = self.env.reset(seed=int(level_seed))
        self.elapsed, self.prev_action, self.success = 0, -1, False
        return self._obs()

    def step(self, action: int):
        self.raw_obs, reward, terminated, truncated, _info = self.env.step(int(action))
        self.elapsed += 1
        self.prev_action = int(action)
        self.success = self.success or self._success(bool(terminated), float(reward))
        done = bool(terminated or truncated) or self.elapsed >= self.max_steps
        return self._obs(), float(reward) + self.reward_shift, done, self.info()

    def info(self) -> dict:
        return {"success": self.success, "distance": float(self.success), "death": False, "timeout": self.elapsed >= self.max_steps and not self.success, "time_to_goal": self.elapsed if self.success else np.nan}

    def expert_action(self):
        if self.expert == "cartpole":
            _x, _xdot, theta, thetadot = np.asarray(self.raw_obs, dtype=float)
            return int(theta + 0.25 * thetadot > 0.0)
        if self.expert == "mountaincar":
            _pos, vel = np.asarray(self.raw_obs, dtype=float)
            return 2 if vel >= 0 else 0
        if self.expert == "acrobot":
            obs = np.asarray(self.raw_obs, dtype=float)
            return 2 if obs[4] + obs[5] >= 0 else 0
        return None

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def _ensure_env(self):
        if self.env is not None:
            return
        try:
            import gymnasium as gym
        except ImportError as e:
            raise RuntimeError("Install with `uv sync --extra benchmarks` to use Gymnasium tasks.") from e
        self.env = gym.make(self.env_id, max_episode_steps=self.max_steps)
        if not hasattr(self.env.action_space, "n"):
            raise ValueError(f"{self.env_id} must have a discrete action space.")

    def _obs_base_dim(self) -> int:
        self._ensure_env()
        space = self.env.observation_space
        return int(space.n if hasattr(space, "n") else np.prod(space.shape))

    def _obs(self):
        space = self.env.observation_space
        if hasattr(space, "n"):
            x = np.zeros(int(space.n), dtype=np.float32)
            x[int(self.raw_obs)] = 1.0
        else:
            x = np.asarray(self.raw_obs, dtype=np.float32).ravel()
            high = np.where(np.isfinite(space.high.ravel()), space.high.ravel(), 10.0)
            low = np.where(np.isfinite(space.low.ravel()), space.low.ravel(), -10.0)
            x = np.clip(2.0 * (x - low) / np.maximum(high - low, 1e-6) - 1.0, -5.0, 5.0)
        prev = np.zeros(self.n_actions, dtype=np.float32)
        if 0 <= self.prev_action < self.n_actions:
            prev[self.prev_action] = 1.0
        return np.concatenate([x.astype(np.float32), prev]).astype(np.float32)

    def _success(self, terminated: bool, reward: float) -> bool:
        x = np.asarray(self.raw_obs, dtype=float).ravel()
        if self.success_rule == "cartpole":
            return self.elapsed >= self.max_steps - 25 and not terminated
        if self.success_rule == "mountaincar":
            return bool(x[0] >= 0.5)
        if self.success_rule == "acrobot":
            return terminated and self.elapsed < self.max_steps
        if self.success_rule == "positive_reward":
            return reward > 0.0
        if self.success_rule == "terminated":
            return terminated
        raise ValueError(f"Unknown success rule: {self.success_rule}")


def _resize_mean(x: np.ndarray, h: int, w: int) -> np.ndarray:
    rows = np.array_split(x, h, axis=0)
    return np.array([[c.mean(axis=(0, 1)) for c in np.array_split(r, w, axis=1)] for r in rows], dtype=np.float32)


def _minigrid_plan(env) -> list[int]:
    grid, start, start_dir = env.grid, tuple(env.agent_pos), int(env.agent_dir)
    has_key = bool(getattr(env, "carrying", None) is not None and env.carrying.type == "key")
    key, door, goal = _find(grid, "key"), _find(grid, "door"), _find(grid, "goal")
    if goal is None:
        return []
    start_state = (start[0], start[1], start_dir, has_key, _door_open(grid, door) if door else True)
    seen, q = {start_state}, [(start_state, [])]
    while q:
        (x, y, d, has_key, door_open), path = q.pop(0)
        if (x, y) == goal:
            return path
        for action, state in _minigrid_next(grid, (x, y, d, has_key, door_open), key, door):
            if state not in seen:
                seen.add(state)
                q.append((state, path + [action]))
    return []


def _minigrid_next(grid, state, key, door):
    x, y, d, has_key, door_open = state
    yield 0, (x, y, (d - 1) % 4, has_key, door_open)
    yield 1, (x, y, (d + 1) % 4, has_key, door_open)
    fx, fy = (np.array([x, y]) + DIRS[d]).tolist()
    front = (fx, fy)
    if key is not None and front == key and not has_key:
        yield 3, (x, y, d, True, door_open)
    if door is not None and front == door and has_key and not door_open:
        yield 5, (x, y, d, has_key, True)
    if _passable(grid, front, door, door_open):
        yield 2, (fx, fy, d, has_key, door_open)


def _find(grid, typ: str):
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is not None and obj.type == typ:
                return (x, y)
    return None


def _door_open(grid, door) -> bool:
    if door is None:
        return True
    obj = grid.get(*door)
    return bool(obj and getattr(obj, "is_open", False))


def _passable(grid, pos, door, door_open):
    x, y = pos
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return False
    obj = grid.get(x, y)
    if obj is None:
        return True
    if obj.type in ("key", "goal"):
        return True
    if pos == door:
        return door_open
    return False
