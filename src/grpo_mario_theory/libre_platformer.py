from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMPTY, SOLID, HAZARD, GOAL = 0, 1, 2, 3
N_ACTIONS = 7
ACTION_NAMES = ["noop", "right", "right_jump", "right_run", "right_run_jump", "left", "jump"]


@dataclass
class Snapshot:
    x: float
    y: float
    vx: float
    vy: float
    on_ground: bool
    elapsed: int
    done: bool
    success: bool
    dead: bool


class LibrePlatformer:
    """A deterministic libre tile-grid platformer for headless binary-reward experiments."""

    def __init__(
        self,
        height: int = 12,
        length: int = 64,
        difficulty: float = 0.35,
        max_steps: int = 240,
        obs_h: int = 9,
        obs_w: int = 13,
    ):
        self.height, self.length, self.difficulty = int(height), int(length), float(difficulty)
        self.max_steps, self.obs_h, self.obs_w = int(max_steps), int(obs_h), int(obs_w)
        self.player_w, self.player_h = 0.72, 0.90
        self.goal_x = self.length - 3
        self.tiles = generate_level(0, self.difficulty, self.length, self.height)
        self.reset(seed=0, level_seed=0)

    @property
    def obs_dim(self) -> int:
        return 4 * self.obs_h * self.obs_w + 4

    def reset(self, seed: int | None = None, level_seed: int | None = None, checkpoint: Snapshot | None = None) -> np.ndarray:
        if level_seed is not None:
            self.tiles = generate_level(level_seed, self.difficulty, self.length, self.height)
            self.goal_x = self.length - 3
        self.rng = np.random.default_rng(seed)
        self.x, self.y, self.vx, self.vy = 1.5, 1.0, 0.0, 0.0
        self.on_ground, self.elapsed, self.done, self.success, self.dead = True, 0, False, False, False
        if checkpoint is not None:
            self.set_state_snapshot(checkpoint)
        return self.observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            return self.observation(), float(self.success), True, self.info()
        action = int(action)
        jump = action in (2, 4, 6)
        right, run, left = action in (1, 2, 3, 4), action in (3, 4), action == 5
        if jump and self.on_ground:
            self.vy = 0.92
            self.on_ground = False
        ax = (0.13 if run else 0.085) if right else (-0.09 if left else 0.0)
        self.vx = np.clip(self.vx + ax, -0.28, 0.58 if run else 0.38)
        if not (right or left):
            self.vx *= 0.80
        self.vy = max(self.vy - 0.055, -0.85)
        self.on_ground = False
        self._move(self.vx, 0.0)
        self._move(0.0, self.vy)
        self.elapsed += 1
        self._update_terminal()
        return self.observation(), float(self.success and self.done), self.done, self.info()

    def get_state_snapshot(self) -> Snapshot:
        return Snapshot(self.x, self.y, self.vx, self.vy, self.on_ground, self.elapsed, self.done, self.success, self.dead)

    def set_state_snapshot(self, s: Snapshot) -> None:
        self.x, self.y, self.vx, self.vy = s.x, s.y, s.vx, s.vy
        self.on_ground, self.elapsed, self.done, self.success, self.dead = s.on_ground, s.elapsed, s.done, s.success, s.dead

    def observation(self) -> np.ndarray:
        cx, cy = int(self.x + self.player_w / 2), int(self.y + self.player_h / 2)
        grid = np.zeros((4, self.obs_h, self.obs_w), dtype=np.float32)
        for iy, wy in enumerate(range(cy + self.obs_h // 2, cy - self.obs_h // 2 - 1, -1)):
            for ix, wx in enumerate(range(cx - self.obs_w // 3, cx - self.obs_w // 3 + self.obs_w)):
                grid[self.tile_at(wx, wy), iy, ix] = 1.0
        extras = np.array([self.vx / 0.58, self.vy / 0.92, float(self.on_ground), self.elapsed / self.max_steps], dtype=np.float32)
        return np.concatenate([grid.ravel(), extras])

    def tile_at(self, x: int | float, y: int | float) -> int:
        xi, yi = int(np.floor(x)), int(np.floor(y))
        if xi < 0:
            return SOLID
        if xi >= self.length or yi < 0 or yi >= self.height:
            return EMPTY
        return int(self.tiles[self.height - 1 - yi, xi])

    def render_ascii(self) -> str:
        chars = {EMPTY: " ", SOLID: "#", HAZARD: "!", GOAL: "G"}
        canvas = np.array([[chars[int(t)] for t in row] for row in self.tiles])
        px, py = int(self.x), self.height - 1 - int(self.y)
        if 0 <= py < self.height and 0 <= px < self.length:
            canvas[py, px] = "A"
        return "\n".join("".join(row) for row in canvas)

    def save_frame_png(self, path: str) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.imshow(self.tiles, cmap=ListedColormap(["white", "black", "red", "green"]), vmin=0, vmax=3)
        ax.add_patch(plt.Rectangle((self.x, self.height - self.y - self.player_h), self.player_w, self.player_h, color="tab:blue"))
        ax.set_axis_off()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def info(self) -> dict:
        return {
            "success": self.success,
            "death": self.dead,
            "timeout": self.done and not self.success and not self.dead,
            "distance": max(0.0, self.x - 1.5),
            "time_to_goal": self.elapsed if self.success else np.nan,
        }

    def _move(self, dx: float, dy: float) -> None:
        steps = max(1, int(max(abs(dx), abs(dy)) / 0.04) + 1)
        for _ in range(steps):
            nx, ny = self.x + dx / steps, self.y + dy / steps
            if dx and self._collides(nx, self.y):
                self.vx = 0.0
            elif dx:
                self.x = nx
            if dy and self._collides(self.x, ny):
                if dy < 0:
                    self.on_ground = True
                self.vy = 0.0
            elif dy:
                self.y = ny

    def _collides(self, x: float, y: float) -> bool:
        xs = (x + 0.04, x + self.player_w - 0.04)
        ys = (y + 0.04, y + self.player_h - 0.04)
        return any(self.tile_at(tx, ty) == SOLID for tx in xs for ty in ys)

    def _touches(self, tile: int) -> bool:
        xs = (self.x + 0.08, self.x + self.player_w - 0.08)
        ys = (self.y + 0.06, self.y + self.player_h - 0.06)
        return any(self.tile_at(tx, ty) == tile for tx in xs for ty in ys)

    def _update_terminal(self) -> None:
        self.success = self._touches(GOAL) or self.x >= self.goal_x
        self.dead = self.y < -1.5 or self._touches(HAZARD)
        self.done = self.success or self.dead or self.elapsed >= self.max_steps


def generate_level(seed: int, difficulty: float = 0.35, length: int = 64, height: int = 12) -> np.ndarray:
    """Generate a deterministic, non-proprietary tile level."""
    rng = np.random.default_rng(seed)
    tiles = np.zeros((height, length), dtype=np.int8)
    tiles[-1, :] = SOLID
    x = 5
    while x < length - 8:
        u = rng.random()
        if u < 0.08 + 0.12 * difficulty:
            w = int(rng.integers(1, 2 + int(3 * difficulty)))
            tiles[-1, x : min(length - 4, x + w)] = EMPTY
            x += w + int(rng.integers(2, 5))
        elif u < 0.18 + 0.18 * difficulty:
            tiles[-2, x] = HAZARD
            x += int(rng.integers(3, 7))
        elif u < 0.28 + 0.12 * difficulty:
            h = int(rng.integers(1, 3))
            tiles[-2 : -2 - h : -1, x] = SOLID
            x += int(rng.integers(3, 6))
        else:
            x += 1
    tiles[-1, :4] = SOLID
    tiles[-1, -6:] = SOLID
    tiles[-2, length - 3] = GOAL
    return tiles


def plan_action(env: LibrePlatformer, horizon: int = 18, beam_width: int = 24) -> int:
    """Beam-search expert used only to create warm-start demonstrations."""
    h = heuristic_action(env)
    if horizon <= 0 or h == 3:
        return h
    root = env.get_state_snapshot()
    beam = [(root, None, env.x)]
    actions = (4, 3, 2, 1, 6, 0, 5)
    for _ in range(horizon):
        nxt = []
        for state, first, _score in beam:
            env.set_state_snapshot(state)
            if env.done:
                nxt.append((state, first, _score))
                continue
            for a in actions:
                env.set_state_snapshot(state)
                _, _, done, info = env.step(a)
                score = _planner_score(env, info)
                first_a = a if first is None else first
                if done and info["success"]:
                    env.set_state_snapshot(root)
                    return first_a
                nxt.append((env.get_state_snapshot(), first_a, score))
        beam = sorted(nxt, key=lambda z: z[2], reverse=True)[:beam_width]
    env.set_state_snapshot(root)
    return int(beam[0][1] if beam and beam[0][1] is not None else 3)


def heuristic_action(env: LibrePlatformer) -> int:
    """Fast rule used directly on easy sections and as a fallback expert."""
    cx = int(env.x + env.player_w)
    threat = False
    for wx in range(cx + 1, min(env.length - 1, cx + 5)):
        gap = env.tile_at(wx, 0) != SOLID
        hazard = env.tile_at(wx, 1) == HAZARD
        obstacle = env.tile_at(wx, 1) == SOLID or env.tile_at(wx, 2) == SOLID
        threat = threat or gap or hazard or obstacle
    if threat:
        return 4 if env.on_ground or env.vy > -0.2 else 3
    return 3


def _planner_score(env: LibrePlatformer, info: dict) -> float:
    if info["success"]:
        return 1e6 - env.elapsed
    if info["death"]:
        return -1e6 + env.x
    ahead = min(env.length - 1, int(env.x + 2))
    hazard_penalty = 20.0 if env.tile_at(ahead, 1) == HAZARD else 0.0
    return 10.0 * env.x + 1.5 * env.vx + 0.2 * env.y - 0.03 * env.elapsed - hazard_penalty
