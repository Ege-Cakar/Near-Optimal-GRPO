from __future__ import annotations

import atexit
import subprocess
from pathlib import Path

import numpy as np


class InfiniteTuxEnv:
    """Headless Python wrapper for the libre Infinite Tux Java environment."""

    def __init__(
        self,
        height: int = 15,
        length: int = 128,
        difficulty: float = 0.35,
        max_steps: int = 400,
        obs_h: int = 9,
        obs_w: int = 13,
        infinite_tux_dir: str = "external/infinite-tux",
        java_cmd: str = "java",
        javac_cmd: str = "javac",
        level_type: int = 0,
    ):
        self.height, self.length, self.difficulty = int(height), int(length), float(difficulty)
        self.max_steps, self.obs_h, self.obs_w = int(max_steps), int(obs_h), int(obs_w)
        self.infinite_tux_dir = Path(infinite_tux_dir)
        self.java_cmd, self.javac_cmd, self.level_type = java_cmd, javac_cmd, int(level_type)
        self.proc: subprocess.Popen[str] | None = None
        self.last_obs: np.ndarray | None = None
        self.last_info: dict = {}

    @property
    def obs_dim(self) -> int:
        return 4 * self.obs_h * self.obs_w + 4

    def reset(self, seed: int | None = None, level_seed: int | None = None, checkpoint=None) -> np.ndarray:
        if checkpoint is not None:
            raise NotImplementedError("InfiniteTuxEnv does not support Python-side state snapshots.")
        self._ensure_proc()
        assert self.proc and self.proc.stdin
        level_seed = int(seed if level_seed is None else level_seed)
        diff = max(0, min(10, int(round(self.difficulty * 10))))
        self.proc.stdin.write(f"RESET {level_seed} {diff} {self.level_type} {self.max_steps} {self.obs_h} {self.obs_w} {self.length}\n")
        self.proc.stdin.flush()
        obs, _reward, _done, info = self._read_obs()
        self.last_obs, self.last_info = obs, info
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self._ensure_proc()
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(f"STEP {int(action)}\n")
        self.proc.stdin.flush()
        obs, reward, done, info = self._read_obs()
        self.last_obs, self.last_info = obs, info
        return obs, reward, done, info

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write("CLOSE\n")
                self.proc.stdin.flush()
        finally:
            self.proc.terminate()
            self.proc = None

    def render_ascii(self) -> str:
        return "<InfiniteTuxEnv uses the Java environment; ASCII rendering is not implemented.>"

    def info(self) -> dict:
        return dict(self.last_info)

    def _ensure_proc(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        classes = build_infinite_tux(self.infinite_tux_dir, self.javac_cmd)
        resources = self.infinite_tux_dir / "deb" / "src" / "main" / "resources"
        home = Path("results") / "infinite_tux_home"
        home.mkdir(parents=True, exist_ok=True)
        cp = f"{classes}:{resources}"
        self.proc = subprocess.Popen(
            [_java_tool(self.java_cmd, "java"), "-Djava.awt.headless=true", f"-Duser.home={home.resolve()}", "-cp", cp, "com.mojang.mario.ai.HeadlessServer"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)

    def _read_obs(self) -> tuple[np.ndarray, float, bool, dict]:
        assert self.proc and self.proc.stdout
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"Infinite Tux server exited without an observation. stderr:\n{err}")
        if line.startswith("ERR"):
            raise RuntimeError(line.strip())
        parts = line.rstrip("\n").split("\t")
        if parts[0] != "OBS" or len(parts) != 10:
            raise RuntimeError(f"Malformed Infinite Tux response: {line[:200]}")
        done, reward = bool(int(parts[1])), float(parts[2])
        success, death, timeout = bool(int(parts[3])), bool(int(parts[4])), bool(int(parts[5]))
        distance, elapsed, time_to_goal = float(parts[6]), int(parts[7]), int(parts[8])
        obs = np.fromstring(parts[9], sep=",", dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise RuntimeError(f"Expected obs_dim={self.obs_dim}, got {obs.shape}")
        info = {
            "success": success,
            "death": death,
            "timeout": timeout,
            "distance": distance,
            "time_to_goal": float(time_to_goal) if success else np.nan,
            "elapsed": elapsed,
        }
        return obs, reward, done, info


def build_infinite_tux(root: Path, javac_cmd: str = "javac") -> Path:
    """Compile the patched Infinite Tux Java sources when class files are missing."""
    root = Path(root).resolve()
    src = root / "deb" / "src" / "main" / "java"
    classes = root / "build" / "classes"
    marker = classes / "com" / "mojang" / "mario" / "ai" / "HeadlessServer.class"
    if marker.exists():
        return classes
    if not src.exists():
        raise FileNotFoundError(f"Missing Infinite Tux source tree at {src}. Clone https://github.com/qbancoffee/infinite-tux there.")
    classes.mkdir(parents=True, exist_ok=True)
    sources = [str(p) for p in src.rglob("*.java")]
    cmd = [_java_tool(javac_cmd, "javac"), "-source", "1.8", "-target", "1.8", "-d", str(classes), *sources]
    try:
        subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("Infinite Tux backend requires a JDK with javac on PATH, or pass --javac-cmd via config.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to compile Infinite Tux headless bridge:\n{e.stderr}") from e
    return classes


def _java_tool(cmd: str, name: str) -> str:
    if cmd != name:
        return cmd
    for prefix in ("/opt/homebrew/opt/openjdk/bin", "/usr/local/opt/openjdk/bin"):
        path = Path(prefix) / name
        if path.exists():
            return str(path)
    return cmd
