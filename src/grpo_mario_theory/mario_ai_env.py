from __future__ import annotations

import atexit
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .infinite_tux_env import InfiniteTuxEnv, _java_tool


class MarioAIPrivateEnv(InfiniteTuxEnv):
    """Private wrapper for amidos2006/Mario-AI-Framework with original Mario art."""

    def __init__(
        self,
        height: int = 16,
        length: int = 150,
        difficulty: float = 0.35,
        max_steps: int = 400,
        obs_h: int = 9,
        obs_w: int = 13,
        mario_ai_dir: str = "external/mario-ai-framework",
        mario_ai_level_set: str = "notch",
        java_cmd: str = "java",
        javac_cmd: str = "javac",
        **_ignored,
    ):
        super().__init__(height, length, difficulty, max_steps, obs_h, obs_w, mario_ai_dir, java_cmd, javac_cmd)
        self.mario_ai_dir = Path(mario_ai_dir)
        self.mario_ai_level_set = mario_ai_level_set

    def reset(self, seed: int | None = None, level_seed: int | None = None, checkpoint=None) -> np.ndarray:
        if checkpoint is not None:
            raise NotImplementedError("MarioAIPrivateEnv does not support Python-side state snapshots.")
        self._ensure_proc()
        assert self.proc and self.proc.stdin
        level_seed = int(seed if level_seed is None else level_seed)
        diff = max(0, min(10, int(round(self.difficulty * 10))))
        self.proc.stdin.write(f"RESET {level_seed} {diff} {self.max_steps} {self.obs_h} {self.obs_w} {self.length} {self.mario_ai_level_set}\n")
        self.proc.stdin.flush()
        obs, _reward, _done, info = self._read_obs()
        self.last_obs, self.last_info = obs, info
        return obs

    def _ensure_proc(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        classes = build_mario_ai_private(self.mario_ai_dir, self.javac_cmd)
        cp = str(classes)
        self.proc = subprocess.Popen(
            [_java_tool(self.java_cmd, "java"), "-Djava.awt.headless=true", "-cp", cp, "grpo.HeadlessServer"],
            cwd=self.mario_ai_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)

    def expert_action(self) -> int:
        self._ensure_proc()
        assert self.proc and self.proc.stdin and self.proc.stdout
        self.proc.stdin.write("EXPERT\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"Mario AI private server exited without an expert action. stderr:\n{err}")
        if line.startswith("ERR"):
            raise RuntimeError(line.strip())
        parts = line.rstrip("\n").split("\t")
        if parts[0] != "ACT" or len(parts) != 2:
            raise RuntimeError(f"Malformed Mario AI private expert response: {line[:200]}")
        return int(parts[1])


def build_mario_ai_private(root: Path, javac_cmd: str = "javac") -> Path:
    root = Path(root).resolve()
    src = root / "src"
    classes = root / "build" / "classes"
    marker = classes / "grpo" / "HeadlessServer.class"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing Mario AI Framework source tree at {src}. Clone https://github.com/amidos2006/Mario-AI-Framework there for private use."
        )
    bridge_src = Path(__file__).resolve().parent / "java" / "mario_ai_private" / "HeadlessServer.java"
    target = src / "grpo" / "HeadlessServer.java"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_text(encoding="utf-8") != bridge_src.read_text(encoding="utf-8"):
        shutil.copyfile(bridge_src, target)
    if marker.exists() and marker.stat().st_mtime >= target.stat().st_mtime:
        return classes
    classes.mkdir(parents=True, exist_ok=True)
    sources = [str(p) for p in src.rglob("*.java")]
    cmd = [_java_tool(javac_cmd, "javac"), "-source", "1.8", "-target", "1.8", "-d", str(classes), *sources]
    try:
        subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("Mario AI private backend requires a JDK with javac on PATH.") from e
    except subprocess.CalledProcessError as e:
        if "Unable to locate a Java Runtime that supports javac" in e.stderr:
            raise RuntimeError("Mario AI private backend requires a JDK with javac; this machine only exposes a Java runtime.") from e
        raise RuntimeError(f"Failed to compile Mario AI private headless bridge:\n{e.stderr}") from e
    return classes
