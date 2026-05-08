from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .envs import make_env
from .policy import NeuralFeaturePolicy, clone_policy, load_policy, save_policy
from .utils import seed_all
from .warmstart_train import (
    _bounded_dataset,
    _collect_dagger,
    _collect_expert,
    _expert_solved_levels,
    _train,
    _train_shaped_rl,
    _write_target_checkpoints,
    evaluate_policy_on_levels,
)


def run_representation_pretrain(
    env_kwargs: dict,
    checkpoint_dir: str | Path,
    csv_dir: str | Path,
    seed: int,
    train_level_seed_start: int,
    eval_level_seed_start: int,
    train_levels: int,
    eval_levels: int,
    max_scan: int,
    iterations: int,
    rollouts_per_iter: int,
    epochs_per_iter: int,
    batch_size: int,
    lr: float,
    target_p0s: list[float],
    require_target_p0: float | None,
    planner_horizon: int,
    planner_beam: int,
    rollin_expert_start: float,
    rollin_expert_end: float,
    max_dataset: int,
    feature_dim: int,
    hidden_dim: int,
    rl_rollouts_per_iter: int,
    rl_updates_per_iter: int,
    rl_lr: float,
    rl_gamma: float,
    entropy_coef: float,
    head_epochs: int,
    head_eval_every: int,
    logit_scales: list[float],
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train neural phi, freeze it, then save measured-p0 linear-head warm starts."""
    seed_all(seed)
    checkpoint_dir, csv_dir = Path(checkpoint_dir), Path(csv_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = _expert_solved_levels(env_kwargs, train_level_seed_start, train_levels, max_scan, seed, planner_horizon, planner_beam, "train")
    eval_seeds = _expert_solved_levels(env_kwargs, eval_level_seed_start, eval_levels, max_scan, seed + 10_000, planner_horizon, planner_beam, "eval")
    pd.DataFrame(
        [{"split": "train", "level_seed": s} for s in train_seeds] + [{"split": "eval", "level_seed": s} for s in eval_seeds]
    ).to_csv(csv_dir / "representation_level_sets.csv", index=False)

    env = make_env(env_kwargs)
    policy = NeuralFeaturePolicy(env.obs_dim, feature_dim, hidden_dim).to(device)
    obs_rows, action_rows, hist = [], [], []
    rng = np.random.default_rng(seed)
    try:
        for it in tqdm(range(iterations + 1), desc="representation pretrain"):
            if it == 0:
                obs, actions = _collect_expert(env, env_kwargs, train_seeds, seed, planner_horizon, planner_beam)
            else:
                t = (it - 1) / max(1, iterations - 1)
                expert_prob = (1 - t) * rollin_expert_start + t * rollin_expert_end
                levels = rng.choice(train_seeds, size=min(rollouts_per_iter, len(train_seeds)), replace=False).tolist()
                obs, actions = _collect_dagger(env, env_kwargs, policy, levels, seed + 1000 * it, expert_prob, planner_horizon, planner_beam, device)
            obs_rows.append(obs)
            action_rows.append(actions)
            x, y = _bounded_dataset(obs_rows, action_rows, max_dataset, rng)
            _train(policy, x, y, epochs_per_iter, batch_size, lr, device)
            if rl_rollouts_per_iter and rl_updates_per_iter:
                levels = rng.choice(train_seeds, size=min(rl_rollouts_per_iter, len(train_seeds)), replace=False).tolist()
                _train_shaped_rl(env, policy, levels, seed + 20_000 + it, rl_updates_per_iter, rl_lr, rl_gamma, entropy_coef, device)
            metrics, _ = evaluate_policy_on_levels(env_kwargs, policy, eval_seeds, seed + 50_000 + it, deterministic=False, device=device)
            path = checkpoint_dir / f"representation_iter{it:03d}_p{metrics['success_rate']:.3f}.pt"
            save_policy(policy, path, {"iteration": it, "p0": metrics["success_rate"], "stage": "representation_pretrain"})
            hist.append({"iteration": it, "p0": metrics["success_rate"], "checkpoint_path": str(path), "train_pairs": len(x), "new_pairs": len(actions), **metrics})
            if require_target_p0 is not None and metrics["success_rate"] >= require_target_p0:
                break
    finally:
        close = getattr(env, "close", None)
        if close:
            close()

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(csv_dir / "representation_pretrain_eval.csv", index=False)
    head_df = _head_sweep(hist_df, env_kwargs, eval_seeds, obs_rows, action_rows, checkpoint_dir, csv_dir, seed, head_epochs, head_eval_every, logit_scales, batch_size, lr, max_dataset, rng, device)
    targets = _write_target_checkpoints(head_df, target_p0s, checkpoint_dir, csv_dir)
    if require_target_p0 is not None and targets["achieved_p0"].max() < require_target_p0:
        raise RuntimeError(f"Frozen-backbone head sweep reached max p0={targets['achieved_p0'].max():.3f}, below required {require_target_p0:.3f}.")
    return hist_df, head_df, targets


def _head_sweep(hist_df, env_kwargs, eval_seeds, obs_rows, action_rows, checkpoint_dir, csv_dir, seed, head_epochs, head_eval_every, logit_scales, batch_size, lr, max_dataset, rng, device):
    best = hist_df.sort_values("p0", ascending=False).iloc[0]
    base = load_policy(best["checkpoint_path"], device=device, freeze_backbone=True)
    x, y = _bounded_dataset(obs_rows, action_rows, max_dataset, rng)
    rows = []

    def record(policy, step, name):
        for scale in logit_scales or [1.0]:
            p = clone_policy(policy).to(device)
            with torch.no_grad():
                p.head.weight.mul_(float(scale))
            metrics, _ = evaluate_policy_on_levels(env_kwargs, p, eval_seeds, seed + 80_000 + 1000 * step + int(100 * float(scale)), deterministic=False, device=device)
            path = checkpoint_dir / f"warmstart_head_{name}_s{float(scale):g}_p{metrics['success_rate']:.3f}.pt"
            save_policy(p, path, {"stage": "frozen_backbone_head", "source_iteration": int(best["iteration"]), "head_step": step, "logit_scale": float(scale), "p0": metrics["success_rate"]})
            rows.append({"iteration": step, "p0": metrics["success_rate"], "logit_scale": float(scale), "checkpoint_path": str(path), **metrics})

    record(base, 0, "best")
    head = load_policy(best["checkpoint_path"], device=device, freeze_backbone=True)
    torch.nn.init.normal_(head.head.weight, mean=0.0, std=0.02)
    record(head, 1, "random")
    for epoch in tqdm(range(1, head_epochs + 1), desc="frozen-head sweep"):
        _train(head, x, y, 1, batch_size, lr, device)
        if epoch % max(1, head_eval_every) == 0:
            record(head, epoch + 1, f"epoch{epoch:03d}")

    out = pd.DataFrame(rows).drop_duplicates("checkpoint_path")
    out.to_csv(csv_dir / "warmstart_head_sweep.csv", index=False)
    return out
