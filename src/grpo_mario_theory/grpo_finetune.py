from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .envs import make_env
from .policy import LinearPolicy, clone_policy, evaluate_policy, load_policy, rollout_policy, save_policy
from .utils import binomial_ci, seed_all


def collect_grpo_batch(
    env_kwargs: dict,
    policy,
    beta: float,
    eps_smooth: float,
    G: int,
    prompts: int,
    level_seed_start: int,
    seed: int,
    device: str = "cpu",
    num_workers: int = 1,
) -> tuple[dict[str, np.ndarray], dict]:
    env = make_env(env_kwargs)
    obs_dim = env.obs_dim
    close = getattr(env, "close", None)
    if close:
        close()
    if num_workers > 1:
        state = _policy_state_cpu(policy)
        tasks = [(env_kwargs, obs_dim, state, level_seed_start + i, seed + 10_000 * i, G) for i in range(prompts)]
        with mp.get_context("spawn").Pool(processes=min(int(num_workers), len(tasks))) as pool:
            grouped_trajs = pool.map(_rollout_prompt_worker, tasks)
    else:
        env = make_env(env_kwargs)
        grouped_trajs = []
        for i in range(prompts):
            trajs = [rollout_policy(env, policy, level_seed_start + i, seed + 10_000 * i + g, False, device) for g in range(G)]
            grouped_trajs.append(trajs)
        close = getattr(env, "close", None)
        if close:
            close()
    obs_rows, action_rows, adv_rows = [], [], []
    skipped = 0
    rewards_all = []
    for trajs in grouped_trajs:
        rewards = np.array([t["success"] for t in trajs], dtype=float)
        rewards_all.extend(rewards.tolist())
        p_hat = float(rewards.mean())
        degenerate = eps_smooth == 0.0 and p_hat in (0.0, 1.0)
        sigma = float(np.sqrt(p_hat * (1.0 - p_hat) + eps_smooth))
        if degenerate or sigma == 0.0:
            skipped += 1
            continue
        adv = (rewards - p_hat) / (beta * sigma)
        for tr, a in zip(trajs, adv):
            obs_rows.append(tr["obs"])
            action_rows.append(tr["actions"])
            adv_rows.append(np.full(len(tr["actions"]), a, dtype=np.float32))
    if obs_rows:
        batch = {
            "obs": np.concatenate(obs_rows).astype(np.float32),
            "actions": np.concatenate(action_rows).astype(np.int64),
            "adv": np.concatenate(adv_rows).astype(np.float32),
        }
    else:
        batch = {"obs": np.empty((0, obs_dim), dtype=np.float32), "actions": np.empty(0, dtype=np.int64), "adv": np.empty(0, dtype=np.float32)}
    stats = {
        "skipped_groups": skipped,
        "total_groups": prompts,
        "skipped_group_fraction": skipped / max(prompts, 1),
        "batch_success_rate": float(np.mean(rewards_all)) if rewards_all else np.nan,
        "train_steps": int(len(batch["actions"])),
    }
    return batch, stats


def evaluate_policy_grpo(env_kwargs: dict, policy, level_seed_start: int, n_rollouts: int, seed: int, device: str, num_workers: int) -> dict:
    if num_workers <= 1:
        return evaluate_policy(env_kwargs, policy, level_seed_start, n_rollouts, seed, False, device)[0]
    state = _policy_state_cpu(policy)
    chunks = _chunks(n_rollouts, int(num_workers))
    tasks = [(env_kwargs, policy.obs_dim, state, level_seed_start, start, count, seed) for start, count in chunks]
    with mp.get_context("spawn").Pool(processes=min(int(num_workers), len(tasks))) as pool:
        rows = [r for chunk in pool.map(_eval_chunk_worker, tasks) for r in chunk]
    s = np.array([r["success"] for r in rows], dtype=float)
    return {
        "success_rate": float(s.mean()) if len(s) else np.nan,
        "q_eval": float(1.0 - s.mean()) if len(s) else np.nan,
        "avg_distance": float(np.mean([r["distance"] for r in rows])) if rows else np.nan,
        "death_rate": float(np.mean([r["death"] for r in rows])) if rows else np.nan,
        "timeout_rate": float(np.mean([r["timeout"] for r in rows])) if rows else np.nan,
        "mean_episode_length": float(np.mean([r["episode_length"] for r in rows])) if rows else np.nan,
    }


def _policy_state_cpu(policy) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def _make_worker_policy(obs_dim: int, state: dict[str, torch.Tensor]) -> LinearPolicy:
    policy = LinearPolicy(obs_dim)
    policy.load_state_dict(state)
    return policy.eval()


def _rollout_prompt_worker(args) -> list[dict]:
    env_kwargs, obs_dim, state, level_seed, seed, G = args
    env = make_env(env_kwargs)
    policy = _make_worker_policy(obs_dim, state)
    try:
        trajs = []
        for g in range(G):
            rollout_seed = seed + g
            seed_all(rollout_seed)
            trajs.append(rollout_policy(env, policy, level_seed, rollout_seed, False, "cpu"))
        return trajs
    finally:
        close = getattr(env, "close", None)
        if close:
            close()


def _eval_chunk_worker(args) -> list[dict]:
    env_kwargs, obs_dim, state, level_seed_start, start, count, seed = args
    env = make_env(env_kwargs)
    policy = _make_worker_policy(obs_dim, state)
    try:
        rows = []
        for i in range(start, start + count):
            rollout_seed = seed + 100_000 + i
            seed_all(rollout_seed)
            tr = rollout_policy(env, policy, level_seed_start + i, rollout_seed, False, "cpu")
            rows.append({k: tr[k] for k in ("success", "distance", "death", "timeout", "time_to_goal", "episode_length")})
        return rows
    finally:
        close = getattr(env, "close", None)
        if close:
            close()


def _chunks(n: int, workers: int) -> list[tuple[int, int]]:
    size = max(1, int(np.ceil(n / max(1, workers))))
    return [(start, min(size, n - start)) for start in range(0, n, size)]


def update_policy_grpo(
    policy,
    old_policy,
    batch: dict[str, np.ndarray],
    lr: float,
    update_epochs: int,
    beta_kl: float,
    batch_size: int = 1024,
    device: str = "cpu",
) -> list[float]:
    if len(batch["actions"]) == 0:
        return []
    ds = TensorDataset(
        torch.as_tensor(batch["obs"], dtype=torch.float32),
        torch.as_tensor(batch["actions"], dtype=torch.long),
        torch.as_tensor(batch["adv"], dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    old_policy.eval()
    losses = []
    for _ in range(update_epochs):
        for obs, actions, adv in loader:
            obs, actions, adv = obs.to(device), actions.to(device), adv.to(device)
            logits, old_logits = policy(obs), old_policy(obs).detach()
            dist, old_dist = Categorical(logits=logits), Categorical(logits=old_logits)
            loss = -(adv * dist.log_prob(actions)).mean() + beta_kl * kl_divergence(dist, old_dist).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
    return losses


def run_grpo_setting(
    env_kwargs: dict,
    checkpoint_path: str | Path,
    M: int,
    warmstart_label: str,
    warmstart_p0: float,
    eps_smooth: float,
    G: int,
    beta: float,
    seed: int,
    iterations: int,
    prompts: int,
    eval_rollouts: int,
    train_level_seed_start: int,
    eval_level_seed_start: int,
    checkpoint_dir: str | Path,
    lr: float,
    update_epochs: int,
    beta_kl: float,
    device: str = "cpu",
    num_workers: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    seed_all(seed)
    policy = load_policy(checkpoint_path, device)
    rows = []
    last_stats = {"skipped_group_fraction": 0.0, "skipped_groups": 0, "total_groups": 0, "batch_success_rate": np.nan, "train_steps": 0}
    desc = f"GRPO {warmstart_label} eps={eps_smooth:g} G={G} beta={beta:g} seed={seed}"
    for it in tqdm(range(iterations + 1), desc=desc, leave=False, disable=not show_progress):
        metrics = evaluate_policy_grpo(env_kwargs, policy, eval_level_seed_start, eval_rollouts, seed + 1000 * it, device, num_workers)
        ci_low, ci_high = binomial_ci(metrics["success_rate"], eval_rollouts)
        rows.append(
            {
                "M": M,
                "warmstart": warmstart_label,
                "warmstart_p0": warmstart_p0,
                "eps_smooth": eps_smooth,
                "G": G,
                "beta": beta,
                "seed": seed,
                "iteration": it,
                "success_rate": metrics["success_rate"],
                "p": metrics["success_rate"],
                "q_eval": metrics["q_eval"],
                "q": metrics["q_eval"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                **{k: metrics[k] for k in ("avg_distance", "death_rate", "timeout_rate", "mean_episode_length")},
                **last_stats,
            }
        )
        if it == iterations:
            break
        old_policy = clone_policy(policy).to(device)
        batch, last_stats = collect_grpo_batch(
            env_kwargs, policy, beta, eps_smooth, G, prompts, train_level_seed_start + 1000 * it, seed + it, device, num_workers
        )
        update_policy_grpo(policy, old_policy, batch, lr, update_epochs, beta_kl, device=device)
    tag = f"M{M}_eps{eps_smooth:g}_G{G}_beta{beta:g}_seed{seed}"
    save_policy(policy, Path(checkpoint_dir) / f"grpo_{tag}_final.pt", {"M": M, "warmstart": warmstart_label, "warmstart_p0": warmstart_p0, "eps_smooth": eps_smooth, "G": G, "beta": beta, "seed": seed})
    return pd.DataFrame(rows)


def run_grpo_suite(
    env_kwargs: dict,
    checkpoint_dir: str | Path,
    csv_dir: str | Path,
    Ms: list[int],
    eps_smooths: list[float],
    Gs: list[int],
    betas: list[float],
    seeds: int,
    iterations: int,
    prompts: int,
    eval_rollouts: int,
    train_level_seed_start: int,
    eval_level_seed_start: int,
    lr: float,
    update_epochs: int,
    beta_kl: float,
    base_seed: int,
    device: str = "cpu",
    num_workers: int = 1,
    show_progress: bool = True,
    checkpoint_specs: list[dict] | None = None,
) -> pd.DataFrame:
    frames = []
    specs = checkpoint_specs or [{"M": int(M), "label": f"M{M}", "achieved_p0": np.nan, "checkpoint_path": str(Path(checkpoint_dir) / f"bc_M{M}.pt")} for M in Ms]
    settings = [(spec, eps, G, beta, s) for spec in specs for eps in eps_smooths for G in Gs for beta in betas for s in range(seeds)]
    for spec, eps, G, beta, s in tqdm(settings, desc="GRPO settings", disable=not show_progress):
        ckpt = Path(spec["checkpoint_path"])
        M = int(spec.get("M", round(1000 * float(spec.get("target_p0", spec.get("achieved_p0", 0.0))))))
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing {ckpt}; run warm-start training or behavior cloning first.")
        run_seed = base_seed + 10_000 * s
        frames.append(
            run_grpo_setting(
                env_kwargs,
                ckpt,
                M,
                str(spec.get("label", f"M{M}")),
                float(spec.get("achieved_p0", np.nan)),
                float(eps),
                int(G),
                float(beta),
                run_seed,
                iterations,
                prompts,
                eval_rollouts,
                train_level_seed_start,
                eval_level_seed_start,
                checkpoint_dir,
                lr,
                update_epochs,
                beta_kl,
                device,
                num_workers,
                show_progress,
            )
        )
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(Path(csv_dir) / "grpo_eval.csv", index=False)
    return df
