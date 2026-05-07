from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import DataLoader, TensorDataset

from .envs import make_env
from .policy import clone_policy, evaluate_policy, load_policy, rollout_policy, save_policy
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
) -> tuple[dict[str, np.ndarray], dict]:
    env = make_env(env_kwargs)
    obs_rows, action_rows, adv_rows = [], [], []
    skipped = 0
    rewards_all = []
    for i in range(prompts):
        trajs = [rollout_policy(env, policy, level_seed_start + i, seed + 10_000 * i + g, False, device) for g in range(G)]
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
        obs_dim = env.obs_dim
        batch = {"obs": np.empty((0, obs_dim), dtype=np.float32), "actions": np.empty(0, dtype=np.int64), "adv": np.empty(0, dtype=np.float32)}
    close = getattr(env, "close", None)
    if close:
        close()
    stats = {
        "skipped_groups": skipped,
        "total_groups": prompts,
        "skipped_group_fraction": skipped / max(prompts, 1),
        "batch_success_rate": float(np.mean(rewards_all)) if rewards_all else np.nan,
        "train_steps": int(len(batch["actions"])),
    }
    return batch, stats


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
) -> pd.DataFrame:
    seed_all(seed)
    policy = load_policy(checkpoint_path, device)
    rows = []
    last_stats = {"skipped_group_fraction": 0.0, "skipped_groups": 0, "total_groups": 0, "batch_success_rate": np.nan, "train_steps": 0}
    for it in range(iterations + 1):
        metrics, _ = evaluate_policy(env_kwargs, policy, eval_level_seed_start, eval_rollouts, seed + 1000 * it, False, device)
        ci_low, ci_high = binomial_ci(metrics["success_rate"], eval_rollouts)
        rows.append(
            {
                "M": M,
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
            env_kwargs, policy, beta, eps_smooth, G, prompts, train_level_seed_start + 1000 * it, seed + it, device
        )
        update_policy_grpo(policy, old_policy, batch, lr, update_epochs, beta_kl, device=device)
    tag = f"M{M}_eps{eps_smooth:g}_G{G}_beta{beta:g}_seed{seed}"
    save_policy(policy, Path(checkpoint_dir) / f"grpo_{tag}_final.pt", {"M": M, "eps_smooth": eps_smooth, "G": G, "beta": beta, "seed": seed})
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
) -> pd.DataFrame:
    frames = []
    for M in Ms:
        ckpt = Path(checkpoint_dir) / f"bc_M{M}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing {ckpt}; run behavior cloning first.")
        for eps in eps_smooths:
            for G in Gs:
                for beta in betas:
                    for s in range(seeds):
                        run_seed = base_seed + 10_000 * s
                        print(f"GRPO M={M} eps={eps:g} G={G} beta={beta:g} seed={run_seed}", flush=True)
                        frames.append(
                            run_grpo_setting(
                                env_kwargs,
                                ckpt,
                                M,
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
                            )
                        )
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(Path(csv_dir) / "grpo_eval.csv", index=False)
    return df
