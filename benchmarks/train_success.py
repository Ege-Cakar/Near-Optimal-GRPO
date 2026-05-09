from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .envs import make_task
from .policy import CNNFeaturePolicy, clone_policy, load_policy, save_policy
from .utils import binomial_ci, save_dual, seed_all


def run_suite(cfg: dict, env_names: list[str], outdir: str | Path, quick: bool, seed: int, device: str) -> None:
    for env_name in env_names:
        paths = _paths(outdir, env_name)
        hist, targets = pretrain(cfg, env_name, paths, quick, seed, device)
        grpo = run_grpo(cfg, env_name, paths, quick, seed, device, targets)
        _plot(hist, grpo, paths["figures"])


def pretrain(cfg: dict, env_name: str, paths: dict[str, Path], quick: bool, seed: int, device: str):
    seed_all(seed)
    s, psec = _section(cfg, "pretrain", quick), cfg["policy"]
    train_levels = list(range(1000, 1000 + int(s["train_levels"])))
    eval_levels = list(range(20000, 20000 + int(s["eval_levels"])))
    pd.DataFrame([{"split": "train", "level_seed": x} for x in train_levels] + [{"split": "eval", "level_seed": x} for x in eval_levels]).to_csv(paths["csv"] / "level_sets.csv", index=False)

    task = make_task(env_name, cfg)
    policy = CNNFeaturePolicy(task.obs_shape, int(psec["feature_dim"]), int(psec["hidden_dim"]), task.n_actions, task.aux_dim).to(device)
    obs_rows, action_rows, hist = [], [], []
    rng = np.random.default_rng(seed)
    try:
        for it in tqdm(range(int(s["iterations"]) + 1), desc=f"{env_name} pretrain"):
            expert_prob = _linear(it, int(s["iterations"]), float(s["expert_prob_start"]), float(s["expert_prob_end"]))
            levels = rng.choice(train_levels, size=min(int(s["rollouts_per_iter"]), len(train_levels)), replace=False).tolist()
            obs, actions, trajs = collect_rollouts(task, policy, levels, seed + 1000 * it, expert_prob, device)
            if len(actions):
                obs_rows.append(obs)
                action_rows.append(actions)
                x, y = _bounded(obs_rows, action_rows, int(s["max_dataset"]), rng)
                train_bc(policy, x, y, int(s["bc_epochs"]), int(s["batch_size"]), float(s["bc_lr"]), device)
            train_pg(policy, trajs, int(s["rl_updates"]), float(s["rl_lr"]), float(s["gamma"]), float(s["entropy_coef"]), device)
            metrics = evaluate(env_name, cfg, policy, eval_levels, seed + 50_000 + it, device, deterministic=False)
            path = paths["checkpoints"] / f"pretrain_iter{it:03d}_p{metrics['p0']:.3f}.pt"
            save_policy(policy, path, {"env": env_name, "iteration": it, "p0": metrics["p0"], "stage": "pretrain"})
            hist.append({"iteration": it, "p0": metrics["p0"], "checkpoint_path": str(path), "train_pairs": sum(len(a) for a in action_rows), **metrics})
            if s.get("require_target_p0") is not None and metrics["p0"] >= float(s["require_target_p0"]):
                break
    finally:
        task.close()
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(paths["csv"] / "pretrain_eval.csv", index=False)
    targets = write_targets(hist_df, s["target_p0s"], paths["checkpoints"], paths["csv"])
    if s.get("require_target_p0") is not None and targets["achieved_p0"].max() < float(s["require_target_p0"]):
        raise RuntimeError(f"{env_name} reached max p0={targets['achieved_p0'].max():.3f}, below required {float(s['require_target_p0']):.3f}.")
    return hist_df, targets


def collect_rollouts(task, policy, levels, seed: int, expert_prob: float, device: str):
    rng = np.random.default_rng(seed)
    obs_rows, action_rows, trajs = [], [], []
    for i, level in enumerate(levels):
        obs = task.reset(seed + i, level)
        done = False
        tr = {"obs": [], "actions": [], "rewards": [], "success": 0.0}
        while not done:
            expert = task.expert_action()
            use_expert = expert is not None and rng.random() < expert_prob
            action = int(expert) if use_expert else policy.act(obs, deterministic=False, device=device)[0]
            if expert is not None:
                obs_rows.append(obs)
                action_rows.append(int(expert))
            tr["obs"].append(obs)
            tr["actions"].append(action)
            obs, reward, done, info = task.step(action)
            tr["rewards"].append(float(reward))
            tr["success"] = float(info["success"])
        tr["obs"] = np.asarray(tr["obs"], np.float32)
        tr["actions"] = np.asarray(tr["actions"], np.int64)
        trajs.append(tr)
    return np.asarray(obs_rows, np.float32), np.asarray(action_rows, np.int64), trajs


def train_bc(policy, obs, actions, epochs: int, batch_size: int, lr: float, device: str):
    if len(actions) == 0 or epochs == 0:
        return
    loader = DataLoader(TensorDataset(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long)), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=lr)
    policy.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.nn.functional.cross_entropy(policy(xb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()


def train_pg(policy, trajs, updates: int, lr: float, gamma: float, entropy_coef: float, device: str):
    if updates == 0 or not trajs:
        return
    obs = np.concatenate([t["obs"] for t in trajs])
    actions = np.concatenate([t["actions"] for t in trajs])
    returns = np.concatenate([_discounted(t["rewards"], gamma) for t in trajs]).astype(np.float32)
    if float(returns.max() - returns.min()) < 1e-8:
        return
    returns = (returns - returns.mean()) / float(returns.std())
    obs_t, act_t, ret_t = torch.as_tensor(obs, dtype=torch.float32, device=device), torch.as_tensor(actions, dtype=torch.long, device=device), torch.as_tensor(returns, dtype=torch.float32, device=device)
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=lr)
    policy.train()
    for _ in range(updates):
        dist = Categorical(logits=policy(obs_t))
        loss = -(ret_t * dist.log_prob(act_t)).mean() - entropy_coef * dist.entropy().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        opt.step()


def evaluate(env_name: str, cfg: dict, policy, levels, seed: int, device: str, deterministic: bool):
    task, rows = make_task(env_name, cfg), []
    try:
        for i, level in enumerate(levels):
            obs = task.reset(seed + i, int(level))
            done, actions = False, 0
            while not done:
                action = policy.act(obs, deterministic=deterministic, device=device)[0]
                obs, _reward, done, info = task.step(action)
                actions += 1
            rows.append({"level_seed": level, "success": float(info["success"]), "episode_length": actions, **info})
    finally:
        task.close()
    s = np.array([r["success"] for r in rows], dtype=float)
    return {"p0": float(s.mean()), "success_rate": float(s.mean()), "q_eval": float(1 - s.mean()), "avg_distance": float(np.mean([r["distance"] for r in rows])), "mean_episode_length": float(np.mean([r["episode_length"] for r in rows]))}


def write_targets(hist: pd.DataFrame, targets, checkpoint_dir: Path, csv_dir: Path):
    rows = []
    for target in targets:
        ok = hist[hist["p0"] >= float(target)]
        pick = ok.sort_values("p0").iloc[0] if len(ok) else hist.sort_values("p0").iloc[-1]
        dst = checkpoint_dir / f"warmstart_p{int(round(1000 * float(target))):04d}.pt"
        dst.write_bytes(Path(pick["checkpoint_path"]).read_bytes())
        rows.append({"label": dst.stem, "target_p0": float(target), "achieved_p0": float(pick["p0"]), "met": bool(pick["p0"] >= float(target)), "iteration": int(pick["iteration"]), "checkpoint_path": str(dst)})
    out = pd.DataFrame(rows)
    out.to_csv(csv_dir / "warmstart_targets.csv", index=False)
    return out


def run_grpo(cfg, env_name: str, paths: dict[str, Path], quick: bool, seed: int, device: str, targets: pd.DataFrame):
    s = _section(cfg, "grpo", quick)
    specs = targets[targets["met"]].to_dict("records")
    rows = []
    for spec in specs:
        for eps in s["eps_smooths"]:
            for G in s["Gs"]:
                for beta in s["betas"]:
                    for si in range(int(s["seeds"])):
                        rows.extend(grpo_setting(cfg, env_name, spec, float(eps), int(G), float(beta), seed + 10_000 * si, s, paths, device))
    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"] / "grpo_eval.csv", index=False)
    return df


def grpo_setting(cfg, env_name: str, spec: dict, eps: float, G: int, beta: float, seed: int, s: dict, paths: dict[str, Path], device: str):
    policy = load_policy(spec["checkpoint_path"], device=device, freeze_backbone=True)
    rows, last = [], {"skipped_group_fraction": 0.0, "batch_success_rate": np.nan, "train_steps": 0}
    eval_levels = list(range(20000, 20000 + int(s["eval_rollouts"])))
    for it in tqdm(range(int(s["iterations"]) + 1), desc=f"{env_name} GRPO {spec['label']} eps={eps:g} G={G}", leave=False):
        metrics = evaluate(env_name, cfg, policy, eval_levels, seed + 1000 * it, device, deterministic=False)
        lo, hi = binomial_ci(metrics["success_rate"], len(eval_levels))
        rows.append({"env": env_name, "warmstart": spec["label"], "warmstart_p0": spec["achieved_p0"], "eps_smooth": eps, "G": G, "beta": beta, "seed": seed, "iteration": it, "p": metrics["success_rate"], "q": metrics["q_eval"], "success_rate": metrics["success_rate"], "ci_low": lo, "ci_high": hi, **last})
        if it == int(s["iterations"]):
            break
        batch, last = collect_grpo_batch(cfg, env_name, policy, G, beta, eps, seed + it, int(s["prompts"]), device)
        update_grpo(policy, clone_policy(policy).to(device), batch, float(s["lr"]), int(s["update_epochs"]), float(s["beta_kl"]), device)
    save_policy(policy, paths["checkpoints"] / f"grpo_{spec['label']}_eps{eps:g}_G{G}_beta{beta:g}_seed{seed}.pt", {"env": env_name, **spec})
    return rows


def collect_grpo_batch(cfg, env_name: str, policy, G: int, beta: float, eps: float, seed: int, prompts: int, device: str):
    task = make_task(env_name, cfg)
    obs_rows, action_rows, adv_rows, rewards_all, skipped = [], [], [], [], 0
    try:
        for i in range(prompts):
            trajs = []
            for g in range(G):
                obs = task.reset(seed + 1000 * i + g, 30000 + i)
                done = False
                tr = {"obs": [], "actions": [], "success": 0.0}
                while not done:
                    action = policy.act(obs, deterministic=False, device=device)[0]
                    tr["obs"].append(obs)
                    tr["actions"].append(action)
                    obs, _reward, done, info = task.step(action)
                    tr["success"] = float(info["success"])
                trajs.append(tr)
            rewards = np.asarray([t["success"] for t in trajs], dtype=float)
            rewards_all.extend(rewards.tolist())
            p_hat = float(rewards.mean())
            sigma = float(np.sqrt(p_hat * (1 - p_hat) + eps))
            if (eps == 0.0 and p_hat in (0.0, 1.0)) or sigma == 0.0:
                skipped += 1
                continue
            for tr, adv in zip(trajs, (rewards - p_hat) / (beta * sigma)):
                obs_rows.append(np.asarray(tr["obs"], np.float32))
                action_rows.append(np.asarray(tr["actions"], np.int64))
                adv_rows.append(np.full(len(tr["actions"]), adv, dtype=np.float32))
    finally:
        task.close()
    if not obs_rows:
        task = make_task(env_name, cfg)
        obs_dim = task.obs_dim
        task.close()
        batch = {"obs": np.empty((0, obs_dim), np.float32), "actions": np.empty(0, np.int64), "adv": np.empty(0, np.float32)}
    else:
        batch = {"obs": np.concatenate(obs_rows), "actions": np.concatenate(action_rows), "adv": np.concatenate(adv_rows)}
    return batch, {"skipped_group_fraction": skipped / max(1, prompts), "batch_success_rate": float(np.mean(rewards_all)) if rewards_all else np.nan, "train_steps": int(len(batch["actions"]))}


def update_grpo(policy, old_policy, batch, lr: float, epochs: int, beta_kl: float, device: str):
    if len(batch["actions"]) == 0:
        return
    loader = DataLoader(TensorDataset(torch.as_tensor(batch["obs"], dtype=torch.float32), torch.as_tensor(batch["actions"], dtype=torch.long), torch.as_tensor(batch["adv"], dtype=torch.float32)), batch_size=1024, shuffle=True)
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        for obs, actions, adv in loader:
            obs, actions, adv = obs.to(device), actions.to(device), adv.to(device)
            dist, old = Categorical(logits=policy(obs)), Categorical(logits=old_policy(obs).detach())
            loss = -(adv * dist.log_prob(actions)).mean() + beta_kl * kl_divergence(dist, old).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()


def _paths(outdir: str | Path, env_name: str):
    root = Path(outdir) / env_name
    paths = {"root": root, "csv": root / "csv", "figures": root / "figures", "checkpoints": root / "checkpoints"}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _section(cfg: dict, name: str, quick: bool):
    out = {k: v for k, v in cfg[name].items() if k != "quick"}
    if quick:
        out.update(cfg[name].get("quick", {}))
    return out


def _bounded(obs_rows, action_rows, max_dataset: int, rng):
    obs, actions = np.concatenate(obs_rows), np.concatenate(action_rows)
    if len(actions) > max_dataset:
        idx = rng.choice(len(actions), size=max_dataset, replace=False)
        obs, actions = obs[idx], actions[idx]
    return obs.astype(np.float32), actions.astype(np.int64)


def _discounted(rewards, gamma: float):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = float(r) + gamma * g
        out.append(g)
    return list(reversed(out))


def _linear(it: int, total: int, start: float, end: float) -> float:
    t = it / max(1, total)
    return (1 - t) * start + t * end


def _plot(hist: pd.DataFrame, grpo: pd.DataFrame, fig_dir: Path):
    import matplotlib.pyplot as plt

    if not hist.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(hist["iteration"], hist["p0"], marker="o", ms=2.5)
        ax.set_xlabel("pretrain iteration")
        ax.set_ylabel("measured p0")
        ax.set_ylim(-0.02, 1.02)
        save_dual(fig, fig_dir / "pretrain_p0")
        plt.close(fig)
    if not grpo.empty:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for key, g in grpo.groupby("warmstart_p0"):
            s = g.groupby("iteration", as_index=False).agg(p=("p", "mean"))
            ax.plot(s["iteration"], s["p"], marker="o", ms=2.5, label=f"p0={key:g}")
        ax.set_xlabel("GRPO iteration")
        ax.set_ylabel("success rate")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        save_dual(fig, fig_dir / "grpo_success")
        plt.close(fig)
