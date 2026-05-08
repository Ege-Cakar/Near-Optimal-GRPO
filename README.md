# Binary Mirror-GRPO Amplification in a Libre Platformer

This repo contains a small, reproducible experimental suite for binary terminal-reward Mirror-GRPO / GRPO success-probability amplification.

The default platformer path uses `LibrePlatformer`, a self-contained generated tile-grid environment with simple geometric tiles only. The repo also includes an optional headless bridge to Infinite Tux under `external/infinite-tux`, the libre Infinite-Mario-style Java project whose assets were replaced with free/open assets. No Nintendo ROMs, Nintendo artwork, or original Super Mario Bros levels are used by either backend.

## Setup

```bash
uv sync
uv run python scripts/run_all.py --quick
uv run python scripts/run_all.py --medium
uv run python scripts/run_all.py --full
```

All scripts also accept:

```bash
--config configs/default.yaml --seed 0 --quick|--medium --outdir results
```

BC, GRPO, and `run_all.py` also accept `--device cpu|mps|cuda|auto`. CPU is the default because these runs are usually environment-rollout bound; MPS is supported when PyTorch reports it as available.

GRPO rollout and evaluation collection can use worker processes:

```bash
uv run python scripts/run_grpo_finetune.py --quick --num-workers 4
uv run python scripts/run_all.py --quick --num-workers 4
```

Workers each create their own environment instance. This helps most for slow Java/NES rollout backends; use CPU for worker inference.

Platformer scripts accept `--env-backend libre|infinite_tux|mario_ai_private|gym_super_mario_bros`. The Infinite Tux backend requires a JDK with `javac` on `PATH`; it compiles the patched headless bridge into `external/infinite-tux/build/classes` on first use. The `mario_ai_private` backend targets the research Mario AI Framework checkout in `external/mario-ai-framework`; it is for private/non-published runs only because that framework states that it uses original Mario art and includes original SMB level files. The `gym_super_mario_bros` backend is also private/non-published: it uses `gym-super-mario-bros==7.4.0`, `nes-py==8.2.1`, `gym==0.25.2`, and the `SuperMarioBrosRandomStages-v0` NES-ROM environment.

Install the optional Gym Mario backend with:

```bash
uv sync --extra gym-mario
```

`--quick` is intended as a laptop smoke run. `--medium` is the default local evidence run: it keeps all main ablation axes but cuts the largest Cartesian products. `--full` uses the larger sweeps in `configs/default.yaml` and is better suited to a cluster. Long loops use progress bars for finite-group trajectories, candidate evaluation, BC checkpoints, and GRPO settings/iterations.

## Experiments

### 1. Population Recurrence

The exact scalar population recurrence is

```text
p_n = sigmoid(E_n)
q_n = 1 - p_n
E_{n+1} = E_n + 1 / (beta * sqrt(p_n q_n + eps_smooth)).
```

For `eps_smooth = 0`, the denominator is singular as `q_n -> 0`:

```text
1 / sqrt(p_n q_n) ~ 1 / sqrt(q_n) ~ exp(E_n / 2).
```

This gives the idealized self-accelerating log-odds update. The hitting-time plots are meant to show behavior consistent with the loglog mechanism after warm start, not to prove the theorem from finite-precision numerics.

For fixed `eps_smooth > 0`, once `q_n << eps_smooth`, the increment is capped near

```text
1 / (beta * sqrt(eps_smooth)),
```

so the tail crosses over toward ordinary exponential decay in `q_n`.

Run:

```bash
uv run python scripts/run_population_recurrence.py --quick
```

Outputs:

- `results/csv/population_recurrence.csv`
- `results/csv/population_hitting_times.csv`
- `results/figures/population_q_vs_iter.{png,pdf}`
- `results/figures/population_hitting_loglog.{png,pdf}`
- `results/figures/population_hitting_log.{png,pdf}`

### 2. Finite-Group Stochastic Recurrence

At each iteration, the finite-group experiment draws

```text
K ~ Binomial(G, p_n)
p_hat = K / G
sigma_hat = sqrt(p_hat(1-p_hat) + eps_smooth).
```

When `eps_smooth = 0` and `K` is `0` or `G`, the group has no within-group reward contrast and the update is skipped. This demonstrates how finite group size creates degenerate groups, noise, and practical sampling floors that limit the ideal population recurrence.

Run:

```bash
uv run python scripts/run_finite_group_recurrence.py --quick
```

Outputs:

- `results/csv/finite_group_recurrence.csv`
- `results/csv/finite_group_summary.csv`
- `results/csv/finite_group_hitting_times.csv`
- `results/figures/finite_group_q_vs_iter.{png,pdf}`
- `results/figures/finite_group_degenerate_fraction.{png,pdf}`

### 3. Platformer Candidate Reweighting

`LibrePlatformer` is a generated side-scrolling tile-grid platformer with gravity, horizontal velocity, jumps, solid-tile collisions, static hazard tiles, a goal tile, death, timeout, ASCII rendering, and optional matplotlib frame export.

With `--env-backend infinite_tux`, the same candidate and learned-policy scripts run against the patched Infinite Tux headless bridge instead.

The finite candidate experiment is closest to the binary theory. It builds a bank of complete candidate action sequences, labels each candidate with terminal success `r_i in {0,1}`, and runs exact mirror reweighting:

```text
p_n = sum_i w_i r_i
sigma_n = sqrt(p_n(1-p_n) + eps_smooth)
w_{n+1,i} proportional to w_{n,i} exp(r_i / (beta sigma_n)).
```

Because the labels are binary, the candidate success odds follow the scalar recurrence exactly up to floating-point error as long as both success and failure mass remain represented.

Run:

```bash
uv run python scripts/run_candidate_platformer.py --quick
```

To use the Infinite Tux backend instead of `LibrePlatformer`:

```bash
uv run python scripts/run_candidate_platformer.py --quick --env-backend infinite_tux
uv run python scripts/run_bc_pretrain.py --quick --env-backend infinite_tux
uv run python scripts/run_grpo_finetune.py --quick --env-backend infinite_tux
```

For private runs against the Mario AI Framework with Mario assets:

```bash
uv run python scripts/run_candidate_platformer.py --quick --env-backend mario_ai_private
uv run python scripts/run_bc_pretrain.py --quick --env-backend mario_ai_private
uv run python scripts/run_grpo_finetune.py --quick --env-backend mario_ai_private
```

This backend expects `external/mario-ai-framework` to contain a local clone of `https://github.com/amidos2006/Mario-AI-Framework` and a JDK with `javac` available. The checkout is ignored by this repo’s `.gitignore` so it is not accidentally included in a publishable artifact.

For private runs against the Kautenja Gym/NES backend:

```bash
uv sync --extra gym-mario
uv run python -c "from grpo_mario_theory.gym_mario_env import GymSuperMarioBrosEnv; env=GymSuperMarioBrosEnv(max_steps=20); obs=env.reset(seed=0, level_seed=0); print(obs.shape); env.close()"
```

This wrapper exposes a fixed feature map from downsampled NES frames plus compact game-state scalars, so the policy remains the same linear softmax policy `phi(x)^T Omega`. The current built-in scripted expert is only a smoke-test warm-start source and does not reliably solve original SMB random stages. For paper-quality GRPO amplification on this backend, use a warm-start checkpoint with measured held-out `p0 > 0`; otherwise the expected result is no binary-reward amplification because there are no successes in support.

Outputs:

- `results/csv/candidate_bank_K*.csv`
- `results/csv/candidate_platformer_curves.csv`
- `results/csv/candidate_platformer_weights.csv`
- `results/figures/candidate_platformer_q_vs_iter.{png,pdf}`

### 4. Behavior-Cloning Warm Start

Warm-start policies are trained from a simple expert in the selected platformer backend. The learned policy is a linear softmax policy with logits `phi(x)^T Omega`, where `phi(x) = [x, 1]` is the flattened observation with a constant bias feature appended. The number of behavior-cloning samples `M` is not the theory variable. The relevant theory variable is the measured initial success probability `p0`, so the BC script evaluates every checkpoint on held-out generated levels and reports measured `p0(M)`.

Run:

```bash
uv run python scripts/run_bc_pretrain.py --quick
```

Outputs:

- `results/checkpoints/bc_M{M}.pt`
- `results/csv/bc_eval.csv`
- `results/csv/bc_losses.csv`
- `results/datasets/expert_*.npz`
- `results/figures/bc_warmstart_p0_vs_M.{png,pdf}`

### 5. Learned-Policy Binary Mirror-GRPO

The GRPO fine-tuning loop samples `G` rollouts for each level prompt, uses only terminal binary success rewards, computes group-normalized advantages, and skips degenerate `eps_smooth = 0` groups with all successes or all failures. It records held-out success rate, confidence intervals, distance, death rate, timeout rate, episode length, and skipped-group fraction after every iteration.

This experiment is intended to demonstrate the amplification mechanism when the warm-start policy already has successful trajectories in support. It should not be read as evidence that binary GRPO solves exploration from scratch. When measured `p0` is near zero, there is little or no success signal to amplify.

For the paper framing, the learned-policy experiments instantiate the tractable linear-policy regime: `phi` is fixed, only the last linear layer `Omega` is trained, and the relevant warm-start quality is measured by held-out `p0`, not by the number of pretraining samples or the backend name. The population and candidate experiments isolate the binary mirror-descent mechanism; the platformer experiments test whether the same mechanism appears when the linear policy interacts with sequential environments and finite sampled groups.

Run after BC checkpoints exist:

```bash
uv run python scripts/run_grpo_finetune.py --quick
```

Outputs:

- `results/checkpoints/grpo_*_final.pt`
- `results/csv/grpo_eval.csv`
- `results/figures/grpo_success_vs_iter_by_M.{png,pdf}`
- `results/figures/grpo_success_vs_iter_by_eps.{png,pdf}`
- `results/figures/grpo_group_size_effect.{png,pdf}`

## One-Command Runs

Small end-to-end run:

```bash
uv run python scripts/run_all.py --quick
```

Medium local run:

```bash
uv run python scripts/run_all.py --medium
```

Full configured run:

```bash
uv run python scripts/run_all.py --full
```

`run_all.py` also runs basic sanity checks:

- population recurrence has `p + q = 1`
- population `q_n` is monotonically nonincreasing
- candidate binary reweighting matches the scalar recurrence
- environment reset/step returns valid observations and info
- tiny behavior-cloning training reduces imitation loss

At the end it writes and prints `results/csv/final_summary.csv` with:

```text
experiment, setting, initial_p, final_p, initial_q, final_q,
hitting_time_tau_1e-2, hitting_time_tau_1e-4, skipped_group_fraction
```

## Plot Map to the Theory

- `population_q_vs_iter`: shows direct failure-probability decay under the exact recurrence.
- `population_hitting_loglog`: checks the unsmoothed accelerated regime against the loglog hitting-time scale.
- `population_hitting_log`: shows the fixed-smoothing tail becoming closer to log-scale hitting time.
- `finite_group_q_vs_iter`: shows how finite empirical groups depart from the population recurrence.
- `finite_group_degenerate_fraction`: shows when all-success or all-failure groups remove reward contrast.
- `candidate_platformer_q_vs_iter`: overlays exact candidate reweighting with the scalar recurrence.
- `bc_warmstart_p0_vs_M`: reports measured warm-start success probability rather than assuming `M` implies `p0`.
- `grpo_success_vs_iter_by_M`: compares amplification from different warm starts.
- `grpo_success_vs_iter_by_eps`: compares unsmoothed/tiny-smoothed behavior to fixed smoothing.
- `grpo_group_size_effect`: compares finite group sizes and skipped/noisy group effects.

The wording in generated figures and this README is intentionally cautious: the experiments are consistent with the population recurrence and demonstrate the amplification mechanism, while finite-group and smoothing effects limit the ideal singular acceleration.
