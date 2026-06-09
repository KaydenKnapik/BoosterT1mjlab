# BoosterT1mjlab

This repository contains the training and evaluation code for the Booster T1 robot using MuJoCo and reinforcement learning. The project uses `uv` for fast, reproducible Python environment management.

## Setup

1. Make sure you have [uv](https://github.com/astral-sh/uv) installed on your system.
2. Clone this repository and navigate into it:
   ```bash
   git clone https://github.com/KaydenKnapik/BoosterT1mjlab.git
   cd BoosterT1mjlab
   ```
*(Note: `uv` will automatically create and manage the virtual environment for you the first time you run a command!)*

## Usage

### 1. List Available Environments
To see a list of all registered environments you can run:
```bash
uv run booster_t1_list_envs
```

### 2. Train the Model

**Standard PPO tasks** (velocity, tracking, flat kick):
```bash
uv run booster_t1_train Mjlab-Velocity-Flat-Booster-T1 --env.scene.num-envs 4096
```

**beyondAMP tasks** (AMP kick — uses motion reference files):
```bash
uv run booster_t1_train_beyondamp Mjlab-AmpKick-Booster-T1-21Dof --num-envs 4096
```

> **Note:** `booster_t1_train` will error if you accidentally point it at a beyondAMP task. If you see this, switch to `booster_t1_train_beyondamp`. Running the wrong trainer silently trains as plain PPO with no motion imitation, which produces bad results (arms flailing, unnatural motion).

Checkpoints and logs will automatically be saved to the `logs/` directory during training.

### 3. Play / Evaluate a Trained Model

**Standard PPO checkpoints:**
```bash
uv run booster_t1_play Mjlab-Velocity-Flat-Booster-T1 --checkpoint-file logs/rsl_rl/t1_velocity/YYYY-MM-DD_HH-MM-SS/model_XXX.pt
```

**beyondAMP checkpoints:**
```bash
uv run booster_t1_play_beyondamp Mjlab-AmpKick-Booster-T1-21Dof --checkpoint-file logs/rsl_rl/t1_amp_kick/YYYY-MM-DD_HH-MM-SS/model_XXX.pt
```
