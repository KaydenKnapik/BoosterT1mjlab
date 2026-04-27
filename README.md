# BoosterT1mjlab

This repository contains the training and evaluation code for the Booster T1 robot using MuJoCo and reinforcement learning. The project uses `uv` for fast, reproducible Python environment management.

## Setup

1. Make sure you have [uv](https://github.com/astral-sh/uv) installed on your system.
2. Clone this repository and navigate into it:
   ```bash
   git clone <your-repo-url>
   cd BoosterT1mjlab
   ```
*(Note: `uv` will automatically create and manage the virtual environment for you the first time you run a command!)*

## Usage

### 1. List Available Environments
To see a list of all registered environments you can run:
```bash
uv run booster_tlist_envs
```

### 2. Train the Model
To start training the Booster T1 in the flat velocity environment (using 4096 parallel environments for fast simulation), run:
```bash
uv run booster_ttrain Mjlab-Velocity-Flat-Booster-T1 --env.scene.num-envs 4096
```
Checkpoints and logs will automatically be saved to the `logs/` directory during training.

### 3. Play / Evaluate a Trained Model
To visualize and test a trained policy, use the play command and point it to your saved checkpoint file:
```bash
uv run booster_t1_play Mjlab-Velocity-Flat-Booster-T1 --checkpoint_file <PATH_TO_YOUR_CHECKPOINT_DIRECTORY>/<YOUR_MODEL_FILE>.pt
```
*Example path: `logs/rsl_rl/t1_velocity/YYYY-MM-DD_HH-MM-SS/model_XXX.pt`*
