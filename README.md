# Generating Near-Body Proximity Data for Robot Learning

This repo contains a custom Isaac Lab Mimic task that runs a Franka placing a cube into a box inside an iTHOR kitchen scene.

## What you get in this repo

- **Custom Mimic env + config**: `franka_place_mimic_env.py`, `franka_place_mimic_env_cfg.py`
- **Task/env config & scene assets**: `env_cfg.py`, `scene_no_middle_light.usda`, and vendored iTHOR kitchen assets under `molmo_assets/`
- **YCB assets**: `ycb_usd/` (repo-local; no `~/ycb_usd` dependency)

## Prereqs

- Linux + NVIDIA GPU drivers suitable for Isaac Sim / Isaac Lab
- A working **Isaac Lab** checkout (follow upstream install instructions for your machine)

This repo does **not** vendor Isaac Lab itself; you install Isaac Lab separately and then install/register this task into it.

## Setup (fresh clone)

### 1) Clone this repo

```bash
git clone <your-github-repo-url>
cd generating-near-body-proximity-data-for-robot-learning
```

### 2) Install Isaac Lab (separately)

Clone Isaac Lab somewhere (example uses `~/IsaacLab`), create/activate its conda env, and install its Python packages as per upstream docs.

At minimum, you should be able to run this from the Isaac Lab repo root:

```bash
./isaaclab.sh -i
```

### 3) Install this task into your Isaac Lab checkout

From *this* repo:

```bash
chmod +x scripts/install_into_isaaclab.sh
ISAACLAB_DIR="$HOME/IsaacLab" ./scripts/install_into_isaaclab.sh
```

Then re-install Isaac Lab Python packages (so your environment sees the updated Mimic env files):

```bash
cd "$HOME/IsaacLab"
./isaaclab.sh -i
```

### 4) Verify the Gym task is registered

In the Isaac Lab conda environment:

```bash
python -c "import gymnasium as gym; import isaaclab_mimic.envs; print(gym.spec('Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0'))"
```

## Running the task / generating data

### Generate Mimic dataset

Run Isaac Lab’s Mimic dataset generator with this task id:

```bash
python -m isaaclab_mimic.scripts.generate_dataset --task Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0 --help
```

Then run it with whatever arguments you normally use (headless / num envs / output path, etc.).

### Kitchen assets (important)

The iTHOR kitchen payload is **vendored into this repo** under:

- `molmo_assets/usd/scenes/ithor/FloorPlan1_physics/Payload/Contents.usda`

So a new user does **not** need `~/.molmospaces`, `ms-download`, or any symlinks for `FloorPlan1_physics`.

## Common gotchas

- If you see `gymnasium.error.NameNotFound` for the task id, you likely didn’t run `scripts/install_into_isaaclab.sh` (or didn’t re-run `./isaaclab.sh -i` afterward).
- If you move this repo to a different path, it’s okay: `franka_place_mimic_env_cfg.py` loads `env_cfg.py` **relative to this repo**, not from a hard-coded absolute path.

