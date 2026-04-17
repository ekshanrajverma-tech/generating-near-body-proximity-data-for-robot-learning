#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${ISAACLAB_DIR:-}" ]]; then
  echo "ERROR: Set ISAACLAB_DIR to your IsaacLab checkout path."
  echo "Example: ISAACLAB_DIR=\$HOME/IsaacLab ${BASH_SOURCE[0]}"
  exit 2
fi

ENV_DIR="$ISAACLAB_DIR/source/isaaclab_mimic/isaaclab_mimic/envs"
INIT_PY="$ENV_DIR/__init__.py"

if [[ ! -d "$ENV_DIR" ]]; then
  echo "ERROR: Could not find isaaclab_mimic envs directory:"
  echo "  $ENV_DIR"
  echo "Make sure ISAACLAB_DIR points to the IsaacLab repo root."
  exit 2
fi

cp -f "$REPO_DIR/franka_place_mimic_env.py" "$ENV_DIR/franka_place_mimic_env.py"
cp -f "$REPO_DIR/franka_place_mimic_env_cfg.py" "$ENV_DIR/franka_place_mimic_env_cfg.py"

TASK_ID="Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0"
if ! grep -q "$TASK_ID" "$INIT_PY"; then
  cat >> "$INIT_PY" <<'EOF'

##
# Franka: place cube into box (custom / research env_cfg)
##

gym.register(
    id="Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0",
    entry_point=f"{__name__}.franka_place_mimic_env:FrankaPlaceCubeIntoBoxMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_place_mimic_env_cfg:FrankaPlaceCubeIntoBoxMimicEnvCfg",
    },
    disable_env_checker=True,
)
EOF
fi

echo "Installed custom mimic task into: $ENV_DIR"
echo "Next: (re)install IsaacLab python packages:"
echo "  cd \"$ISAACLAB_DIR\" && ./isaaclab.sh -i"
