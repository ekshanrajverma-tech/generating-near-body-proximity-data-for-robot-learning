import math
import os
import random
from typing import List, Tuple

import numpy as np
import torch

# Global seed: same YCB mesh order in every process (record_demos / annotate_demos).
random.seed(42)
np.random.seed(42)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.devices import Se3KeyboardCfg
from isaaclab.devices.device_base import DevicesCfg
import isaaclab.envs.mdp as base_mdp

# ---------------------------------------------------------------------------
# Kitchen — iTHOR FloorPlan
# ---------------------------------------------------------------------------
FLOOR_PLAN  = 1
KITCHEN_USD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scene_no_middle_light.usda"
)
# Wrapper over fr3_hand_converted.usd: disables bad PhysX spheres on link6_sensor_* (see sim log).
_FR3_SPAWN_USD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fr3_hand_converted_spawn.usda",
)

# Compared to ~/data_collection_isaaclab pick-and-place (SeattleLabTable at z=0, robot at env origin):
#   this kitchen uses a counter at ~1.1 m: robot base is elevated and cube/box sit on the counter, not the
#   same XY layout as the lab table. Teleop key mapping is still robot-base frame; reach, clutter (cabinets,
#   YCB props), and self-collision geometry differ — that is why it cannot feel identical even with the same
#   IK scale/sensitivity as env.py there.
#
# Surface z confirmed from iTHOR scene reference objects:
#   bread center z=1.178 (~6cm half-height)  → surface ≈ 1.118
#   apple center z=1.156 (~4cm half-height)  → surface ≈ 1.116
#   tomato center z=1.142 (~3.5cm half-height) → surface ≈ 1.107
# Use 1.10 for robot base (gravity-disabled, so it floats fine);
# add _SPAWN_Z_BUFFER to all dynamic objects to guarantee no sub-surface spawn.
COUNTER_Z       = 1.10   # kitchen counter surface z (robot base reference)
_SPAWN_Z_BUFFER = 0.012  # minimal clearance — lower = less "pop"; 4cm was causing objects to jump
# Isaac `box.usd` root vs visible counter top — kitchen mesh is above nominal COUNTER_Z; lift until base reads flush.
_BOX_SPAWN_EXTRA_Z = 0.055
# Robot / task layout:
# Face along the longer counter axis (+Y) to get more teleop travel before running out of table.
ROBOT_POS       = (-0.10, -0.48, COUNTER_Z)
_ROBOT_BASE_YAW = math.pi / 2.0
_h = 0.5 * _ROBOT_BASE_YAW
# Quaternion (w, x, y, z) in world frame — must be yaw about **Z**, not X:
# (cos, sin, 0, 0) rotates about X and tips the arm; (cos, 0, 0, sin) is π/2 yaw about Z.
ROBOT_ROT       = (float(math.cos(_h)), 0.0, 0.0, float(math.sin(_h)))
# Shoulder yaw — 0 matches pick-place reference. If the mesh still opens the wrong way on this USD, try ±2.74.
_FR3_JOINT1_TASK_YAW = 0.0
# Counter XY bounds — inset slightly from physical edges to keep objects fully on the surface.
COUNTER_X_RANGE = (-0.50, 0.40)
COUNTER_Y_RANGE = (-0.75, 0.60)

# Keep task props in front of the robot after the +90° base yaw.
# _CUBE_XY / _BOX_XY are set after kitchen-disk helpers (search for a guaranteed-valid pair).
_CUBE_SIZE = 0.09   # DexCube at 0.8x scale ≈ 0.088m half-diagonal
_BOX_SIZE  = 0.09

# Franka workspace envelope — objects must be within reach and in front of robot.
_ROBOT_XY = np.array([ROBOT_POS[0], ROBOT_POS[1]], dtype=np.float64)
FRANKA_MAX_REACH_XY = 0.62   # max XY dist from robot base for comfortable grasp
FRANKA_MIN_DIST_XY  = 0.28   # too close = arm can't fold; was 0.22 = too tight
FRANKA_MIN_FORWARD  = 0.26   # object must be at least this far in +Y from robot base

# Kitchen structural exclusion zones -- keep YCB / cube / box away from the sink, counter edges,
# and appliance areas. All counter clutter (bread, apple, bowl, tomato, knife, book, etc.) is now
# deactivated in scene_no_middle_light.usda, so only structural zones remain.
TASK_SPAWN_EXCLUDE_XY_DEFAULT: Tuple[Tuple[float, float, float], ...] = (
    (0.28, 0.38, 0.30),   # sink / basin area (+Y)
    (0.40, 0.22, 0.18),   # along-counter +X edge / appliance zone
    (-0.42, 0.35, 0.16),  # far-left counter edge
)

# Static iTHOR / Molmo counter props (env XY, meters) — hard disks for spawn + YCB parking logic.
# Apple / Book / bowl are ``active = false`` in ``scene_no_middle_light.usda`` (no exclude disks for them).
MOLMO_STATIC_EXCLUDE_XY: Tuple[Tuple[float, float, float], ...] = (
    (-0.10, -0.48, 0.30),  # Robot base working volume
)


def _parse_spawn_exclude_xy(
    raw: Tuple[Tuple[float, float, float], ...],
) -> List[Tuple[np.ndarray, float]]:
    return [(np.array([float(a), float(b)], dtype=np.float64), float(r)) for a, b, r in raw]


_MOLMO_SPAWN_EXCLUDE_PARSED: List[Tuple[np.ndarray, float]] = _parse_spawn_exclude_xy(MOLMO_STATIC_EXCLUDE_XY)


def _xy_footprint_clear_of_kitchen_disks(
    xy: np.ndarray,
    kitchen_disks: List[Tuple[np.ndarray, float]],
    object_footprint_radius: float,
    margin_m: float,
) -> bool:
    """True if a disk at xy with radius object_footprint_radius clears all kitchen exclusion disks."""
    for c, rad in kitchen_disks:
        if np.linalg.norm(xy - c) < rad + object_footprint_radius + margin_m:
            return False
    return True


# cube_in_box uses both |dx| and |dy| < this value — spawns need a margin on at least one axis.
_PLACE_SUCCESS_XY = 0.09


def _spawn_xy_clear_of_place_success(cxy: np.ndarray, bxy: np.ndarray, margin: float = 0.02) -> bool:
    """True if cube/box XY cannot immediately satisfy cube_in_box (per-axis clearance)."""
    dx = abs(float(cxy[0] - bxy[0]))
    dy = abs(float(cxy[1] - bxy[1]))
    need = _PLACE_SUCCESS_XY + margin
    return (dx >= need) or (dy >= need)


def _compute_reference_cube_box_xy() -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Pick default cube/box centers that satisfy the same rules as randomize_ycb_objects."""
    _robot_xy = np.array([ROBOT_POS[0], ROBOT_POS[1]], dtype=np.float64)
    disks = _parse_spawn_exclude_xy(TASK_SPAWN_EXCLUDE_XY_DEFAULT)
    km = 0.04
    min_cb, max_cb = 0.22, 0.40

    def _in_w(xy: np.ndarray) -> bool:
        d = float(np.linalg.norm(xy - _robot_xy))
        forward = xy[1] - _robot_xy[1]
        return (
            FRANKA_MIN_DIST_XY <= d <= FRANKA_MAX_REACH_XY
            and forward >= FRANKA_MIN_FORWARD
            and COUNTER_X_RANGE[0] + 0.06 <= xy[0] <= COUNTER_X_RANGE[1] - 0.06
            and COUNTER_Y_RANGE[0] + 0.06 <= xy[1] <= COUNTER_Y_RANGE[1] - 0.06
        )

    def _pair_ok(cxy: np.ndarray, bxy: np.ndarray) -> bool:
        d = float(np.linalg.norm(cxy - bxy))
        return (
            _in_w(cxy)
            and _in_w(bxy)
            and min_cb <= d <= max_cb
            and _spawn_xy_clear_of_place_success(cxy, bxy)
            and _xy_footprint_clear_of_kitchen_disks(cxy, disks, _CUBE_SIZE, km)
            and _xy_footprint_clear_of_kitchen_disks(bxy, disks, _BOX_SIZE, km)
        )

    for cx in np.linspace(-0.26, 0.14, 22):
        for cy in np.linspace(-0.14, 0.16, 22):
            for bxc in np.linspace(-0.22, 0.16, 22):
                for byc in np.linspace(-0.10, 0.24, 22):
                    cxy = np.array([float(cx), float(cy)])
                    bxy = np.array([float(bxc), float(byc)])
                    if _pair_ok(cxy, bxy):
                        return (float(cx), float(cy)), (float(bxc), float(byc))
    return (-0.14, -0.04), (0.06, 0.10)


_CUBE_XY, _BOX_XY = _compute_reference_cube_box_xy()


# Table camera rotation:
# Your screenshots show the TableCamera is not actually looking at the franka workspace.
# We compute a quaternion by a simple look-at:
# - local camera forward axis = +Z (confirmed from the original quaternion: it maps (0,0,1) to the desired forward).
# - local camera right axis  = +X
# - local camera up axis     = +Y
#
# The quaternion is computed so that the camera forward (+Z) points from the camera position to the
# workspace center (midpoint between cube and box).
# Camera placed in FRONT of franka (along +Y, the direction the robot faces) so it sees the robot face-on.
# The look-at math below auto-computes the rotation to aim at the workspace center.
_TABLE_CAM_POS = (ROBOT_POS[0], ROBOT_POS[1] + 1.8, 1.95)  # keep camera farther from robot
_WORKSPACE_CENTER = (
    0.5 * (_CUBE_XY[0] + _BOX_XY[0]),
    0.5 * (_CUBE_XY[1] + _BOX_XY[1]),
    COUNTER_Z + 0.10,
)

_fx = _WORKSPACE_CENTER[0] - _TABLE_CAM_POS[0]
_fy = _WORKSPACE_CENTER[1] - _TABLE_CAM_POS[1]
_fz = _WORKSPACE_CENTER[2] - _TABLE_CAM_POS[2]
_f_norm = math.sqrt(_fx * _fx + _fy * _fy + _fz * _fz) + 1e-12
_fx /= _f_norm
_fy /= _f_norm
_fz /= _f_norm

# world up
_ux, _uy, _uz = 0.0, 0.0, 1.0

# right = forward x world_up
_rx = _fy * _uz - _fz * _uy
_ry = _fz * _ux - _fx * _uz
_rz = _fx * _uy - _fy * _ux
_r_norm = math.sqrt(_rx * _rx + _ry * _ry + _rz * _rz) + 1e-12
_rx /= _r_norm
_ry /= _r_norm
_rz /= _r_norm

# up = forward x right (matches the original mapping for local +Y)
_sx = _fy * _rz - _fz * _ry
_sy = _fz * _rx - _fx * _rz
_sz = _fx * _ry - _fy * _rx

# Rotation matrix:
# Columns are world directions of local axes (right, up, forward).
# With local basis (X=right, Y=up, Z=forward), we get:
# R =
#   [ right.x   up.x     forward.x ]
#   [ right.y   up.y     forward.y ]
#   [ right.z   up.z     forward.z ]
_R00, _R01, _R02 = _rx, _sx, _fx
_R10, _R11, _R12 = _ry, _sy, _fy
_R20, _R21, _R22 = _rz, _sz, _fz

_trace = _R00 + _R11 + _R22
if _trace > 0.0:
    _S = math.sqrt(_trace + 1.0) * 2.0
    _qw = 0.25 * _S
    _qx = (_R21 - _R12) / _S
    _qy = (_R02 - _R20) / _S
    _qz = (_R10 - _R01) / _S
else:
    # Fallback cases: pick the largest diagonal element.
    if _R00 > _R11 and _R00 > _R22:
        _S = math.sqrt(1.0 + _R00 - _R11 - _R22) * 2.0
        _qw = (_R21 - _R12) / _S
        _qx = 0.25 * _S
        _qy = (_R01 + _R10) / _S
        _qz = (_R02 + _R20) / _S
    elif _R11 > _R22:
        _S = math.sqrt(1.0 + _R11 - _R00 - _R22) * 2.0
        _qw = (_R02 - _R20) / _S
        _qx = (_R01 + _R10) / _S
        _qy = 0.25 * _S
        _qz = (_R12 + _R21) / _S
    else:
        _S = math.sqrt(1.0 + _R22 - _R00 - _R11) * 2.0
        _qw = (_R10 - _R01) / _S
        _qx = (_R02 + _R20) / _S
        _qy = (_R12 + _R21) / _S
        _qz = 0.25 * _S

# normalize quaternion
_q_norm = math.sqrt(_qw * _qw + _qx * _qx + _qy * _qy + _qz * _qz) + 1e-12
TABLE_CAM_ROT = (
    float(_qw / _q_norm),
    float(_qx / _q_norm),
    float(_qy / _q_norm),
    float(_qz / _q_norm),
)

# ---------------------------------------------------------------------------
# YCB pool
# ---------------------------------------------------------------------------
_N = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"
# Local YCB USDs live inside this repo (pulled alongside code).
_L = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ycb_usd")

# Spawn Z:  For centered-origin USDs (Nucleus) pass h = half_height.
#           For bottom-origin USDs (local) pass h ≈ 0 (gravity settles them).
_Z = lambda h: COUNTER_Z + h + _SPAWN_Z_BUFFER
# Quaternions (w, x, y, z).
# Nucleus Axis_Aligned_Physics USDs are centered with tallest axis along Y (lying flat).
# Rx(+90°) rotates Y→Z so boxes/bottles/cans stand upright on the counter.
# Local ~/ycb_usd USDs already have Z-up with origin at the bottom → identity is correct.
_IDENTITY = [1.0, 0.0, 0.0, 0.0]
_STAND_UP = [0.7071068, 0.7071068, 0.0, 0.0]   # Rx(+90°): Y→Z

YCB_OBJECTS_CFG = {
    # --- Nucleus (centered origin, Y-tallest → need _STAND_UP) ---
    # _Z(half_height_Y) is correct because origin is at bbox center.
    "ycb_cracker_box":    {"usd": f"{_N}/003_cracker_box.usd",     "z": _Z(0.107), "rot": [0.7071068, -0.7071068, 0.0, 0.0], "size": 0.12},
    "ycb_sugar_box":      {"usd": f"{_N}/004_sugar_box.usd",       "z": _Z(0.090), "rot": [0.7071068, -0.7071068, 0.0, 0.0], "size": 0.10},
    "ycb_soup_can":       {"usd": f"{_N}/005_tomato_soup_can.usd", "z": _Z(0.052), "rot": [0.7071068, -0.7071068, 0.0, 0.0], "size": 0.07},
    "ycb_mustard_bottle": {"usd": f"{_N}/006_mustard_bottle.usd",  "z": _Z(0.100), "rot": [0.7071068, -0.7071068, 0.0, 0.0], "size": 0.08},
    # --- Local ~/ycb_usd (origin at bottom, already Z-up → _IDENTITY) ---
    # _Z(~0) because origin is at the bottom; gravity settles any buffer.
    "ycb_banana":         {"usd": f"{_L}/011_banana/011_banana.usd",                   "z": _Z(0.002), "rot": _IDENTITY, "size": 0.10},
    "ycb_bowl":           {"usd": f"{_L}/024_bowl/024_bowl.usd",                       "z": _Z(0.002), "rot": _IDENTITY, "size": 0.12},
    "ycb_mug":            {"usd": f"{_L}/025_mug/025_mug.usd",                         "z": _Z(0.002), "rot": _IDENTITY, "size": 0.09},
    "ycb_chef_can":       {"usd": f"{_L}/002_master_chef_can/002_master_chef_can.usd", "z": _Z(0.002), "rot": _IDENTITY, "size": 0.08},
    "ycb_bleach_cleanser":{"usd": f"{_L}/021_bleach_cleanser/021_bleach_cleanser.usd", "z": _Z(0.002), "rot": _IDENTITY, "size": 0.09},
    "ycb_canned_meat":    {"usd": f"{_L}/010_potted_meat_can/010_potted_meat_can.usd", "z": _Z(0.005), "rot": _IDENTITY, "size": 0.07},
    "ycb_tuna_can":       {"usd": f"{_L}/007_tuna_fish_can/007_tuna_fish_can.usd",    "z": _Z(0.005), "rot": _IDENTITY, "size": 0.07},
}
YCB_OBJECTS_CFG = {k: v for k, v in YCB_OBJECTS_CFG.items()
                   if v["usd"].startswith(str(_N)) or os.path.exists(v["usd"])}
YCB_KEYS = list(YCB_OBJECTS_CFG.keys())
print(f"[env_cfg] YCB pool ({len(YCB_KEYS)} objects): {YCB_KEYS}")

NUM_YCB_SLOTS = 10

# 15 candidates for 10 slots — stadium layout using far back, left/right wings, rear base.
# Nominal use: X in [-0.45, 0.38], Y in [-0.72, 0.52]; guard skips overlaps with task + robot.
YCB_FIXED_CANDIDATES: Tuple[Tuple[float, float], ...] = (
    # Zone 1: far back
    (-0.25, 0.45),
    (0.00, 0.42),
    (0.30, 0.35),
    # Zone 2: right wing
    (0.38, 0.10),
    (0.38, -0.15),
    (0.38, -0.45),
    # Zone 3: left wing
    (-0.45, -0.20),
    (-0.45, -0.04),
    (-0.45, 0.2),
    (-0.45, -0.50),
    # Zone 4: rear base
    (-0.35, -0.60),
    (0.10, -0.60),
    (-0.20, -0.55),
    (0.28, -0.60),
    # Zone 5: center-deep
    (0.15, 0.20),
)

_SLOT_NAMES = [f"ycb_slot{i}" for i in range(NUM_YCB_SLOTS)]
_SLOT_XY: List[Tuple[float, float]] = list(YCB_FIXED_CANDIDATES[:NUM_YCB_SLOTS])

_FALLBACK_YCB = list(YCB_OBJECTS_CFG.keys())[0] if YCB_OBJECTS_CFG else None
# Permanent mesh per slot (seeded random.sample above → identical across scripts).
_INIT = (
    random.sample(list(YCB_KEYS), NUM_YCB_SLOTS)
    if len(YCB_KEYS) >= NUM_YCB_SLOTS
    else list(YCB_KEYS)
)
while len(_INIT) < NUM_YCB_SLOTS:
    _INIT.append(_INIT[0] if _INIT else _FALLBACK_YCB)
print(
    f"[env_cfg] YCB slots={NUM_YCB_SLOTS} (random/np seed=42) → "
    f"ycb_slot0..{NUM_YCB_SLOTS - 1} = {_INIT}"
)

_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=1.0,
    disable_gravity=False,
)

# ---------------------------------------------------------------------------
# SPAD sensor layout — matches fr3_hand_converted.usd prim structure (29 total)
#   link2: sensor_0..6  (7)
#   link3: sensor_0..7  (8)
#   link5: sensor_0..5  (6)
#   link6: sensor_0..7  (8)
# Sensor prim path (default prim /fr3_hand is overlaid onto Robot/, so flat):
#   {ENV_REGEX_NS}/Robot/{link}_{sensor}/depth_camera
# ---------------------------------------------------------------------------
_SPAD_LAYOUT = {
    "link2": list(range(7)),   # 7  → total: 7
    "link3": list(range(8)),   # 8  → total: 15
    "link5": list(range(6)),   # 6  → total: 21
    "link6": list(range(8)),   # 8  → total: 29
}

_SPAD_CAM_PARAMS = dict(
    update_period=1 / 60,
    height=8,
    width=8,
    data_types=["distance_to_camera"],
    update_latest_camera_pose=True,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=10.86,
        focus_distance=2.0,
        horizontal_aperture=9.0,
        clipping_range=(0.05, 4.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="ros",
    ),
)


def _build_spad_cfgs() -> dict:
    """Build CameraCfg dict keyed as spad_fr3_{link}_sensor_{N}."""
    cfgs: dict = {}
    for link, sensor_ids in _SPAD_LAYOUT.items():
        for sid in sensor_ids:
            sensor_name = f"sensor_{sid}"
            attr_name   = f"spad_fr3_{link}_{sensor_name}"
            prim_path   = (
                "{{ENV_REGEX_NS}}/Robot/{link}_{sensor}/depth_camera"
            ).format(link=link, sensor=sensor_name)
            cfgs[attr_name] = CameraCfg(prim_path=prim_path, **_SPAD_CAM_PARAMS)
    return cfgs


_SPAD_CFGS = _build_spad_cfgs()
print(f"[env_cfg] SPAD sensor count verified: {len(_SPAD_CFGS)}")
print(f"[env_cfg] Built {len(_SPAD_CFGS)} SPAD sensor configs: "
      f"{list(_SPAD_CFGS.keys())[:5]} …")

# ---------------------------------------------------------------------------
# Randomization helpers
# ---------------------------------------------------------------------------
def randomize_ycb_objects(env, env_ids) -> None:
    # After cube/box: robot + task bubbles + inter-YCB spacing; jitter; _INIT fixes slot→mesh.
    device = env.device
    _robot_xy = np.array([ROBOT_POS[0], ROBOT_POS[1]], dtype=np.float64)

    _raw_excl = getattr(env.cfg, "task_spawn_exclude_xy", None)
    if _raw_excl is None:
        _raw_excl = TASK_SPAWN_EXCLUDE_XY_DEFAULT
    _kitchen_xy_excl: List[Tuple[np.ndarray, float]] = _parse_spawn_exclude_xy(_raw_excl)
    kitchen_margin = float(getattr(env.cfg, "task_spawn_kitchen_footprint_margin_m", 0.04))
    _molmo_margin = float(getattr(env.cfg, "molmo_spawn_exclude_margin_m", 0.0))

    # --- Workspace-aware placement ---
    _min_cube_box_xy = float(getattr(env.cfg, "cube_box_spawn_min_xy", 0.22))
    _max_cube_box_xy = float(getattr(env.cfg, "cube_box_spawn_max_xy", 0.40))

    def _in_workspace(xy):
        d = float(np.linalg.norm(xy - _robot_xy))
        forward = xy[1] - _robot_xy[1]
        return (
            FRANKA_MIN_DIST_XY <= d <= FRANKA_MAX_REACH_XY
            and forward >= FRANKA_MIN_FORWARD
            and COUNTER_X_RANGE[0] + 0.06 <= xy[0] <= COUNTER_X_RANGE[1] - 0.06
            and COUNTER_Y_RANGE[0] + 0.06 <= xy[1] <= COUNTER_Y_RANGE[1] - 0.06
        )

    def _valid_pair(cxy, bxy):
        d = float(np.linalg.norm(cxy - bxy))
        return (
            _in_workspace(cxy)
            and _in_workspace(bxy)
            and _min_cube_box_xy <= d <= _max_cube_box_xy
            and _spawn_xy_clear_of_place_success(cxy, bxy)
            and _xy_footprint_clear_of_kitchen_disks(cxy, _kitchen_xy_excl, _CUBE_SIZE, kitchen_margin)
            and _xy_footprint_clear_of_kitchen_disks(bxy, _kitchen_xy_excl, _BOX_SIZE, kitchen_margin)
            and _xy_footprint_clear_of_kitchen_disks(cxy, _MOLMO_SPAWN_EXCLUDE_PARSED, _CUBE_SIZE, _molmo_margin)
            and _xy_footprint_clear_of_kitchen_disks(bxy, _MOLMO_SPAWN_EXCLUDE_PARSED, _BOX_SIZE, _molmo_margin)
        )

    _CUBE_RAND_X = (-0.24, 0.12)
    _CUBE_RAND_Y = (-0.12, 0.14)
    _BOX_RAND_X  = (-0.20, 0.16)
    _BOX_RAND_Y  = (-0.06, 0.24)

    cube_x = cube_y = bx = by = None
    for _ in range(800):
        cx_try = random.uniform(*_CUBE_RAND_X)
        cy_try = random.uniform(*_CUBE_RAND_Y)
        bx_try = random.uniform(*_BOX_RAND_X)
        by_try = random.uniform(*_BOX_RAND_Y)
        cxy = np.array([cx_try, cy_try])
        bxy = np.array([bx_try, by_try])
        if _valid_pair(cxy, bxy):
            cube_x, cube_y = cx_try, cy_try
            bx, by = bx_try, by_try
            break
    if cube_x is None:
        for cx in np.linspace(_CUBE_RAND_X[0], _CUBE_RAND_X[1], 21):
            for cy in np.linspace(_CUBE_RAND_Y[0], _CUBE_RAND_Y[1], 21):
                for bxc in np.linspace(_BOX_RAND_X[0], _BOX_RAND_X[1], 15):
                    for byc in np.linspace(_BOX_RAND_Y[0], _BOX_RAND_Y[1], 15):
                        cxy = np.array([float(cx), float(cy)])
                        bxy = np.array([float(bxc), float(byc)])
                        if _valid_pair(cxy, bxy):
                            cube_x, cube_y = float(cx), float(cy)
                            bx, by = float(bxc), float(byc)
                            break
                    if cube_x is not None:
                        break
                if cube_x is not None:
                    break
            if cube_x is not None:
                break
    if cube_x is None:
        cube_x, cube_y = _CUBE_XY[0], _CUBE_XY[1]
        bx, by = _BOX_XY[0], _BOX_XY[1]
    # If defaults fail validation (tuned constants / kitchen layout), scan for any legal pair.
    if not _valid_pair(np.array([cube_x, cube_y]), np.array([bx, by])):
        found = False
        for cx in np.linspace(-0.26, 0.14, 22):
            for cy in np.linspace(-0.14, 0.16, 22):
                for bxc in np.linspace(-0.22, 0.16, 22):
                    for byc in np.linspace(-0.10, 0.24, 22):
                        cxy = np.array([float(cx), float(cy)])
                        bxy = np.array([float(bxc), float(byc)])
                        if _valid_pair(cxy, bxy):
                            cube_x, cube_y = float(cx), float(cy)
                            bx, by = float(bxc), float(byc)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break

    cxy_final = np.array([cube_x, cube_y])
    bxy_final = np.array([bx, by])
    # Hint for debugging / future terms — do not tie a termination to this (false negatives loop-reset).
    env._placement_valid = bool(_valid_pair(cxy_final, bxy_final))

    cube = env.scene["cube"]
    box  = env.scene["box"]
    _zero_vel = torch.zeros((1, 6), device=device, dtype=torch.float32)

    cube_z = COUNTER_Z + 0.044 + _SPAWN_Z_BUFFER
    cube_pos_w = torch.tensor([[cube_x, cube_y, cube_z]], device=device, dtype=torch.float32)
    cube_pos_w += env.scene.env_origins[env_ids]
    cube_quat   = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    cube.write_root_pose_to_sim(torch.cat([cube_pos_w, cube_quat], dim=-1), env_ids=env_ids)
    cube.write_root_velocity_to_sim(_zero_vel, env_ids=env_ids)

    box_z = COUNTER_Z + 0.044 + _SPAWN_Z_BUFFER + _BOX_SPAWN_EXTRA_Z
    box_pos_w = torch.tensor([[bx, by, box_z]], device=device, dtype=torch.float32)
    box_pos_w += env.scene.env_origins[env_ids]
    box_quat   = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    box.write_root_pose_to_sim(torch.cat([box_pos_w, box_quat], dim=-1), env_ids=env_ids)
    box.write_root_velocity_to_sim(_zero_vel, env_ids=env_ids)

    box_xy = np.array([bx, by], dtype=np.float64)
    cube_xy = np.array([cube_x, cube_y], dtype=np.float64)
    MIN_DIST_FROM_ROBOT = 0.35
    MIN_DIST_FROM_TASK = 0.28
    MIN_INTER_OBJECT = 0.22

    available_spots: List[Tuple[float, float]] = random.sample(
        list(YCB_FIXED_CANDIDATES), len(YCB_FIXED_CANDIDATES)
    )
    placed_positions: List[np.ndarray] = []

    for i in range(NUM_YCB_SLOTS):
        slot_name = _SLOT_NAMES[i]
        obj = env.scene[slot_name]
        info = YCB_OBJECTS_CFG[_INIT[i]]
        found_spot = False
        x = 0.0
        y = 0.0
        for si, spot in enumerate(available_spots):
            sx, sy = float(spot[0]), float(spot[1])
            p = np.array([sx, sy], dtype=np.float64)
            d_robot = float(np.linalg.norm(p - _robot_xy))
            d_box = float(np.linalg.norm(p - box_xy))
            d_cube = float(np.linalg.norm(p - cube_xy))
            if d_robot > MIN_DIST_FROM_ROBOT and d_box > MIN_DIST_FROM_TASK and d_cube > MIN_DIST_FROM_TASK:
                if all(float(np.linalg.norm(p - prev)) > MIN_INTER_OBJECT for prev in placed_positions):
                    x, y = sx, sy
                    placed_positions.append(p.copy())
                    available_spots.pop(si)
                    found_spot = True
                    break
        if not found_spot:
            x = -0.45
            y = min(0.40 + (i * 0.02), float(COUNTER_Y_RANGE[1]) - 0.04)
            placed_positions.append(np.array([x, y], dtype=np.float64))

        x += random.uniform(-0.01, 0.01)
        y += random.uniform(-0.01, 0.01)

        z = info["z"]
        q = list(info["rot"])
        new_pos = torch.tensor([[x, y, z]], device=device, dtype=torch.float32) + env.scene.env_origins[
            env_ids
        ]
        quat = torch.tensor([q], device=device, dtype=torch.float32)
        root_pose = torch.cat([new_pos, quat], dim=-1)
        obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(_zero_vel, env_ids=env_ids)


# Tactile skins: link*_skin/visuals in fr3_hand_converted.usd (meshes may be instanced).
_FR3_SKIN_GREEN_MAT_PATH = "/World/Looks/fr3_prox_learn_skin_green"
_FR3_SKIN_LINKS = ("link2_skin", "link3_skin", "link5_skin", "link6_skin")
_FR3_SKIN_DIFFUSE = (0.18, 0.68, 0.24)


def apply_fr3_skin_green_visual(env, env_ids: torch.Tensor | slice | None = None) -> None:
    """Tint FR3 tactile skins green without touching the rest of the arm.

    Binds a PreviewSurface at ``_FR3_SKIN_GREEN_MAT_PATH`` using Omniverse
    ``BindMaterialCommand`` (Isaac Lab's ``bind_visual_material`` skips instanced prims).
    Targets ``…/link*_skin/visuals`` and ``…/link*_skin``, then any Mesh under ``Robot`` whose
    path contains ``_skin``.
    """
    from pxr import Gf, Usd, UsdGeom, UsdShade

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    if not stage.GetPrimAtPath(_FR3_SKIN_GREEN_MAT_PATH).IsValid():
        mcfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=_FR3_SKIN_DIFFUSE,
            metallic=0.05,
            roughness=0.55,
        )
        mcfg.func(_FR3_SKIN_GREEN_MAT_PATH, mcfg)

    mat_prim = stage.GetPrimAtPath(_FR3_SKIN_GREEN_MAT_PATH)
    if not mat_prim.IsValid():
        return
    material = UsdShade.Material(mat_prim)

    if env_ids is None or env_ids is slice(None):
        ids = list(range(env.num_envs))
    elif isinstance(env_ids, torch.Tensor):
        ids = [int(i) for i in env_ids.reshape(-1).cpu().tolist()]
    else:
        ids = [int(i) for i in env_ids]

    def _bind(prim_path: str) -> None:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        try:
            import omni.kit.commands

            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=str(prim_path),
                material_path=_FR3_SKIN_GREEN_MAT_PATH,
                strength="strongerThanDescendants",
                stage=stage,
            )
        except Exception:
            try:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants
                )
            except Exception:
                try:
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                except Exception:
                    pass

    def _paint_gprim(p) -> None:
        if not p.IsA(UsdGeom.Gprim):
            return
        g = UsdGeom.Gprim(p)
        c = Gf.Vec3f(float(_FR3_SKIN_DIFFUSE[0]), float(_FR3_SKIN_DIFFUSE[1]), float(_FR3_SKIN_DIFFUSE[2]))
        dc = g.GetDisplayColorAttr()
        if dc:
            dc.Set([c])
        else:
            g.CreateDisplayColorAttr([c])

    for eid in ids:
        base = env.scene.env_prim_paths[eid]
        robot_path = f"{base}/Robot"
        robot = stage.GetPrimAtPath(robot_path)
        if not robot.IsValid():
            continue
        for sl in _FR3_SKIN_LINKS:
            _bind(f"{robot_path}/{sl}/visuals")
            _bind(f"{robot_path}/{sl}")
        for p in Usd.PrimRange(robot):
            ps = str(p.GetPath())
            if "_skin" not in ps:
                continue
            if p.IsA(UsdGeom.Mesh):
                try:
                    if p.IsInstanceable():
                        p.SetInstanceable(False)
                except Exception:
                    pass
                _bind(ps)
            _paint_gprim(p)


# ---------------------------------------------------------------------------
# Grasp / place subtask signals  (used by record_demos + mimic annotator)
# ---------------------------------------------------------------------------
def _cube_inside_box(cube_pos, box_pos) -> torch.Tensor:
    dx     = torch.abs(cube_pos[:, 0] - box_pos[:, 0])
    dy     = torch.abs(cube_pos[:, 1] - box_pos[:, 1])
    z_diff = cube_pos[:, 2] - box_pos[:, 2]
    return (dx < _PLACE_SUCCESS_XY) & (dy < _PLACE_SUCCESS_XY) & (z_diff > -0.12) & (z_diff < 0.15)


def grasp_signal(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    gripper_threshold: float = 0.05,
    proximity_threshold: float = 0.10,
) -> torch.Tensor:
    """1 when gripper is closed, near the cube, and cube is at least lightly lifted.

    Keep this less strict than task success:
    - Too-tight proximity/lift thresholds delay grasp term annotation and can make
      generated trajectories skip the "near-pick" phase or fail to enter place.
    """
    from isaaclab.assets import Articulation, RigidObject
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject   = env.scene[object_cfg.name]
    finger_indices, _   = robot.find_joints("fr3_finger_joint.*")
    finger_pos          = robot.data.joint_pos[:, finger_indices]
    gripper_width       = finger_pos.sum(dim=1)
    ee_frame  = env.scene["ee_frame"]
    eef_pos   = ee_frame.data.target_pos_w[..., 0, :]
    cube_pos  = cube.data.root_pos_w[:, :3]
    dist      = torch.norm(eef_pos - cube_pos, dim=1)
    # 3 cm lift above counter is enough to indicate a successful pickup event.
    lifted    = cube.data.root_pos_w[:, 2] > (COUNTER_Z + 0.03)
    return ((gripper_width < gripper_threshold) & (dist < proximity_threshold) & lifted).float()


def place_signal(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """1 when cube is inside the box."""
    cube_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3]
    box_pos  = env.scene[box_cfg.name].data.root_pos_w[:, :3]
    return _cube_inside_box(cube_pos, box_pos).float()


def cube_in_box(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    cube_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3]
    box_pos  = env.scene[box_cfg.name].data.root_pos_w[:, :3]
    return _cube_inside_box(cube_pos, box_pos)


# ---------------------------------------------------------------------------
# Robot — FR3 with skins + hand gripper (_FR3_SPAWN_USD → fr3_hand + link6 collision patch)
# ---------------------------------------------------------------------------
FR3_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FR3_SPAWN_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            # Lower values reduce “popping” when contacts penetrate.
            max_depenetration_velocity=1.5,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Let upper/lower arm links reject each other in PhysX (was False → arm could "fold through" itself).
            # If the FR3 USD causes jitter or explosion at reset, set back to False.
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS,
        rot=ROBOT_ROT,
        joint_pos={
            "fr3_joint1":  _FR3_JOINT1_TASK_YAW,
            "fr3_joint2": -0.569,
            "fr3_joint3":  0.0,
            "fr3_joint4": -2.810,
            # Wrist: fingers −Z; jaw opening along world +X (was +Y with j7=+π/2). Re-tune j6/j7 if FK quat drifts.
            "fr3_joint5":  0.0,
            "fr3_joint6":  2.241,
            "fr3_joint7":  0.0,
            "fr3_finger_joint1": 0.04,
            "fr3_finger_joint2": 0.04,
        },
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=260.0,
            damping=55.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=260.0,
            damping=55.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    # <1.0 keeps the IK target away from hard stops so the arm doesn’t “snap” flat at limits.
    soft_joint_pos_limit_factor=0.96,
)

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class YCBSceneCfg(InteractiveSceneCfg):

    robot = FR3_CFG

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )

    kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path=KITCHEN_USD),
    )

    # ---- Pick-and-place objects ----------------------------------------
    # DexCube at 0.8x scale: half-height ≈ 0.044m; +_SPAWN_Z_BUFFER avoids surface clipping.
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[_CUBE_XY[0], _CUBE_XY[1], COUNTER_Z + 0.044 + _SPAWN_Z_BUFFER],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=_RIGID_PROPS,
        ),
    )

    # box.usd at 0.8x scale: half-height ≈ 0.044m; +_BOX_SPAWN_EXTRA_Z aligns visible bottom with counter.
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[_BOX_XY[0], _BOX_XY[1], COUNTER_Z + 0.044 + _SPAWN_Z_BUFFER + _BOX_SPAWN_EXTRA_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Box/box.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=_RIGID_PROPS,
        ),
    )

    # ---- EE frame transformer -------------------------------------------
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.107]),
            )
        ],
    )

    # ---- Cameras --------------------------------------------------------
    # Re-centered to new long-side-facing layout so teleop workspace stays in-frame.
    # Rotation is module-level (`TABLE_CAM_ROT`) to avoid defining tuples inside this configclass.

    table_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TableCamera",
        update_period=0.0,
        height=256, width=256,
        data_types=["rgb", "distance_to_camera"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.1, 6.0)),
        offset=CameraCfg.OffsetCfg(pos=_TABLE_CAM_POS,
                                    rot=TABLE_CAM_ROT, convention="ros"),
    )


    # Top-down camera — visualization only, NOT recorded in dataset (not in PolicyCfg).
    # High Z (near scene ceiling) so the arm/gripper never intersects the camera frustum prim; wide clip + aperture.
    # Open in Isaac Sim: Create New Viewport → Camera → /World/envs/env_0/TopDownCamera
##    top_down_camera = CameraCfg(
##        prim_path="{ENV_REGEX_NS}/TopDownCamera",
##        update_period=0.1,
##        height=720, width=1280,
#        data_types=["rgb"],
#        spawn=sim_utils.PinholeCameraCfg(
#            focal_length=24.0,
#            focus_distance=400.0,
#            horizontal_aperture=56.0,
#            clipping_range=(0.15, 10.0),
#        ),
#        offset=CameraCfg.OffsetCfg(
#            pos=(
#                ROBOT_POS[0] + 0.12,
#                ROBOT_POS[1] + 0.04,
#                2.6,
#            ),
#            rot=(0.0, -1.0, 0.0, 0.0),
#            convention="ros",
#        ),
#    )
#
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_hand/wrist_cam",
        update_period=0.0,
        height=128, width=128,
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                          horizontal_aperture=20.955, clipping_range=(0.1, 2.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.13, 0.0, -0.15),
                                    rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"),
    )

    def __post_init__(self):
        super().__post_init__()
        # Attach all 37 SPAD sensors dynamically — avoids any key-name mismatch.
        for attr_name, cfg in _SPAD_CFGS.items():
            setattr(self, attr_name, cfg)
        for i in range(NUM_YCB_SLOTS):
            key = _INIT[i]
            info = YCB_OBJECTS_CFG[key]
            xy = _SLOT_XY[i]
            setattr(
                self,
                f"ycb_slot{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/YcbSlot{i}",
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=[xy[0], xy[1], info["z"]],
                        rot=list(info["rot"]),
                    ),
                    spawn=UsdFileCfg(usd_path=info["usd"], rigid_props=_RIGID_PROPS),
                ),
            )


# ---------------------------------------------------------------------------
# Helper: EE quaternion from FrameTransformer (world frame, wxyz)
# ---------------------------------------------------------------------------
def _eef_pos(
    env,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    from isaaclab.sensors import FrameTransformer
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins


def _eef_quat(
    env,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    from isaaclab.sensors import FrameTransformer
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


def _sim_timestamp(env) -> torch.Tensor:
    """Simulation time in seconds: step_index × step_dt. Shape (N_envs, 1)."""
    step = env._sim_step_counter // env.cfg.decimation
    return torch.full((env.num_envs, 1), step * env.step_dt,
                      dtype=torch.float32, device=env.device)


def _spad_sensor_pos_w(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame XYZ position of a SPAD/camera sensor. Shape (N_envs, 3)."""
    from isaaclab.sensors import Camera
    sensor: Camera = env.scene[sensor_cfg.name]
    return sensor.data.pos_w


def _spad_sensor_quat_w(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame quaternion (wxyz) of a SPAD/camera sensor. Shape (N_envs, 4)."""
    from isaaclab.sensors import Camera
    sensor: Camera = env.scene[sensor_cfg.name]
    return sensor.data.quat_w_world


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        timestamp  = ObsTerm(func=_sim_timestamp)
        joint_pos  = ObsTerm(func=base_mdp.joint_pos)
        joint_vel  = ObsTerm(func=base_mdp.joint_vel)
        eef_pos    = ObsTerm(func=_eef_pos)
        eef_quat   = ObsTerm(func=_eef_quat)
        cube_pos   = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})
        box_pos    = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("box")})
        table_cam  = ObsTerm(func=base_mdp.image,
                             params={"sensor_cfg": SceneEntityCfg("table_camera"),
                                     "data_type": "rgb", "normalize": False})
        wrist_cam  = ObsTerm(func=base_mdp.image,
                             params={"sensor_cfg": SceneEntityCfg("wrist_cam"),
                                     "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            # All 37 SPAD sensors: depth + world-frame pose per sensor.
            # Recorded as flat obs keys:
            #   obs/spad_fr3_link6_sensor_0        (T, 8, 8)
            #   obs/spad_fr3_link6_sensor_0_pos_w  (T, 3)
            #   obs/spad_fr3_link6_sensor_0_quat_w (T, 4)
            for attr_name in _SPAD_CFGS:
                sc = SceneEntityCfg(attr_name)
                setattr(self, attr_name, ObsTerm(
                    func=base_mdp.image,
                    params={"sensor_cfg": sc, "data_type": "distance_to_camera", "normalize": False},
                ))
                setattr(self, f"{attr_name}_pos_w", ObsTerm(
                    func=_spad_sensor_pos_w,
                    params={"sensor_cfg": sc},
                ))
                setattr(self, f"{attr_name}_quat_w", ObsTerm(
                    func=_spad_sensor_quat_w,
                    params={"sensor_cfg": sc},
                ))
            for i in range(NUM_YCB_SLOTS):
                sn = f"ycb_slot{i}"
                ac = SceneEntityCfg(sn)
                setattr(
                    self,
                    f"{sn}_pos",
                    ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": ac}),
                )
                setattr(
                    self,
                    f"{sn}_quat",
                    ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": ac}),
                )

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        table_cam_rgb = ObsTerm(func=base_mdp.image,
                                params={"sensor_cfg": SceneEntityCfg("table_camera"),
                                        "data_type": "rgb", "normalize": False})
        wrist_cam_rgb = ObsTerm(func=base_mdp.image,
                                params={"sensor_cfg": SceneEntityCfg("wrist_cam"),
                                        "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskTermsCfg(ObsGroup):
        """Signals consumed by record_demos + mimic annotator."""
        grasp = ObsTerm(func=grasp_signal)
        place = ObsTerm(func=place_signal)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy       = PolicyCfg()
    rgb_camera   = RGBCameraPolicyCfg()
    subtask_terms = SubtaskTermsCfg()


# ---------------------------------------------------------------------------
# Actions  — DifferentialIK arm  +  binary gripper
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["fr3_joint[1-7]"],
        body_name="fr3_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            # Match Isaac Lab Franka cube-stack IK-rel (`stack_ik_rel_env_cfg.py`): default λ=0.01.
            # (λ=0.06 was making teleop feel heavy / slow to converge vs SeattleLab Panda.)
            ik_params={"lambda_val": 0.01},
        ),
        # Keyboard teleop: scale × Se3 deltas; raise for faster motion, lower if DIK overshoots in skillgen.
        scale=0.85,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    gripper_action = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["fr3_finger_joint.*"],
        open_command_expr={"fr3_finger_joint.*": 0.04},
        close_command_expr={"fr3_finger_joint.*": 0.0},
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_objects = EventTerm(
        func=randomize_ycb_objects,
        mode="reset",
        params={},
    )
    fr3_skin_green_visual = EventTerm(
        func=apply_fr3_skin_green_visual,
        mode="reset",
        params={},
    )
    fr3_skin_green_startup = EventTerm(
        func=apply_fr3_skin_green_visual,
        mode="startup",
        params={},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success  = DoneTerm(func=cube_in_box)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    pass


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@configclass
class FrankaYCBEnvCfg(ManagerBasedRLEnvCfg):
    scene        = YCBSceneCfg(num_envs=1, env_spacing=2.5)
    observations = ObservationsCfg()
    actions      = ActionsCfg()
    rewards      = RewardsCfg()
    events       = EventCfg()
    terminations = TerminationsCfg()

    # YCB clutter: annulus around the randomized cube (independent r,θ per object + spacing).
    ycb_randomize_near_cube: bool = True
    ycb_near_cube_radius_min: float = 0.26  # m — annulus inner edge (clear cube + props)
    ycb_near_cube_radius_max: float = 0.55  # m — annulus outer edge — wider spread on counter
    # 1.0 = every prop uses the annulus (still random positions); <1 blends with full counter.
    ycb_near_cube_per_object_probability: float = 1.0
    ycb_near_cube_max_placement_tries: int = 400
    # Min XY spacing between YCB props (and vs cube/box via placed radii) — higher = less overlap.
    ycb_inter_object_min_xy: float = 0.22
    # Min / max XY distance between cube and box — gripper must clear box when picking cube.
    cube_box_spawn_min_xy: float = 0.3
    cube_box_spawn_max_xy: float = 0.45
    # Min XY distance from pink box center for YCB clutter (also uses _BOX_SIZE + spacing per object).
    ycb_box_spawn_min_xy: float = 0.30
    # (x, y, radius) disks in env XY — keep cube, box, and YCB out of fixed kitchen USD props. () = off.
    task_spawn_exclude_xy: tuple = TASK_SPAWN_EXCLUDE_XY_DEFAULT
    # Extra gap when testing object footprint vs kitchen disks (object radius + margin).
    task_spawn_kitchen_footprint_margin_m: float = 0.04
    # Extra gap vs ``MOLMO_STATIC_EXCLUDE_XY`` (robot disk only; Apple/Book/bowl off in USDA) for cube/box/YCB.
    molmo_spawn_exclude_margin_m: float = 0.0
    # Uniform jitter (m) on precomputed YCB parking XY each reset.
    ycb_spot_jitter_m: float = 0.01

    # Set in __post_init__ so `sim_device` matches `self.sim.device` (same pattern as
    # `isaaclab_tasks/.../stack/config/franka/stack_ik_rel_env_cfg.py`).
    teleop_devices: DevicesCfg | None = None

    def __post_init__(self):
        self.episode_length_s    = 100.0
        # Higher physics rate improves contact + joint tracking stability.
        self.sim.dt              = 1 / 120
        self.decimation          = 4
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 3
        # Keyboard deltas → DIK (paired with ActionsCfg.arm_action.scale above).
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.115,
                    rot_sensitivity=0.115,
                    sim_device=self.sim.device,
                ),
            }
        )


FrankaYCBEnvCfg_PLAY           = FrankaYCBEnvCfg
FrankaYCBAllObjectsEnvCfg_PLAY = FrankaYCBEnvCfg
