

"""
navigation_apriltag.py
----------------------
Visual-aided navigator for Raspberry Pi using AprilTags to continuously correct pose.

TL;DR
  - Plan path on PC as usual -> export planned_actions.json (or waypoints).
  - On RPi, run this file: it will follow the plan but continuously re-localize with AprilTags.
  - If tags are momentarily lost, it falls back to short dead-reckoning steps until tags reappear.

Expected inputs
  (1) planned_actions.json  - from OccupancyGrid.py (either a full object with 'actions' or 'waypoints', or just a list)
  (2) camera.yaml           - camera intrinsics (fx, fy, cx, cy[, k1..])
  (3) apriltags.json        - tag map in the same global/map frame as the plan:
        {
          "tags": [
            {"id": 5, "x": 2.350, "y": 1.100, "yaw_deg": 90.0},
            {"id": 7, "x": 4.800, "y": 1.100, "yaw_deg": 90.0},
            ...
          ]
        }
      All positions are meters; yaw_deg is rotation of the tag's +x axis relative to the map's +x (CCW positive).

Requires
  - OpenCV (cv2)
  - numpy
  - pupil_apriltags  (pip install pupil-apriltags)
  - pyyaml           (for reading camera.yaml)
  - car_run_turn.py  (motor primitives: forward(seconds, trim), turnLeft(seconds), turnRight(seconds), stop())

Notes
  - This implementation uses simple SE2 math and assumes AprilTag detections provide T_cam_tag (pose of the tag in camera frame).
    We invert it to get T_tag_cam, then compose with the known map pose of the tag and the camera extrinsics to estimate T_map_robot.
  - For robustness you can later add filtering (e.g., a complementary/KF), but this version uses per-frame median fusion when multiple tags are seen.

Usage examples
  # Dry-run only prints what it would do:
  python3 navigation_apriltag.py --dry

  # Real run (example parameters; adjust to your calibration):
  python3 navigation_apriltag.py --plan planned_actions.json --cam camera.yaml --tags apriltags.json \
      --speed 0.22 --trim -0.15 --t90 0.72 --tag-size 0.055 --angle-thresh 10 --lookahead 0.50

"""

from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# --- Optional imports with nice errors ---
try:
    import cv2
except Exception as e:
    cv2 = None
try:
    import yaml
except Exception:
    yaml = None
try:
    from pupil_apriltags import Detector
except Exception:
    Detector = None

# --- Motor primitives (must exist on RPi) ---
try:
    from car_run_turn import forward, backward, turnLeft, turnRight, stop
except Exception:
    # Fallback dummies for dry-run on PC
    def forward(seconds: float, trim: float = 0.0):
        print(f"[MOTOR] forward {seconds:.2f}s trim={trim:+.2f}")
    def backward(seconds: float):
        print(f"[MOTOR] backward {seconds:.2f}s")
    def turnLeft(seconds: float):
        print(f"[MOTOR] turnLeft {seconds:.2f}s")
    def turnRight(seconds: float):
        print(f"[MOTOR] turnRight {seconds:.2f}s")
    def stop():
        print("[MOTOR] stop()")

# -----------------------
# Geometry / SE2 helpers
# -----------------------
def se2(x: float, y: float, yaw: float) -> np.ndarray:
    """Return 3x3 SE2 transform matrix (row-major)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]], dtype=float)

def inv_se2(T: np.ndarray) -> np.ndarray:
    R = T[:2, :2]
    t = T[:2, 2]
    Rt = R.T
    tinv = -Rt @ t
    Tinv = np.eye(3)
    Tinv[:2, :2] = Rt
    Tinv[:2, 2] = tinv
    return Tinv

def yaw_from_R(R: np.ndarray) -> float:
    """Extract planar yaw from rotation matrix; assumes small roll/pitch."""
    return math.atan2(R[1, 0], R[0, 0])

# -----------------------
# Data structures
# -----------------------
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class CameraExtrinsics:
    """Camera pose in robot frame (T_robot_cam)."""
    tx: float = 0.0
    ty: float = 0.0
    yaw: float = 0.0  # radians

@dataclass
class TagPose:
    """Tag pose in map frame (T_map_tag)."""
    x: float
    y: float
    yaw: float  # radians

# -----------------------
# I/O helpers
# -----------------------
def load_camera_yaml(path: str) -> CameraIntrinsics:
    if yaml is None:
        raise RuntimeError("pyyaml not installed. pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    # Accept several common schemas
    if "camera_matrix" in obj and "data" in obj["camera_matrix"]:
        k = obj["camera_matrix"]["data"]
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    elif "fx" in obj and "fy" in obj and "cx" in obj and "cy" in obj:
        fx, fy, cx, cy = float(obj["fx"]), float(obj["fy"]), float(obj["cx"]), float(obj["cy"])
    else:
        raise ValueError("camera.yaml missing fx,fy,cx,cy or camera_matrix.data")
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

def load_tag_map(path: str) -> Dict[int, TagPose]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    tags = obj.get("tags", obj)  # allow raw list
    out: Dict[int, TagPose] = {}
    for entry in tags:
        tid = int(entry["id"])
        x = float(entry["x"])
        y = float(entry["y"])
        yaw_deg = float(entry.get("yaw_deg", 0.0))
        out[tid] = TagPose(x=x, y=y, yaw=math.radians(yaw_deg))
    return out

def load_plan(path: str) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    """
    Return (waypoints, maybe_initial_yaw).
    Accept either:
      - {"waypoints":[[x,y],...], "start_yaw_deg":...}
      - {"actions":[...]}  (we will derive waypoints assuming orthogonal moves & 90° turns)
      - [[x,y], ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, tuple)):
        return [(float(a), float(b)) for a, b in obj], None
    if "waypoints" in obj:
        wps = [(float(a), float(b)) for a, b in obj["waypoints"]]
        yaw0 = math.radians(float(obj.get("start_yaw_deg", 0.0)))
        return wps, yaw0
    if "actions" in obj:
        return actions_to_waypoints(obj["actions"]), None
    # fallback: treat as empty
    return [], None

def actions_to_waypoints(actions: List[dict]) -> List[Tuple[float, float]]:
    """Integrate a Manhattan path into absolute waypoints in map frame starting at (0,0) heading +X."""
    x, y, yaw = 0.0, 0.0, 0.0
    wps = [(x, y)]
    for a in actions:
        t = a.get("type")
        if t == "turn":
            d = str(a.get("dir", "")).lower()
            if d == "left":
                yaw += math.pi/2
            elif d == "right":
                yaw -= math.pi/2
            elif d in ("u", "u-turn", "uturn"):
                yaw += math.pi
        elif t == "move":
            meters = float(a.get("meters", 0.0))
            x += meters * math.cos(yaw)
            y += meters * math.sin(yaw)
            wps.append((x, y))
    return wps

# -----------------------
# AprilTag pose to robot pose
# -----------------------
def estimate_robot_pose_from_tags(detections, tag_map: Dict[int, TagPose],
                                  K: CameraIntrinsics, tag_size_m: float,
                                  T_robot_cam: CameraExtrinsics) -> Optional[Tuple[float, float, float]]:
    """
    Returns (x, y, yaw) in map frame if any tag is matched; else None.
    Assumes detector returns T_cam_tag (tag pose in camera frame) via (pose_R, pose_t).
    """
    if detections is None:
        return None
    poses = []
    for det in detections:
        tid = int(getattr(det, "tag_id", getattr(det, "id", -1)))
        if tid not in tag_map:
            continue
        # If we have pose, use it
        pose_R = getattr(det, "pose_R", None)
        pose_t = getattr(det, "pose_t", None)
        if pose_R is None or pose_t is None:
            # Some detector configs require estimate_tag_pose=True and camera params
            continue
        R_ct = np.array(pose_R, dtype=float)  # rotation from tag -> cam? (see below)
        t_ct = np.array(pose_t, dtype=float).reshape(3)  # translation tag->cam in cam frame
        # We assume R_ct, t_ct give T_cam_tag, i.e., x_tag in cam = R_ct @ x + t_ct.
        # Then T_tag_cam = inverse:
        T_tag_cam = np.eye(3)
        yaw_ct = yaw_from_R(R_ct[:2, :2])
        # For planar SE2, approximate inverse by negating yaw and rotating translation in XY
        T_cam_tag_SE2 = se2(t_ct[0], t_ct[1], yaw_ct)  # cam->tag
        T_tag_cam_SE2 = inv_se2(T_cam_tag_SE2)

        tag_pose = tag_map[tid]
        T_map_tag = se2(tag_pose.x, tag_pose.y, tag_pose.yaw)
        T_map_cam = T_map_tag @ T_tag_cam_SE2

        # Apply camera extrinsics (camera in robot frame)
        T_rc = se2(T_robot_cam.tx, T_robot_cam.ty, T_robot_cam.yaw)  # T_robot_cam
        T_cr = inv_se2(T_rc)
        T_map_robot = T_map_cam @ T_cr

        x_r, y_r = float(T_map_robot[0, 2]), float(T_map_robot[1, 2])
        yaw_r = yaw_from_R(T_map_robot[:2, :2])
        poses.append((x_r, y_r, yaw_r))

    if not poses:
        return None
    # Robust fusion: median of available tag-based estimates
    arr = np.array(poses)
    x = float(np.median(arr[:, 0]))
    y = float(np.median(arr[:, 1]))
    # circular median for yaw
    yaw = float(math.atan2(np.median(np.sin(arr[:, 2])), np.median(np.cos(arr[:, 2]))))
    return x, y, yaw

# -----------------------
# Controller
# -----------------------
def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a

def control_step(robot_xyy: Tuple[float, float, float],
                 target_xy: Tuple[float, float],
                 speed_mps: float,
                 t90: float,
                 trim: float,
                 angle_thresh_rad: float,
                 step_m: float,
                 dry: bool) -> None:
    """One control step: rotate if heading error too large, else move a short straight step."""
    rx, ry, ryaw = robot_xyy
    tx, ty = target_xy
    dx, dy = tx - rx, ty - ry
    dist = math.hypot(dx, dy)
    target_yaw = math.atan2(dy, dx)
    err_yaw = wrap_to_pi(target_yaw - ryaw)

    if abs(err_yaw) > angle_thresh_rad:
        # Rotate toward target: convert yaw error to seconds using t90
        sec_per_rad = t90 / (math.pi/2.0)
        sec = min(abs(err_yaw) * sec_per_rad, t90)  # clamp one 90° at a time
        direction = "left" if err_yaw > 0 else "right"
        print(f"[CTRL] rotate {direction} by {math.degrees(err_yaw):.1f}° -> {sec:.2f}s")
        if not dry:
            (turnLeft if direction == "left" else turnRight)(sec)
        return

    # Go straight a small step (bounded by remaining distance)
    step = min(step_m, dist)
    sec = step / max(1e-6, speed_mps)
    print(f"[CTRL] forward {step:.3f} m -> {sec:.2f}s (trim={trim:+.2f})")
    if not dry:
        forward(sec, trim=trim)

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="AprilTag-aided navigation (visual localization + waypoint following)")
    ap.add_argument("--plan", default="planned_actions.json", help="Path to plan json (contains actions or waypoints)")
    ap.add_argument("--cam",  default="camera.yaml", help="Path to camera intrinsics yaml (fx, fy, cx, cy)")
    ap.add_argument("--tags", default="apriltags.json", help="Path to AprilTag map json")
    ap.add_argument("--tag-size", type=float, default=0.055, help="Physical tag size (edge length, meters)")
    ap.add_argument("--family", default="tag36h11", help="AprilTag family for detector")
    ap.add_argument("--speed", type=float, default=0.22, help="Linear speed m/s (calibrated)")
    ap.add_argument("--t90", type=float, default=0.72, help="Seconds per 90-degree in-place turn")
    ap.add_argument("--trim", type=float, default=-0.15, help="Forward trim (positive weakens RIGHT wheel)")
    ap.add_argument("--lookahead", type=float, default=0.50, help="Lookahead distance toward next waypoint (m)")
    ap.add_argument("--angle-thresh", type=float, default=10.0, help="Rotate-in-place threshold (deg)")
    ap.add_argument("--step", type=float, default=0.15, help="Straight step distance when aligned (m)")
    ap.add_argument("--tag-timeout", type=float, default=1.5, help="If no tags this long (s), keep only tiny steps")
    ap.add_argument("--cam-yaw-deg", type=float, default=0.0, help="Camera yaw relative to robot frame (deg, CCW+)")
    ap.add_argument("--cam-tx", type=float, default=0.0, help="Camera X in robot frame (m)")
    ap.add_argument("--cam-ty", type=float, default=0.0, help="Camera Y in robot frame (m)")
    ap.add_argument("--dry", action="store_true", help="Print only, do not move motors")
    args = ap.parse_args()

    print("=== AprilTag Navigator start ===")
    print(f"Plan      : {args.plan}")
    print(f"Camera    : {args.cam}")
    print(f"Tags map  : {args.tags}")
    print(f"Tag size  : {args.tag_size} m, family={args.family}")
    print(f"Speed     : {args.speed:.3f} m/s, t90={args.t90:.2f}s, trim={args.trim:+.2f}")
    print(f"Lookahead : {args.lookahead:.2f} m, angle_thresh={args.angle_thresh:.1f}° step={args.step:.2f} m")
    print(f"Cam extr  : yaw={args.cam_yaw_deg:.1f}° tx={args.cam_tx:.3f} ty={args.cam_ty:.3f}")
    print(f"Dry-run   : {args.dry}")
    print("===============================")

    # Load resources
    waypoints, yaw0 = load_plan(args.plan)
    if not waypoints:
        print("[WARN] no waypoints in plan; will do nothing.")
        return
    tag_map = load_tag_map(args.tags)
    K = load_camera_yaml(args.cam)
    T_rc = CameraExtrinsics(tx=args.cam_tx, ty=args.cam_ty, yaw=math.radians(args.cam_yaw_deg))

    # Video source
    if cv2 is None:
        raise RuntimeError("OpenCV not available. Install opencv-python.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0")

    if Detector is None:
        raise RuntimeError("pupil_apriltags not installed. pip install pupil-apriltags")
    det = Detector(families=args.family, nthreads=1, quad_decimate=1.5, quad_sigma=0.0, refine_edges=True)

    last_tag_ts = 0.0
    robot_pose: Optional[Tuple[float, float, float]] = None  # (x,y,yaw) in map
    waypoint_idx = 0
    angle_thresh_rad = math.radians(args.angle_thresh)

    try:
        while waypoint_idx < len(waypoints):
            ok, frame = cap.read()
            if not ok:
                print("[WARN] camera read failed")
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect + estimate pose
            dets = det.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(K.fx, K.fy, K.cx, K.cy),
                tag_size=args.tag_size,
            )

            pose_now = estimate_robot_pose_from_tags(dets, tag_map, K, args.tag_size, T_rc)
            if pose_now is not None:
                robot_pose = pose_now
                last_tag_ts = time.time()

            # If never localized yet, rotate slowly searching for tags
            if robot_pose is None:
                print("[LOC] No pose yet. Searching tags...")
                if not args.dry:
                    turnLeft(min(0.3, args.t90))  # short sweep
                time.sleep(0.05)
                continue

            # Compute target using lookahead on current segment
            # Move toward current waypoint; if close enough, advance
            tgt = waypoints[waypoint_idx]
            rx, ry, ryaw = robot_pose
            dx, dy = tgt[0] - rx, tgt[1] - ry
            dist_to_wp = math.hypot(dx, dy)
            if dist_to_wp < max(0.2, args.step * 0.8):
                print(f"[NAV] reached wp#{waypoint_idx} @ ({tgt[0]:.2f},{tgt[1]:.2f})")
                waypoint_idx += 1
                continue

            # If far, create a lookahead point along the segment
            seg_len = max(1e-6, dist_to_wp)
            la = min(args.lookahead, seg_len)
            target = (rx + (dx/seg_len)*la, ry + (dy/seg_len)*la)

            # If tags lost for too long, reduce aggression
            since = time.time() - last_tag_ts
            step_m = args.step * (0.3 if since > args.tag_timeout else 1.0)
            if since > args.tag_timeout:
                print(f"[LOC] tags lost for {since:.1f}s -> conservative step {step_m:.2f} m")

            control_step(robot_pose, target, args.speed, args.t90, args.trim, angle_thresh_rad, step_m, args.dry)

        print("=== Navigation complete ===")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        stop()

if __name__ == "__main__":
    main()