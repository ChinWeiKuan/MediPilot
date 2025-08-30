"""
navigation.py
----------------
Runtime navigator for Raspberry Pi.

Usage (on RPi):
  python3 navigation.py --plan planned_actions.json --speed 0.22 --trim -0.15 --t90l 0.72 --t90r 0.70

- Reads an actions JSON exported by OccupancyGrid.py (planned_actions.json by default)
- Converts "move meters" into seconds using calibrated linear speed (m/s)
- Executes moves/turns using your existing motor control functions from car_run_turn.py
- Defaults are set from your latest calibration: turn 90° (L/R) ≈ 0.72s / 0.70s, trim = -0.15
"""

import json
import argparse
import time

# === Calibrations / defaults (can be overridden by CLI) ===
DEFAULT_SPEED_MPS = 0.325      # Forward speed (m/s)
TURN_90_LEFT_S = 0.65          # Turn left 90° s
TURN_90_RIGHT_S = 0.65         # Turn right 90° s
SLACK_S = 0.05                 # 步驟間的小間隔，避免驅動黏連
DEFAULT_TRIM = 0         # 實測：f 1 -0.15 可走直

# 匯入你的馬達控制原語
try:
    from car_run_turn import forward, backward, turnLeft, turnRight, stop
except Exception as e:
    raise RuntimeError("無法匯入 car_run_turn.py，請確認此檔案在 RPI 上存在且可用") from e


def run_actions(actions, speed_mps=DEFAULT_SPEED_MPS, trim=DEFAULT_TRIM,
                t90_left=TURN_90_LEFT_S, t90_right=TURN_90_RIGHT_S, dry_run=False):
    """
    執行規劃端匯出的動作列表。每個元素長這樣：
      - {"type": "move", "meters": 1.25, "dir": "E"}
      - {"type": "turn", "dir": "left"|"right"|"u-turn"}
    """
    for i, act in enumerate(actions, 1):
        atype = act.get("type")
        if atype == "move":
            meters = float(act.get("meters", 0.0))
            secs = meters / float(speed_mps) if speed_mps > 0 else 0.0
            print(f"[{i:02d}] MOVE {meters:.3f} m  -> {secs:.2f} s (speed={speed_mps:.3f} m/s, trim={trim:+.2f})")
            if not dry_run:
                forward(secs, trim)
                time.sleep(SLACK_S)
        elif atype == "turn":
            d = act.get("dir", "").lower()
            if d == "left":
                print(f"[{i:02d}] TURN LEFT 90° -> {t90_left:.2f} s")
                if not dry_run:
                    turnLeft(t90_left)
            elif d == "right":
                print(f"[{i:02d}] TURN RIGHT 90° -> {t90_right:.2f} s")
                if not dry_run:
                    turnRight(t90_right)
            elif d in ("u-turn", "uturn", "u"):
                print(f"[{i:02d}] U-TURN 180° -> {2*t90_right:.2f} s")
                if not dry_run:
                    # 預設用右轉完成 180°，如需左轉可在規劃層輸入兩次 left
                    turnRight(t90_right)
                    time.sleep(SLACK_S)
                    turnRight(t90_right)
            else:
                print(f"[{i:02d}] [WARN] Unknown turn dir: {d!r}, skipping")
            if not dry_run:
                time.sleep(SLACK_S)
        else:
            print(f"[{i:02d}] [WARN] Unknown action type: {atype!r}, skipping")
    print("=== Navigation finished ===")


def main():
    ap = argparse.ArgumentParser(description="Execute planned actions on RPi (no point cloud needed).")
    ap.add_argument("--plan", default="planned_actions.json", help="Path to JSON plan exported by OccupancyGrid.py")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED_MPS, help="Linear speed in m/s (calibrated)")
    ap.add_argument("--trim",  type=float, default=DEFAULT_TRIM, help="Forward trim (positive weakens RIGHT wheel)")
    ap.add_argument("--t90",   type=float, default=None, help="[Deprecated] Seconds per 90° turn for BOTH sides; overrides --t90l/--t90r if set")
    ap.add_argument("--t90l",  type=float, default=TURN_90_LEFT_S,  help="Seconds per 90° LEFT turn")
    ap.add_argument("--t90r",  type=float, default=TURN_90_RIGHT_S, help="Seconds per 90° RIGHT turn")
    ap.add_argument("--dry",   action="store_true", help="Print what would run without moving motors")
    args = ap.parse_args()

    # Back-compat: if --t90 is provided, apply it to both left/right
    if args.t90 is not None:
        args.t90l = args.t90
        args.t90r = args.t90

    print("=== Navigation run start ===")
    print(f"Plan file : {args.plan}")
    print(f"Speed     : {args.speed:.3f} m/s")
    print(f"Trim      : {args.trim:+.2f}")
    print(f"T90-L     : {args.t90l:.2f} s per 90° LEFT")
    print(f"T90-R     : {args.t90r:.2f} s per 90° RIGHT")
    print(f"Dry-run   : {args.dry}")
    print("============================")

    with open(args.plan, "r", encoding="utf-8") as f:
        obj = json.load(f)

    actions = obj.get("actions", obj)  # 允許整包物件或純 actions 陣列
    if not isinstance(actions, list):
        raise ValueError("Plan JSON 缺少 'actions' 陣列")

    if args.dry:
        print(f"Plan size: {len(actions)} steps")
    run_actions(actions, speed_mps=args.speed, trim=args.trim, t90_left=args.t90l, t90_right=args.t90r, dry_run=args.dry)


if __name__ == "__main__":
    main()