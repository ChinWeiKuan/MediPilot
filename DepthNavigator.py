import argparse
import cv2
import numpy as np

def decide_action(depth: np.ndarray,
                  roi_bottom: float = 0.45,   # 只看畫面底部 45%
                  center_pctl: int = 20,      # 中央用較低分位（更敏感障礙）
                  side_pctl: int = 50,        # 左右用中位數
                  block_thresh: float = 0.17, # 中央「通暢度」低於此值：STOP
                  turn_deadband: float = 0.06 # 左右差異小於此值：FORWARD
                  ):
    """
    depth: (H, W) float32，相對深度（MiDaS 輸出）
    return: action(str), dbg(dict)
    """
    H, W = depth.shape
    y0 = int(H * (1.0 - roi_bottom))
    roi = depth[y0:, :]

    # 用百分位建立相對尺度（避免 MiDaS 數值方向/範圍不同）
    near = np.percentile(roi, 25)
    far  = np.percentile(roi, 75)
    if far - near < 1e-6:
        far = near + 1e-6

    # 分成左/中/右
    xL = int(W * 0.33)
    xC = int(W * (0.33 + 0.34))

    def pct(a, p):
        if a.size == 0: return 0.0
        return float(np.percentile(a, p))

    L = pct(roi[:, :xL], side_pctl)       if xL > 0   else 0.0
    C = pct(roi[:, xL:xC], center_pctl)   if xC > xL  else 0.0
    R = pct(roi[:, xC:], side_pctl)       if W > xC   else 0.0

    # 正規化到 [0,1]：值越大 = 越通暢
    def norm(x): return float(np.clip((x - near) / (far - near + 1e-8), 0.0, 1.0))
    Ln, Cn, Rn = norm(L), norm(C), norm(R)

    # 先看中央是否很擠 → 停
    if Cn < block_thresh:
        action = "STOP"
    else:
        # 左右比較
        diff = Ln - Rn  # >0 表示左邊更空 -> LEFT
        if abs(diff) < turn_deadband:
            action = "FORWARD"
        else:
            action = "LEFT" if diff > 0 else "RIGHT"

    dbg = {
        "H": H, "W": W, "roi_y0": y0,
        "near": near, "far": far,
        "L_raw": L, "C_raw": C, "R_raw": R,
        "Ln": Ln, "Cn": Cn, "Rn": Rn,
        "action": action
    }
    return action, dbg

def visualize_once(rgb: np.ndarray, depth: np.ndarray, dbg: dict,
                   out_path="discrete_overlay.png"):
    """畫出 ROI 與三區分界，並把決策結果標在圖上。"""
    H, W = depth.shape
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    # ROI 與三區
    y0 = dbg["roi_y0"]
    cv2.rectangle(vis, (0, y0), (W - 1, H - 1), (0, 255, 0), 2)
    xL = int(W * 0.33)
    xC = int(W * (0.33 + 0.34))
    for x in [xL, xC]:
        cv2.line(vis, (x, y0), (x, H - 1), (255, 255, 255), 1)

    txt = f"ACTION={dbg['action']} | Ln={dbg['Ln']:.2f} Cn={dbg['Cn']:.2f} Rn={dbg['Rn']:.2f}"
    cv2.putText(vis, txt, (10, max(25, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2)

    # 另存深度著色圖方便對照
    p2, p98 = np.percentile(depth, (2, 98))
    depth_clip = np.clip(depth, p2, p98)
    depth_norm = (255 * (depth_clip - depth_clip.min()) /
                  (depth_clip.max() - depth_clip.min() + 1e-8)).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    cv2.imwrite("depth_output_2.png", depth_color)
    cv2.imwrite(out_path, vis)