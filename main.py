import cv2
import time

from MiDaS import MiDaS
from DepthNavigator import  decide_action, visualize_once

def _preprocess_frame_for_midas(bgr: np.ndarray, max_side: int = 512) -> np.ndarray:
    """BGR -> RGB，並限制長邊到 max_side（與 preprocess_img 一致的效果）"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb  # uint8, (H,W,3)

def main():
    # initialize MIDaS
    midas = MiDaS("MiDaS_small", "cpu")

    # initialize Camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # 後備方案
        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open camera 0. If on Raspberry Pi (Bookworm), try: `libcamerify python3 main.py`")

    # inference
    interval_s = 2.0
    print("Started. Press Ctrl+C to stop.")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            rgb = _preprocess_frame_for_midas(bgr, max_side=512)
            depth = midas.predict_depth(rgb)
            action, dbg = decide_action(depth)  # 只輸出四選一：LEFT/RIGHT/FORWARD/STOP

            print(f"[{time.strftime('%H:%M:%S')}] ACTION={action} "
                  f"(Ln={dbg['Ln']:.2f}, Cn={dbg['Cn']:.2f}, Rn={dbg['Rn']:.2f})",
                  flush=True)

            time.sleep(interval_s)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    # img_rgb = midas.preprocess_img(img="input.jpg")
    # depth = midas.predict_depth(img_rgb)
    # midas.visualize_depth(depth)
    # action, dbg = decide_action(depth)
    # visualize_once(img_rgb, depth, dbg)
    # print(action)

if __name__ == '__main__':
    main()