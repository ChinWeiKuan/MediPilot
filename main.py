import cv2
import time
import numpy as np
import os

from MiDaS import MiDaS
from DepthNavigator import  decide_action, visualize_once
from picamera2 import Picamera2

def _shrink_to_max_side(rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    scale = max_side / max(h, w)
    return cv2.resize(rgb, (int(w*scale), int(h*scale)), inter
    polation=cv2.INTER_AREA)

def main():
    # initialize MIDaS
    midas = MiDaS("MiDaS_small", "cpu")

    # initialize Camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    os.makedirs("captures", exist_ok=True)

    # Setup OpenCV display window if possible
    display_enabled = bool(os.environ.get("DISPLAY"))
    if display_enabled:
        try:
            cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
            print("Preview  : OpenCV window enabled (VNC/X11)")
        except Exception as e:
            print(f"Preview  : disabled ({e})")
            display_enabled = False
    else:
        print("Preview  : disabled (no DISPLAY)")

    # inference
    interval_s = 0.5
    print("Started. Press Ctrl+C to stop.")
    try:
        while True:
            # Picamera2 直接回傳 RGB888 (H,W,3), uint8
            rgb = picam2.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # 顯示到 VNC/OpenCV 視窗
            if display_enabled:
                try:
                    cv2.imshow("Camera", bgr)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"[Preview error] {e}")
                    display_enabled = False

            # 每 0.5s 存一張圖
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join("captures", f"{ts}.jpg")
            cv2.imwrite(out_path, bgr)
            print(f"Saved {out_path}")

            rgb = _shrink_to_max_side(rgb, max_side=512)

            depth = midas.predict_depth(rgb)
            action, dbg = decide_action(depth)

            print(f"[{time.strftime('%H:%M:%S')}] ACTION={action} "
                  f"(Ln={dbg['Ln']:.2f}, Cn={dbg['Cn']:.2f}, Rn={dbg['Rn']:.2f})",
                  flush=True)

            time.sleep(interval_s)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if 'display_enabled' in locals() and display_enabled:
                cv2.destroyAllWindows()
        except Exception:
            pass
        picam2.stop()

    # img_rgb = midas.preprocess_img(img="input.jpg")
    # depth = midas.predict_depth(img_rgb)
    # midas.visualize_depth(depth)
    # action, dbg = decide_action(depth)
    # visualize_once(img_rgb, depth, dbg)
    # print(action)

if __name__ == '__main__':
    main()