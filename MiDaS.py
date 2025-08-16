
import torch, cv2, timm, numpy as np

class MiDaS:
    def __init__(self, model_type="MiDaS_small", device="cpu"):
        self.device = torch.device(device)
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transforms = transforms.small_transform

    def preprocess_img(self, img: str) -> np.ndarray:
        # Load inputs
        img_bgr = cv2.imread(img)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Limit the size to 512px
        h, w = img_rgb.shape[:2]
        scale = 512.0 / max(h, w)
        if scale < 1.0:
            img_rgb = cv2.resize(
                img_rgb, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )
        return img_rgb

    def predict_depth(self, img: np.ndarray) -> np.ndarray:
        img_tensor = self.transforms(img).to(self.device)  # shape: (1, 3, H, W)
        with torch.no_grad():
            pred = self.model(img_tensor)  # (1, H', W')
            pred = torch.nn.functional.interpolate(  # Resize to original size
                pred.unsqueeze(1), size=img.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()  # (H, W)
        depth = pred.cpu().numpy()
        return depth

    def visualize_depth(self, depth: np.ndarray):
        p2, p98 = np.percentile(depth, (2, 98))
        depth_clipped = np.clip(depth, p2, p98)
        depth_norm = (255 * (depth_clipped - depth_clipped.min()) /
                      (np.ptp(depth_clipped) + 1e-8)).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        cv2.imwrite("depth_output.png", depth_color)