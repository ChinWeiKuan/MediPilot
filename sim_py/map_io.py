import cv2, yaml, numpy as np, os
def load_map(yaml_path):
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
    img_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
    # 讀 PGM：白=free(255), 黑=occupied(0)；依 negate 調整
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if meta.get('negate', 0) == 1:
        img = 255 - img
    occ = (img < int(meta.get('occupied_thresh',0.65)*255)).astype(np.uint8)  # 1=障礙
    free = (img > int(meta.get('free_thresh',0.2)*255)).astype(np.uint8)      # 1=可走
    grid = np.where(occ==1, 1, np.where(free==1, 0, -1))  # 1=障礙,0=可走,-1=未知
    res = float(meta['resolution'])
    origin = np.array(meta.get('origin', [0,0,0]), dtype=float)
    return grid, res, origin