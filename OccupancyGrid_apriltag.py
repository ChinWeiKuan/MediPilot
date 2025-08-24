import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque
import cv2
import json
ㄛm
from pupil_apriltags import Detector

# === Config ===
# COORD_MODE: 'map' → 以「地圖ap左下角」為 (0,0)（全部正數，直覺給公尺）
#              'world' → 使用 PLY 原始世界座標（可能含負數）
COORD_MODE = 'map'

# AprilTag / Camera config
TAG_CFG = "apriltags.json"      # {"tag_size_m": 0.16, "tags":[{"id":3,"x":...,"y":...,"z":...,"yaw_deg":...}, ...]}
CAMERA_YAML = "camera.yaml"     # 含 camera_matrix, dist_coeffs
APRILTAG_FAMILY = "tag36h11"    # 常用族群

# 讀取點雲
pcd = o3d.io.read_point_cloud("Scene_test.ply")
pts = np.asarray(pcd.points)   # Nx3, [x,y,z]
print("點數:", pts.shape)

print("bbox(x,y,z) min:", pts.min(0), "max:", pts.max(0))
print("range meters? (max-min):", (pts.max(0)-pts.min(0)))
print("norm length:", np.linalg.norm(pts.max(0)-pts.min(0)))

# --- Camera / Tag DB helpers ---

def load_camera_yaml(path):
    K = None; dist = None
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if fs.isOpened():
        K = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("dist_coeffs").mat()
        fs.release()
    if K is None:
        # 後備：嘗試以 JSON/Numpy 方式讀取，或給予單位陣避免程式中斷
        K = np.array([[500,0,320],[0,500,240],[0,0,1]], dtype=np.float32)
        dist = np.zeros((1,5), dtype=np.float32)
        print("[WARN] 無法讀取 camera.yaml，改用預設內參。")
    return K.astype(np.float32), dist.astype(np.float32)


def load_tag_db(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        size = float(cfg.get("tag_size_m", 0.16))
        tags = {int(t["id"]): t for t in cfg.get("tags", [])}
        return size, tags
    except Exception as e:
        print(f"[WARN] 無法讀取 {path}: {e}; 以空資料繼續。")
        return 0.16, {}


def rt_to_T(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R.astype(np.float32)
    T[:3, 3] = t.reshape(3).astype(np.float32)
    return T


def yaw_T(yaw_rad):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T = np.eye(4, dtype=np.float32)
    T[0,0]=c; T[0,1]=-s; T[1,0]=s; T[1,1]=c
    return T

# 用 RANSAC 找地板平面
plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print("偵測到的地板平面方程:", plane_model)

# 計算每個點到地板的距離
pts = np.asarray(pcd.points)
dist = (pts @ np.array([a,b,c])) + d

# 保留距離在 [0.2m, 1.6m] 之間的點（刪掉地板和天花板，只留牆體）
mask = (dist > 0) & (dist < 1.5)
pcd = pcd.select_by_index(np.where(mask)[0])

# 更新點雲
pts = np.asarray(pcd.points)

# 投影：只保留 x,y
xy = pts[:, :2]


# 正規化座標（轉成正數，方便轉影像）
min_xy = xy.min(axis=0)
xy_norm = xy - min_xy

# 選網格解析度（例如 5cm = 0.05m）
resolution = 0.05
grid_xy = (xy_norm / resolution).astype(int)
origin = min_xy.copy()  # 地圖原點 (meters)：把左下角對齊到 (0,0) 前的平移量

map_w_m = (grid_xy.max(axis=0)[0] + 1) * resolution
map_h_m = (grid_xy.max(axis=0)[1] + 1) * resolution

# 建立空白地圖（grid 寬高，以格數表示）
w, h = grid_xy.max(axis=0) + 1
grid = np.ones((h, w), dtype=np.uint8) * 255  # 白色=可走

# 將點雲標記為障礙 (黑色)
for x,y in grid_xy:
    grid[y, x] = 0

# 讀取相機/Tag 設定（供後續整合使用）
K, dist = load_camera_yaml(CAMERA_YAML)
TAG_SIZE_M, TAGS_MAP = load_tag_db(TAG_CFG)
AT_DETECTOR = Detector(families=APRILTAG_FAMILY)

# 在地圖視覺化上疊畫已知 Tag 位置（藍色），不改變 grid 值
# 這只是標記地標，方便你對齊、檢查；不會影響A*
TAG_OVERLAY = np.stack([grid.copy() for _ in range(3)], axis=-1)
for tid, t in TAGS_MAP.items():
    tx, ty = float(t.get("x", 0.0)), float(t.get("y", 0.0))
    ix = int(np.floor((tx - origin[0]) / resolution))
    iy = int(np.floor((ty - origin[1]) / resolution))
    if 0 <= ix < w and 0 <= iy < h:
        TAG_OVERLAY[max(0,iy-1):min(h,iy+2), max(0,ix-1):min(w,ix+2)] = [0, 0, 255]  # 藍色小框 3x3

# 存成圖
plt.imshow(grid, cmap="gray")
plt.gca().invert_yaxis()  # Y 軸方向對齊常規影像
# ===== 使用者輸入座標的座標系轉換 =====

def to_world_meters(pt_m, origin, res, w, h):
    """把使用者輸入的公尺座標轉成世界座標：
    - COORD_MODE == 'map': 以地圖左下角(0,0)為原點 → 轉回 world: (x+origin_x, y+origin_y)
    - COORD_MODE == 'world': 不轉換
    """
    if COORD_MODE == 'map':
        return (pt_m[0] + origin[0], pt_m[1] + origin[1])
    return pt_m

# ===== A* 規劃輔助：座標轉換（meters <-> cells） =====

def meters_to_cells(x, y, origin, res):
    """(公尺) -> (格子索引)
    以左下角原點 origin 為參考，回傳 (ix, iy)；注意 grid[y, x] 的索引順序。
    """
    ix = int(np.floor((x - origin[0]) / res))
    iy = int(np.floor((y - origin[1]) / res))
    return ix, iy


def cells_to_meters(ix, iy, origin, res):
    """(格子索引) -> (公尺)；取網格中心點以降低量化誤差。"""
    x = (ix + 0.5) * res + origin[0]
    y = (iy + 0.5) * res + origin[1]
    return x, y


# ===== A* 本體 =====

def astar(grid, start, goal, allow_diag=False):
    """
    grid: 2D uint8, 0=障礙, 255=可走
    start, goal: (ix, iy)
    return: list of (ix, iy) from start->goal (含兩端)；找不到則返回 []
    """
    h, w = grid.shape
    if allow_diag:
        neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        step_cost = lambda dx,dy: 1.4142 if dx!=0 and dy!=0 else 1.0
    else:
        neighbors = [(1,0),(-1,0),(0,1),(0,-1)]
        step_cost = lambda dx,dy: 1.0

    def in_bounds(ix, iy):
        return 0 <= ix < w and 0 <= iy < h

    def is_free(ix, iy):
        return grid[iy, ix] != 0

    def heuristic(a, b):
        ax, ay = a; bx, by = b
        if allow_diag:
            # 八連通的一致啟發式（Octile 距離）
            dx, dy = abs(ax-bx), abs(ay-by)
            return (dx + dy) + (1.4142 - 2) * min(dx, dy)
        else:
            # 四連通的曼哈頓距離
            return abs(ax-bx) + abs(ay-by)

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came = {start: None}
    gscore = {start: 0.0}

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            # 回溯
            path = [cur]
            while came[cur] is not None:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        cx, cy = cur
        for dx, dy in neighbors:
            nx, ny = cx+dx, cy+dy
            if not in_bounds(nx, ny) or not is_free(nx, ny):
                continue
            tentative = gscore[cur] + step_cost(dx, dy)
            if (nx, ny) not in gscore or tentative < gscore[(nx, ny)]:
                gscore[(nx, ny)] = tentative
                f = tentative + heuristic((nx, ny), goal)
                heapq.heappush(open_heap, (f, (nx, ny)))
                came[(nx, ny)] = cur
    return []


# ===== 路徑簡化（只保留轉角） =====

def simplify_cells_path(path_cells):
    if len(path_cells) <= 2:
        return path_cells
    simp = [path_cells[0]]
    prev_dx = prev_dy = None
    for (x1,y1),(x2,y2) in zip(path_cells, path_cells[1:]):
        dx, dy = x2 - x1, y2 - y1
        # 正規化方向（避免 2,0 或 -2,0 這種跳格）
        dx = 0 if dx==0 else (1 if dx>0 else -1)
        dy = 0 if dy==0 else (1 if dy>0 else -1)
        if (dx, dy) != (prev_dx, prev_dy):
            simp.append((x2, y2))
            prev_dx, prev_dy = dx, dy
    if simp[-1] != path_cells[-1]:
        simp.append(path_cells[-1])
    return simp


# ===== 使用者：在這裡設定起點/終點（公尺） =====
# 若 COORD_MODE='map' → 以地圖左下角為 (0,0)；
# 若 COORD_MODE='world' → 以 PLY 世界座標為準（可能含負值）。
start_m_user = (1.0, 2.0)
goal_m_user  = (3.5, 4.0)

# 轉成世界座標（再轉格子）
start_m = to_world_meters(start_m_user, origin, resolution, w, h)
goal_m  = to_world_meters(goal_m_user,  origin, resolution, w, h)

# 轉成格子索引
start_ij = meters_to_cells(*start_m, origin, resolution)
goal_ij  = meters_to_cells(*goal_m,  origin, resolution)

# 邊界與可行性檢查
sx, sy = start_ij; gx, gy = goal_ij
sx = np.clip(sx, 0, w-1); sy = np.clip(sy, 0, h-1)
gx = np.clip(gx, 0, w-1); gy = np.clip(gy, 0, h-1)
start_ij = (int(sx), int(sy)); goal_ij = (int(gx), int(gy))

# 顯示座標（cells + meters in map/world frames）
start_m_map = (start_m[0] - origin[0], start_m[1] - origin[1])
goal_m_map  = (goal_m[0]  - origin[0], goal_m[1]  - origin[1])
print(f"Start (cells): {start_ij}, meters(world): {cells_to_meters(*start_ij, origin, resolution)}, meters(map): {start_m_map}")
print(f"Goal  (cells): {goal_ij}, meters(world): {cells_to_meters(*goal_ij, origin, resolution)}, meters(map): {goal_m_map}")

# --- Debug: 無論是否能規劃成功，都先輸出一張標示起點/終點的圖 ---
# 轉為RGB以便標示顏色
grid_points = np.stack([grid.copy() for _ in range(3)], axis=-1)
# 起點與終點標示（若落在邊界會自動裁切）
if 0 <= start_ij[0] < w and 0 <= start_ij[1] < h:
    grid_points[start_ij[1], start_ij[0]] = [255, 0, 0]   # 紅色 = 起點
if 0 <= goal_ij[0] < w and 0 <= goal_ij[1] < h:
    grid_points[goal_ij[1], goal_ij[0]] = [255, 255, 0]   # 黃色 = 終點

# 以公尺為座標軸輸出（更直覺）
if COORD_MODE == 'map':
    extent = [0, w*resolution, 0, h*resolution]
else:
    extent = [origin[0], origin[0] + w*resolution, origin[1], origin[1] + h*resolution]

# 疊上 tag 標記圖層，再畫 start/goal
points_vis = TAG_OVERLAY.copy()
# 將 points(紅/黃)覆蓋到 overlay 上
mask_pts = (grid_points[...,0]!=TAG_OVERLAY[...,0]) | (grid_points[...,1]!=TAG_OVERLAY[...,1]) | (grid_points[...,2]!=TAG_OVERLAY[...,2])
points_vis[mask_pts] = grid_points[mask_pts]
plt.figure(); plt.imshow(points_vis, origin='lower', extent=extent)
plt.title("Start/Goal + AprilTags (meters)")
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.savefig("map_points.png", dpi=300, bbox_inches='tight')


if grid[start_ij[1], start_ij[0]] == 0:
    raise ValueError(f"Start {start_ij} 落在障礙上，請調整座標")
if grid[goal_ij[1],  goal_ij[0]]  == 0:
    raise ValueError(f"Goal  {goal_ij} 落在障礙上，請調整座標")

# 規劃
path_cells = astar(grid, start_ij, goal_ij, allow_diag=True)
if not path_cells:
    # 進一步偵錯：從起點做一次 flood-fill，看看可達區域長什麼樣子
    h_, w_ = grid.shape
    vis = np.zeros_like(grid, dtype=np.uint8)
    q = deque()
    if 0 <= start_ij[0] < w_ and 0 <= start_ij[1] < h_ and grid[start_ij[1], start_ij[0]] != 0:
        q.append(start_ij)
        vis[start_ij[1], start_ij[0]] = 1
    nbrs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    while q:
        cx, cy = q.popleft()
        for dx, dy in nbrs:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < w_ and 0 <= ny < h_ and vis[ny, nx] == 0 and grid[ny, nx] != 0:
                vis[ny, nx] = 1
                q.append((nx, ny))
    # 製作可達區域疊圖
    reach_vis = np.stack([grid.copy() for _ in range(3)], axis=-1)  # 轉成RGB
    reach_vis[..., 1] = np.maximum(reach_vis[..., 1], (vis*255).astype(np.uint8))  # 綠色可達區
    if 0 <= start_ij[0] < w_ and 0 <= start_ij[1] < h_:
        reach_vis[start_ij[1], start_ij[0]] = [255, 0, 0]     # 紅=起點
    if 0 <= goal_ij[0] < w_ and 0 <= goal_ij[1] < h_:
        reach_vis[goal_ij[1], goal_ij[0]] = [255, 255, 0]     # 黃=終點

    if COORD_MODE == 'map':
        extent = [0, w*resolution, 0, h*resolution]
    else:
        extent = [origin[0], origin[0] + w*resolution, origin[1], origin[1] + h*resolution]
    plt.figure(); plt.imshow(reach_vis, origin='lower', extent=extent)
    plt.title("Reachable region from start (green) – meters")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.savefig("map_reachability.png", dpi=300, bbox_inches='tight')

    print("[WARN] A* 找不到可行路徑。已輸出:")
    print("  - map_points.png (起點/終點位置)")
    print("  - map_reachability.png (起點可達區域)")
    import sys; sys.exit(2)

# 簡化並轉回公尺座標
path_cells_s = simplify_cells_path(path_cells)
path_meters = [cells_to_meters(ix, iy, origin, resolution) for ix, iy in path_cells_s]
print("規劃節點數 (格):", len(path_cells))
print("簡化節點數 (轉角):", len(path_cells_s))
print("Waypoints (meters):")
for p in path_meters:
    print(f"  {p[0]:.3f}, {p[1]:.3f}")

# 繪製路徑疊圖
grid_vis = np.stack([grid.copy() for _ in range(3)], axis=-1)
for ix, iy in path_cells:
    grid_vis[iy, ix] = [128, 128, 128]  # 灰色路徑
# 起點與終點標示
grid_vis[start_ij[1], start_ij[0]] = [255, 0, 0]    # 紅色 = 起點
grid_vis[goal_ij[1],  goal_ij[0]]  = [255, 255, 0]  # 黃色 = 終點

# === AprilTag 位姿估計（供後續串接相機用；此檔先不開相機，只提供函式） ===

def estimate_pose_from_frame(frame_bgr, K, dist, tag_size_m, tags_map, detector):
    """輸入一張 BGR 影像，回傳一組或多組 T_map_cam 估計（若識別到已知ID）。
    回傳: list of (tag_id, T_map_cam)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    dets = detector.detect(gray, estimate_tag_pose=True,
                           camera_params=[K[0,0], K[1,1], K[0,2], K[1,2]],
                           tag_size=tag_size_m)
    results = []
    for d in dets:
        tid = int(d.tag_id)
        if tid not in tags_map:
            continue
        R_cam_tag, t_cam_tag = d.pose_R, d.pose_t.reshape(3)
        T_cam_tag = rt_to_T(R_cam_tag, t_cam_tag)
        t = tags_map[tid]
        T_map_tag = np.eye(4, dtype=np.float32)
        T_map_tag[:3,3] = [float(t.get("x",0.0)), float(t.get("y",0.0)), float(t.get("z",1.4))]
        T_map_tag = T_map_tag @ yaw_T(np.deg2rad(float(t.get("yaw_deg", 0.0))))
        T_map_cam = T_map_tag @ np.linalg.inv(T_cam_tag)
        results.append((tid, T_map_cam))
    return results

# 存成圖（原始地圖 + 含路徑地圖）
if COORD_MODE == 'map':
    extent = [0, w*resolution, 0, h*resolution]
else:
    extent = [origin[0], origin[0] + w*resolution, origin[1], origin[1] + h*resolution]

plt.figure(); plt.imshow(grid, cmap="gray", origin='lower', extent=extent); plt.title("Occupancy Grid (meters)")
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.savefig("map.png", dpi=300, bbox_inches='tight')

# 把 Tag 藍色疊到路徑圖上
path_vis = grid_vis.copy()
blue_mask = (TAG_OVERLAY[...,2] > TAG_OVERLAY[...,0]) & (TAG_OVERLAY[...,2] > TAG_OVERLAY[...,1])
path_vis[blue_mask] = TAG_OVERLAY[blue_mask]
plt.figure(); plt.imshow(path_vis, origin='lower', extent=extent); plt.title("Grid with A* Path + AprilTags (meters)")
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.savefig("map_with_path.png", dpi=300, bbox_inches='tight')
plt.show()
