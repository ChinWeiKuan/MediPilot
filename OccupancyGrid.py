import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque
import cv2

# === Config ===
# COORD_MODE: 'map' → 以「地圖左下角」為 (0,0)（全部正數，直覺給公尺）
#              'world' → 使用 PLY 原始世界座標（可能含負數）
COORD_MODE = 'map'
# 是否允許斜向（45°）移動；若為 False → 只允許 4-連通（僅 90°/180° 轉向）
ALLOW_DIAG = False
TURN_COST = 0.6   # 轉彎成本（90°）；提高以減少轉彎次數
# 掉頭成本（180°）；通常比 90° 更高，鼓勵規劃避免掉頭
UTURN_COST = 1.2

# ---- Safety preferences (keep to the center of corridors) ----
# 1) Inflate obstacles by this radius so planner won't skim furniture edges
INFLATION_RADIUS_M = 0.0   # 不膨脹
# 2) Add a "repulsion" cost that grows near obstacles via distance transform
DISTFIELD_WEIGHT   = 0.0   # 不施加靠中間的軟代價（純粹最少轉彎傾向）
DISTFIELD_EPS      = 1e-3      # avoid divide-by-zero in cost

# 讀取點雲
pcd = o3d.io.read_point_cloud("stancode_0823.ply")
pts = np.asarray(pcd.points)   # Nx3, [x,y,z]
print("點數:", pts.shape)

print("bbox(x,y,z) min:", pts.min(0), "max:", pts.max(0))
print("range meters? (max-min):", (pts.max(0)-pts.min(0)))
print("norm length:", np.linalg.norm(pts.max(0)-pts.min(0)))

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
mask = (dist > 0) & (dist < 4)
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

# Cache original obstacle mask (before inflation)
obs_orig = (grid == 0).astype(np.uint8)

# --- Obstacle inflation (safety margin) ---
r_cells = int(np.ceil(INFLATION_RADIUS_M / resolution))
if r_cells > 0:
    kernel = np.ones((2*r_cells + 1, 2*r_cells + 1), np.uint8)
    obs_infl = cv2.dilate(obs_orig, kernel, iterations=1)
    grid[obs_infl == 1] = 0

    # Compute inflation-only ring: inflated minus original
    ring = (obs_infl == 1) & (obs_orig == 0)

    # Build an RGB overlay where:
    #   - white = free space
    #   - black = original obstacles
    #   - orange = added safety buffer (inflation ring)
    overlay = np.ones((grid.shape[0], grid.shape[1], 3), dtype=np.uint8) * 255
    overlay[obs_infl == 1] = [255, 165, 0]  # orange for inflated area (will be overwritten by orig to black)
    overlay[obs_orig == 1] = [0, 0, 0]      # black for original obstacles

# --- Soft repulsion cost (distance field) ---
# For non-zero (free) pixels, distanceTransform returns distance to nearest zero (obstacle)
free_u8 = (grid != 0).astype(np.uint8)
dist_pix = cv2.distanceTransform(free_u8, cv2.DIST_L2, 3)  # in cells
dist_m   = dist_pix * float(resolution)
# Higher cost near obstacles; ~0 cost in the middle of wide corridors
cell_cost = DISTFIELD_WEIGHT * (1.0 / (dist_m + DISTFIELD_EPS))
# Obstacles won't be traversed anyway, but keep their cost very high for completeness
cell_cost[grid == 0] = 1e6

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

def astar(grid, start, goal, allow_diag=False, turn_cost=0.0, cell_cost=None):
    """
    A* 加入「轉彎成本」：
    - 狀態 = (x, y, dir_idx)；dir_idx ∈ {0:E, 1:N, 2:W, 3:S, None(起點)}
    - 每步成本 = 1（若 allow_diag=False）或含對角成本；若 dir 改變，另外加 turn_cost
    回傳 cell 序列（不含 dir），若找不到回傳 []
    """
    h, w = grid.shape
    if allow_diag:
        dirs = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        diag = {(1,1), (1,-1), (-1,1), (-1,-1)}
        def step_cost(dx, dy): return 1.4142 if (dx,dy) in diag else 1.0
    else:
        dirs = [(1,0),(0,1),(-1,0),(0,-1)]  # E, N, W, S
        def step_cost(dx, dy): return 1.0

    dir_to_idx = {d:i for i,d in enumerate(dirs)}

    def in_bounds(ix, iy): return 0 <= ix < w and 0 <= iy < h
    def is_free(ix, iy):  return grid[iy, ix] != 0

    def heuristic(a, b):
        ax, ay = a; bx, by = b
        if allow_diag:
            dx, dy = abs(ax-bx), abs(ay-by)
            return (dx + dy) + (1.4142 - 2) * min(dx, dy)
        else:
            return abs(ax-bx) + abs(ay-by)

    import heapq
    open_heap = []
    # 起點 state：方向未知，用 None 表示
    start_state = (start[0], start[1], None)
    heapq.heappush(open_heap, (0.0, start_state))
    came = {start_state: None}
    gscore = {start_state: 0.0}

    goal_best_state = None
    goal_best_g = float("inf")

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        cx, cy, cur_dir = cur
        # 若已到達 goal（任意 dir），更新最優終點
        if (cx, cy) == goal and gscore[cur] < goal_best_g:
            goal_best_state = cur
            goal_best_g = gscore[cur]
            # 繼續從 heap 擷取也可以，但已有最佳 g 時可提前結束
            # break

        # 擴展鄰居
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny) or not is_free(nx, ny):
                continue
            step_c = step_cost(dx, dy)
            ndir = dir_to_idx[(dx, dy)]
            # 是否發生轉向
            extra = 0.0
            if cur_dir is not None and ndir != cur_dir:
                # 針對 4-連通：相差 2 表示 180° 掉頭，其餘為 90°
                if not allow_diag and abs(ndir - cur_dir) % 4 == 2:
                    extra = UTURN_COST
                else:
                    extra = turn_cost
            base = gscore[cur] + step_c + extra
            cc   = 0.0 if cell_cost is None else float(cell_cost[ny, nx])
            tentative = base + cc
            nstate = (nx, ny, ndir)
            if nstate not in gscore or tentative < gscore[nstate]:
                gscore[nstate] = tentative
                f = tentative + heuristic((nx, ny), goal)
                heapq.heappush(open_heap, (f, nstate))
                came[nstate] = cur

    if goal_best_state is None:
        return []

    # 回溯狀態路徑，抽出 (x,y)
    path_states = []
    s = goal_best_state
    while s is not None:
        path_states.append(s)
        s = came[s]
    path_states.reverse()
    path = []
    prev_xy = None
    for x, y, _ in path_states:
        if prev_xy != (x, y):
            path.append((x, y))
            prev_xy = (x, y)
    return path


# ===== 路徑簡化（只保留轉角） =====

def simplify_cells_path(path_cells):
    """
    將逐格路徑簡化為「每段直線的端點」清單。
    修正：遇到方向改變時，加入的是「上一段的最後一點 (x1,y1)」，而不是新方向的第一點，
    以避免連續端點之間出現斜向位移（如 (-1,1)）。
    """
    n = len(path_cells)
    if n <= 2:
        return path_cells[:]  # 0,1,2 點均不需簡化或已是最簡

    simp = [path_cells[0]]
    # 前一段方向
    def norm_step(dx, dy):
        dx = 0 if dx == 0 else (1 if dx > 0 else -1)
        dy = 0 if dy == 0 else (1 if dy > 0 else -1)
        return dx, dy

    prev_dx = prev_dy = None
    for (x1, y1), (x2, y2) in zip(path_cells, path_cells[1:]):
        dx, dy = norm_step(x2 - x1, y2 - y1)
        if (dx, dy) != (prev_dx, prev_dy):
            # 方向發生改變：把上一段的最後一點 (x1,y1) 當作轉角端點加入
            if prev_dx is not None:
                simp.append((x1, y1))
            prev_dx, prev_dy = dx, dy
    # 迴圈結束後，補上最終端點
    if simp[-1] != path_cells[-1]:
        simp.append(path_cells[-1])
    return simp


# ===== 使用者：在這裡設定起點/終點（公尺） =====
# 若 COORD_MODE='map' → 以地圖左下角為 (0,0)；
# 若 COORD_MODE='world' → 以 PLY 世界座標為準（可能含負值）。
start_m_user = (5.0, 2.0)
goal_m_user  = (2.65, 6.5)

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
plt.figure(); plt.imshow(grid_points, origin='lower', extent=extent)
plt.title("Start/Goal on Grid (meters)")
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.savefig("map_points.png", dpi=300, bbox_inches='tight')


if grid[start_ij[1], start_ij[0]] == 0:
    raise ValueError(f"Start {start_ij} 落在障礙上，請調整座標")
if grid[goal_ij[1],  goal_ij[0]]  == 0:
    raise ValueError(f"Goal  {goal_ij} 落在障礙上，請調整座標")

# 規劃
path_cells = astar(grid, start_ij, goal_ij, allow_diag=ALLOW_DIAG, turn_cost=TURN_COST, cell_cost=cell_cost)
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

# 將路徑轉成行動步驟（只會有 90°/180° 轉向，因為 ALLOW_DIAG=False）
def path_to_actions(path_cells_s, res):
    """
    將「簡化後的 cell 路徑端點」轉為行動步驟（直走/轉彎）。
    修正：每一段直線的距離應為相鄰端點的曼哈頓距離（以格為單位），
    而不是僅以段數計 1 格。此處會：
      1) 計算每段 axis-aligned 的長度（cells）
      2) 合併連續同方向的段
      3) 在方向改變時插入左/右/掉頭
    回傳：(actions, pretty_lines)
    """
    if len(path_cells_s) < 2:
        return [], []

    # 方向與方位角定義（4-連通）
    DIR2HEAD = {(1,0): 0, (0,1): 90, (-1,0): 180, (0,-1): 270}
    HEAD2NAME = {0: 'E', 90: 'N', 180: 'W', 270: 'S'}

    def norm_step(dx, dy):
        dx = 0 if dx == 0 else (1 if dx > 0 else -1)
        dy = 0 if dy == 0 else (1 if dy > 0 else -1)
        return dx, dy

    # 1) 由端點對，建立 (dir, length_cells) 列表
    segs = []  # [(dx_unit, dy_unit, length_cells)]
    for (x1, y1), (x2, y2) in zip(path_cells_s, path_cells_s[1:]):
        dx_c, dy_c = x2 - x1, y2 - y1
        # 必須是軸向
        if dx_c != 0 and dy_c != 0:
            raise ValueError(f"簡化後路徑出現非軸向段: ({x1},{y1})->({x2},{y2})")
        dx_u, dy_u = norm_step(dx_c, dy_c)
        length = abs(dx_c) + abs(dy_c)  # 由於軸向，L1=實際格數
        if length == 0:
            continue
        segs.append((dx_u, dy_u, length))

    # 2) 合併連續同方向
    merged = []
    for seg in segs:
        if not merged:
            merged.append(list(seg))  # [dx_u, dy_u, length]
        else:
            if (merged[-1][0], merged[-1][1]) == (seg[0], seg[1]):
                merged[-1][2] += seg[2]
            else:
                merged.append(list(seg))

    # 3) 轉為動作：直走(公尺) + 需要時的轉彎
    heading = DIR2HEAD[(merged[0][0], merged[0][1])]  # 初始朝向為第一段方向
    actions, pretty = [], []

    def flush_move(len_cells, heading):
        if len_cells <= 0:
            return
        meters = len_cells * res
        actions.append({"type": "move", "meters": meters, "dir": HEAD2NAME[heading]})
        pretty.append(f"直走 {meters:.2f} m（朝 {HEAD2NAME[heading]}）")

    # 逐段輸出
    flush_move(merged[0][2], heading)
    for dx_u, dy_u, length in merged[1:]:
        new_head = DIR2HEAD[(dx_u, dy_u)]
        delta = (new_head - heading) % 360
        if delta == 90:
            actions.append({"type": "turn", "dir": "left"})
            pretty.append("左轉 90°")
        elif delta == 270:
            actions.append({"type": "turn", "dir": "right"})
            pretty.append("右轉 90°")
        elif delta == 180:
            actions.append({"type": "turn", "dir": "u-turn"})
            pretty.append("掉頭 180°")
        # 同方向 delta==0 不加 turn
        heading = new_head
        flush_move(length, heading)

    return actions, pretty

# 生成並輸出行動步驟，同時將列表寫入一個 JSON 檔以利後續機器人控制：
actions, pretty = path_to_actions(path_cells_s, resolution)
total_len_m = sum(a["meters"] for a in actions if a["type"] == "move")
print(f"Total planned path length ≈ {total_len_m:.2f} m")
print("\nAction steps:")
for line in pretty:
    print(" -", line)

# 另存一份 JSON（含原始 path 與 actions）
try:
    import json
    out_obj = {
        "resolution_m": float(resolution),
        "origin_world_xy": [float(origin[0]), float(origin[1])],
        "start_cell": [int(start_ij[0]), int(start_ij[1])],
        "goal_cell": [int(goal_ij[0]), int(goal_ij[1])],
        "path_cells": [[int(ix), int(iy)] for ix,iy in path_cells],
        "path_cells_simplified": [[int(ix), int(iy)] for ix,iy in path_cells_s],
        "actions": actions
    }
    with open("planned_actions.json", "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print('已輸出行動步驟 → "planned_actions.json"')
except Exception as e:
    print("[WARN] 無法寫出 planned_actions.json:", e)

# 繪製路徑疊圖
grid_vis = np.stack([grid.copy() for _ in range(3)], axis=-1)
# Lightly tint the inflation ring in the path figure as well
if r_cells > 0:
    ring_rgb = grid_vis.copy()
    tint = np.array([255, 165, 0], dtype=np.uint8)
    grid_vis[ring] = (0.5*grid_vis[ring] + 0.5*tint).astype(np.uint8)
for ix, iy in path_cells:
    grid_vis[iy, ix] = [128, 128, 128]  # 灰色路徑
# 起點與終點標示
grid_vis[start_ij[1], start_ij[0]] = [255, 0, 0]    # 紅色 = 起點
grid_vis[goal_ij[1],  goal_ij[0]]  = [255, 255, 0]  # 黃色 = 終點

# 存成圖（原始地圖 + 含路徑地圖）
if COORD_MODE == 'map':
    extent = [0, w*resolution, 0, h*resolution]
else:
    extent = [origin[0], origin[0] + w*resolution, origin[1], origin[1] + h*resolution]

plt.figure(); plt.imshow(grid, cmap="gray", origin='lower', extent=extent); plt.title("Occupancy Grid (meters)")
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.savefig("map.png", dpi=300, bbox_inches='tight')

plt.figure(); plt.imshow(grid_vis, origin='lower', extent=extent); plt.title("Grid with A* Path (meters)")
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.savefig("map_with_path.png", dpi=300, bbox_inches='tight')

# Save inflation overlay (black=original obstacle, orange=inflated-only)
if r_cells > 0:
    plt.figure(); plt.imshow(overlay, origin='lower', extent=extent)
    plt.title("Obstacle inflation overlay (black=original, orange=inflated-only)")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.savefig("map_inflation_overlay.png", dpi=300, bbox_inches='tight')

# Visualize distance-field cost (for tuning)
try:
    plt.figure(); plt.imshow(cell_cost, origin='lower', extent=extent)
    plt.title("Distance-field cost (higher = closer to obstacle)")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.colorbar(label="added cost per step")
    plt.savefig("map_costfield.png", dpi=300, bbox_inches='tight')
except Exception as _e:
    print("[WARN] costfield viz skipped:", _e)

plt.show()
