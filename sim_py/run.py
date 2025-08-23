import numpy as np, matplotlib.pyplot as plt
from map_io import load_map
from planner import astar
from controller import follow_path_step
from simulate import grid_to_xy, simplify_path

# 1) 載入地圖
grid, res, origin = load_map('sim_py/maps/hospital_map.yaml')
H,W = grid.shape

# 2) 定義 A/B（用像素/格網座標；你也可改用公尺座標再轉）
start_ij = (H-20, 20)     # 例：左下角附近
goal_ij  = (H-60, W-40)   # 例：右上角附近

# 3) 規劃
path_idx = astar(grid, start_ij, goal_ij)
assert path_idx is not None, "找不到從 A 到 B 的路徑（地圖可能被牆堵死）"

# 4) 轉公尺、平滑
path_xy  = grid_to_xy(np.array(path_idx), res, origin)
path_xy  = simplify_path(path_xy, eps=0.02)

# 5) 初始位姿（面向路徑第一段）
x,y = path_xy[0]
yaw = np.arctan2(path_xy[min(1,len(path_xy)-1),1]-y, path_xy[min(1,len(path_xy)-1),0]-x)

# 6) 視覺化動畫
plt.ion()
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(1-grid, cmap='gray', origin='lower', extent=[origin[0], origin[0]+W*res, origin[1], origin[1]+H*res])
ax.plot(path_xy[:,0], path_xy[:,1], linewidth=2)
robot_dot, = ax.plot([x],[y], marker='o', markersize=6)
ax.set_aspect('equal', 'box'); ax.set_title('Python Nav Sandbox')

done=False
while not done:
    x,y,yaw,done = follow_path_step(x,y,yaw,path_xy, dt=0.05)
    robot_dot.set_data([x],[y])
    plt.pause(0.01)

plt.ioff(); plt.show()