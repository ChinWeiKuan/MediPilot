import numpy as np
def follow_path_step(x, y, yaw, path_xy, v_max=0.4, w_max=1.2, dt=0.05, lookahead=0.4):
    # 找最近的前視點
    dists = np.hypot(path_xy[:,0]-x, path_xy[:,1]-y)
    idx = dists.argmin()
    # 向前看一段距離
    while idx < len(path_xy)-1 and np.linalg.norm(path_xy[idx]-[x,y]) < lookahead:
        idx += 1
    target = path_xy[min(idx, len(path_xy)-1)]
    # Pure Pursuit 控制
    dx, dy = target[0]-x, target[1]-y
    th = np.arctan2(dy, dx)
    ang_err = np.arctan2(np.sin(th - yaw), np.cos(th - yaw))
    v = np.clip(0.3 + 0.7*np.cos(ang_err), 0.0, v_max)  # 轉彎降速
    w = np.clip(2.0*ang_err, -w_max, w_max)
    # 單軌車模型
    x += v*np.cos(yaw)*dt
    y += v*np.sin(yaw)*dt
    yaw = (yaw + w*dt + np.pi)%(2*np.pi) - np.pi
    done = np.linalg.norm(target-[x,y])<0.1 and idx>=len(path_xy)-1
    return x,y,yaw,done