import numpy as np, cv2
def grid_to_xy(path_idx, res, origin):
    # grid 索引(i,j) → 世界座標 (x,y)
    pts=[]
    for i,j in path_idx:
        x = origin[0] + (j+0.5)*res
        y = origin[1] + ( (path_idx.shape[0]-i-0.5)*res )  # 影像座標轉 y
        pts.append([x,y])
    return np.array(pts, dtype=float)
def simplify_path(path_xy, eps=0.02):
    # RDP 簡化（可選）
    if len(path_xy)<=2: return path_xy
    from math import hypot
    def dp(pts):
        if len(pts)<=2: return pts
        a,b=pts[0],pts[-1]
        dmax,k=0,0
        ax,ay=a; bx,by=b
        for i in range(1,len(pts)-1):
            px,py=pts[i]
            den = ((bx-ax)**2+(by-ay)**2)**0.5 + 1e-9
            d = abs((by-ay)*px - (bx-ax)*py + bx*ay - by*ax)/den
            if d>dmax: dmax,k=d,i
        if dmax>eps:
            left=dp(pts[:k+1]); right=dp(pts[k:])
            return left[:-1]+right
        else:
            return [a,b]
    return np.array(dp(list(map(tuple,path_xy))))