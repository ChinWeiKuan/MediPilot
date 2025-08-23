import heapq, numpy as np
def astar(grid, start, goal):
    H,W = grid.shape
    def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    nbs = [(1,0),(-1,0),(0,1),(0,-1)]
    g = {start:0}; parent = {start:None}
    pq=[(h(start,goal), start)]
    closed=set()
    while pq:
        _, u = heapq.heappop(pq)
        if u in closed: continue
        if u==goal: break
        closed.add(u)
        for dx,dy in nbs:
            v=(u[0]+dx,u[1]+dy)
            if not (0<=v[0]<H and 0<=v[1]<W): continue
            if grid[v]==1: continue  # 障礙
            ng=g[u]+1
            if ng<g.get(v,1e9):
                g[v]=ng; parent[v]=u
                heapq.heappush(pq,(ng+h(v,goal),v))
    if goal not in parent: return None
    path=[]; cur=goal
    while cur is not None: path.append(cur); cur=parent[cur]
    return path[::-1]