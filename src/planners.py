# -*- coding: utf-8 -*-
"""
Yol Planlayıcılar
=================
1) DynamicAStar : Değişiklik tetiklemeli Dynamic A* — occupancy grid değiştiğinde
   etkilenen bölge kontrol edilir ve global yol yeniden planlanır.
   Heuristic: Öklidyen mesafe (admissible).
2) APFPlanner   : Yapay Potansiyel Alan (karşılaştırma taban çizgisi).
"""

import heapq
import numpy as np


# ----------------------------------------------------------------------------
# A* / Dynamic A*
# ----------------------------------------------------------------------------

class DynamicAStar:
    """Occupancy grid üzerinde A*; harita değişince otomatik yeniden planlama.

    'Dynamic A*' davranışı: robot ilerlerken haritada (yeni algılanan engel)
    değişiklik olursa ve bu değişiklik mevcut yolun üzerine düşerse yol
    yeniden hesaplanır. 8-komşuluk, Öklidyen heuristic.
    """

    SQRT2 = np.sqrt(2.0)

    def __init__(self, resolution, robot_radius=0.25):
        self.res = resolution
        # Robot yarıçapı kadar şişirme (kapı geçitlerini kapatmamak için +marj yok;
        # lokal güvenlik payı DWA'nın clearance kontrolündedir)
        self.inflate = int(np.ceil(robot_radius / resolution))
        self.path = None          # dünya koordinatlarında (N,2)
        self.path_cells = None
        self.last_grid = None
        self.replan_count = 0

    # ------------------------------------------------------------------
    def _inflate_grid(self, grid):
        """Engelleri robot yarıçapı kadar şişir (configuration space). Saf NumPy."""
        g = grid.astype(bool)
        out = g.copy()
        r = self.inflate
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.zeros_like(g)
                ys = slice(max(0, dy), g.shape[0] + min(0, dy))
                xs = slice(max(0, dx), g.shape[1] + min(0, dx))
                ys_src = slice(max(0, -dy), g.shape[0] + min(0, -dy))
                xs_src = slice(max(0, -dx), g.shape[1] + min(0, -dx))
                shifted[ys, xs] = g[ys_src, xs_src]
                out |= shifted
        return out

    def plan(self, grid, start_xy, goal_xy):
        """A* araması. start/goal dünya koordinatları. Yol (N,2) veya None."""
        ny, nx = grid.shape
        inf_grid = self._inflate_grid(grid)

        def to_cell(p):
            return (int(np.clip(p[0] / self.res, 0, nx - 1)),
                    int(np.clip(p[1] / self.res, 0, ny - 1)))

        start = to_cell(start_xy)
        goal = to_cell(goal_xy)

        # Başlangıç/hedef şişirilmiş engel içindeyse en yakın boş hücreye kaydır
        start = self._nearest_free(inf_grid, start)
        goal = self._nearest_free(inf_grid, goal)
        if start is None or goal is None:
            return None

        gx, gy = goal
        open_heap = [(0.0, start)]
        g_cost = {start: 0.0}
        parent = {start: None}
        closed = set()
        neighbors = [(-1, -1, self.SQRT2), (-1, 0, 1), (-1, 1, self.SQRT2),
                     (0, -1, 1), (0, 1, 1),
                     (1, -1, self.SQRT2), (1, 0, 1), (1, 1, self.SQRT2)]

        found = False
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal:
                found = True
                break
            cx, cy = cur
            for ddx, ddy, cost in neighbors:
                nxt = (cx + ddx, cy + ddy)
                if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                    continue
                if inf_grid[nxt[1], nxt[0]]:
                    continue
                # Diyagonal geçişte köşe kesme engelle
                if ddx != 0 and ddy != 0:
                    if inf_grid[cy, cx + ddx] or inf_grid[cy + ddy, cx]:
                        continue
                ng = g_cost[cur] + cost
                if nxt not in g_cost or ng < g_cost[nxt]:
                    g_cost[nxt] = ng
                    h = np.hypot(nxt[0] - gx, nxt[1] - gy)   # Öklidyen heuristic
                    heapq.heappush(open_heap, (ng + h, nxt))
                    parent[nxt] = cur

        if not found:
            return None

        cells = []
        cur = goal
        while cur is not None:
            cells.append(cur)
            cur = parent[cur]
        cells.reverse()
        self.path_cells = cells
        self.last_grid = inf_grid
        path = np.array([[(c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res] for c in cells])
        self.path = self._smooth(path)
        return self.path

    def _nearest_free(self, inf_grid, cell, max_r=15):
        if not inf_grid[cell[1], cell[0]]:
            return cell
        ny, nx = inf_grid.shape
        for r in range(1, max_r):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    c = (cell[0] + dx, cell[1] + dy)
                    if 0 <= c[0] < nx and 0 <= c[1] < ny and not inf_grid[c[1], c[0]]:
                        return c
        return None

    @staticmethod
    def _smooth(path, weight_data=0.5, weight_smooth=0.25, iterations=40):
        """Gradyan tabanlı yol yumuşatma."""
        if len(path) < 3:
            return path
        new = path.copy()
        for _ in range(iterations):
            for i in range(1, len(path) - 1):
                new[i] += weight_data * (path[i] - new[i]) + \
                          weight_smooth * (new[i - 1] + new[i + 1] - 2 * new[i])
        return new

    # ------------------------------------------------------------------
    def needs_replan(self, grid, robot_xy, lookahead=12):
        """Haritadaki değişiklik mevcut yolun önünü kapatıyor mu?"""
        if self.path_cells is None:
            return True
        inf_grid = self._inflate_grid(grid)
        # Robotun yol üzerindeki en yakın noktasından ileriye bak
        pts = np.array(self.path_cells, dtype=float)
        rp = np.array([robot_xy[0] / self.res, robot_xy[1] / self.res])
        d = np.linalg.norm(pts - rp, axis=1)
        i0 = int(np.argmin(d))
        for (cx, cy) in self.path_cells[i0:i0 + lookahead]:
            if inf_grid[cy, cx]:
                return True
        return False

    def replan(self, grid, start_xy, goal_xy):
        self.replan_count += 1
        return self.plan(grid, start_xy, goal_xy)


# ----------------------------------------------------------------------------
# Yapay Potansiyel Alan (APF) — taban çizgisi yöntem
# ----------------------------------------------------------------------------

class APFPlanner:
    """Klasik yapay potansiyel alan: çekici hedef + itici engeller.

    Lokal minimuma takılma sorunu bilinen zayıflığıdır; karşılaştırma için
    geleneksel taban çizgisi olarak kullanılır.
    """

    def __init__(self, k_att=1.0, k_rep=0.8, d0=1.2):
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0

    def command(self, robot, goal, ranges, angles):
        """LIDAR taramasından itme, hedeften çekme kuvveti ile (v, w) üret."""
        # Çekici kuvvet
        gx, gy = goal[0] - robot.x, goal[1] - robot.y
        d_goal = np.hypot(gx, gy)
        f_att = self.k_att * np.array([gx, gy]) / max(d_goal, 1e-6)

        # İtici kuvvet (LIDAR noktalarından)
        f_rep = np.zeros(2)
        for r, a in zip(ranges, angles):
            if r < self.d0:
                mag = self.k_rep * (1.0 / r - 1.0 / self.d0) / (r * r)
                f_rep -= mag * np.array([np.cos(a), np.sin(a)])

        f = f_att + f_rep
        target_theta = np.arctan2(f[1], f[0])
        dth = (target_theta - robot.theta + np.pi) % (2 * np.pi) - np.pi
        w = np.clip(2.0 * dth, -robot.MAX_W, robot.MAX_W)
        v = robot.MAX_V * max(0.0, np.cos(dth)) * min(1.0, d_goal)
        # Engel çok yakınsa yavaşla
        if np.min(ranges) < 0.4:
            v *= 0.3
        return v, w
