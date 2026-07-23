# -*- coding: utf-8 -*-
"""
Dinamik Ofis Ortamı Simülasyonu
================================
2B ofis ortamı: duvarlar, mobilyalar (statik engeller), insanlar (dinamik engeller),
diferansiyel sürüşlü mobil robot ve LIDAR sensör simülasyonu.

TÜBİTAK 2209-A: Dinamik Ofis Ortamlarında Mobil Robotlar İçin
Hibrit Uyarlanabilir Yapay Zeka Tabanlı Optimum Yol Planlaması
"""

import numpy as np

# ----------------------------------------------------------------------------
# Yardımcı geometri fonksiyonları
# ----------------------------------------------------------------------------

def rect_contains(rect, x, y, margin=0.0):
    """rect = (x0, y0, w, h). Nokta dikdörtgenin içinde mi (margin ile genişletilmiş)?"""
    x0, y0, w, h = rect
    return (x0 - margin <= x <= x0 + w + margin) and (y0 - margin <= y <= y0 + h + margin)


def ray_rect_intersection(ox, oy, dx, dy, rect):
    """Işın (ox,oy)+t*(dx,dy) ile dikdörtgen kesişimi. En yakın t (>=0) veya None."""
    x0, y0, w, h = rect
    tmin, tmax = 0.0, np.inf
    for (o, d, lo, hi) in ((ox, dx, x0, x0 + w), (oy, dy, y0, y0 + h)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin if tmin >= 0 else None


def ray_circle_intersection(ox, oy, dx, dy, cx, cy, r):
    """Işın ile daire kesişimi. En yakın t (>=0) veya None."""
    fx, fy = ox - cx, oy - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    if t1 >= 0:
        return t1
    if t2 >= 0:
        return t2
    return None


# ----------------------------------------------------------------------------
# Dinamik engel: İnsan
# ----------------------------------------------------------------------------

class Human:
    """Ofiste yol noktaları arasında yürüyen insan (dinamik engel)."""

    RADIUS = 0.30  # m

    def __init__(self, waypoints, speed=0.6, rng=None):
        self.waypoints = [np.array(w, dtype=float) for w in waypoints]
        self.speed = speed
        self.rng = rng or np.random.default_rng()
        self.idx = 0
        self.pos = self.waypoints[0].copy()
        self.vel = np.zeros(2)
        self.pause = 0.0

    def step(self, dt, robot_pos=None):
        if self.pause > 0:
            self.pause -= dt
            self.vel[:] = 0
            return
        target = self.waypoints[(self.idx + 1) % len(self.waypoints)]
        d = target - self.pos
        dist = np.linalg.norm(d)
        if dist < 0.15:
            self.idx = (self.idx + 1) % len(self.waypoints)
            # İnsanlar bazen durup bekler (telefon, sohbet...)
            if self.rng.random() < 0.3:
                self.pause = self.rng.uniform(0.5, 2.0)
            return
        direction = d / dist
        # Hafif rastgele sapma -> gerçekçi insan hareketi
        noise = self.rng.normal(0, 0.08, 2)
        v = direction * self.speed + noise
        # Sosyal kaçınma: insanlar robota çok yaklaşınca yavaşlar ve yana kayar
        if robot_pos is not None:
            dr = self.pos - np.asarray(robot_pos)
            d_rob = np.linalg.norm(dr)
            if d_rob < 1.2:
                v = v * 0.4 + (dr / max(d_rob, 1e-6)) * 0.4  # robottan uzaklaş
        self.vel = v
        self.pos = self.pos + v * dt


# ----------------------------------------------------------------------------
# Ofis Ortamı
# ----------------------------------------------------------------------------

class OfficeEnvironment:
    """Dinamik ofis ortamı: statik engeller + insanlar + occupancy grid."""

    def __init__(self, width=20.0, height=12.0, resolution=0.2, scenario=0, seed=None):
        self.width = width
        self.height = height
        self.res = resolution
        self.rng = np.random.default_rng(seed)
        self.nx = int(width / resolution)
        self.ny = int(height / resolution)

        self.walls = []       # dış duvarlar + iç bölmeler (rect listesi)
        self.furniture = []   # mobilyalar (rect listesi)
        self.humans = []      # dinamik engeller
        self.landmarks = []   # EKF-SLAM için landmark noktaları (mobilya köşeleri)
        self.start = np.array([1.0, 1.0])
        self.goal = np.array([18.5, 10.5])

        self._build_scenario(scenario)
        self._extract_landmarks()
        self.static_grid = self._build_static_grid()

    # ------------------------------------------------------------------
    # Senaryolar: 10 farklı ofis düzeni / insan yoğunluğu
    # ------------------------------------------------------------------
    def _build_scenario(self, s):
        W, H = self.width, self.height
        t = 0.15  # duvar kalınlığı
        # Dış duvarlar (hepsinde ortak)
        self.walls = [
            (0, 0, W, t), (0, H - t, W, t), (0, 0, t, H), (W - t, 0, t, H),
        ]

        def add_room_divider_h(y, x0, x1, door_x, door_w=1.2):
            """Yatay bölme duvarı, kapı boşluğu ile."""
            if door_x - x0 > 0.1:
                self.walls.append((x0, y, door_x - x0, t))
            if x1 - (door_x + door_w) > 0.1:
                self.walls.append((door_x + door_w, y, x1 - door_x - door_w, t))

        def add_room_divider_v(x, y0, y1, door_y, door_w=1.2):
            if door_y - y0 > 0.1:
                self.walls.append((x, y0, t, door_y - y0))
            if y1 - (door_y + door_w) > 0.1:
                self.walls.append((x, door_y + door_w, t, y1 - door_y - door_w))

        # Temel ofis düzeni: 2 oda + açık ofis + koridor
        add_room_divider_v(6.0, 0, 7.0, 3.0)          # sol oda bölmesi
        add_room_divider_h(7.0, 0, 6.0, 2.0)          # sol oda üst duvar
        add_room_divider_v(13.0, 5.0, H, 8.5)         # sağ oda bölmesi
        add_room_divider_h(5.0, 13.0, W, 16.0)        # sağ oda alt duvar

        # Senaryoya göre mobilya düzenleri
        base_furniture = [
            (2.5, 3.0, 1.6, 0.8),    # masa
            (8.0, 2.0, 2.0, 1.0),    # toplantı masası
            (9.0, 8.5, 1.6, 0.8),    # masa
            (15.0, 7.0, 1.2, 2.0),   # dolap
            (4.0, 9.0, 1.6, 0.8),    # masa
        ]
        extra_sets = [
            [],
            [(11.0, 5.5, 1.0, 1.0)],
            [(6.8, 8.0, 1.2, 1.2), (11.5, 1.0, 0.8, 1.5)],
            [(3.0, 5.5, 1.0, 2.2)],
            [(10.0, 10.2, 2.2, 0.8), (14.0, 1.5, 1.5, 0.9)],
            [(7.5, 5.8, 2.4, 0.7)],
            [(2.0, 8.2, 0.9, 0.9), (16.5, 2.0, 1.2, 1.2)],
            [(12.0, 9.0, 0.8, 2.0), (5.0, 1.8, 1.4, 0.7)],
            [(8.8, 6.8, 1.2, 1.0), (15.8, 9.5, 1.6, 0.8)],
            [(3.5, 4.8, 2.0, 0.6), (10.5, 3.8, 0.9, 1.8), (17.0, 6.5, 1.0, 1.0)],
        ]
        self.furniture = base_furniture + extra_sets[s % len(extra_sets)]

        # İnsan sayısı senaryo ile artar: 2..6
        n_humans = 2 + (s % 5)
        human_routes = [
            [(4.5, 1.0), (10.0, 1.2), (10.0, 4.5), (4.8, 4.2)],
            [(7.0, 6.0), (12.0, 6.2), (12.2, 10.5), (7.2, 10.2)],
            [(14.5, 1.0), (18.5, 1.2), (18.2, 4.0), (14.2, 3.8)],
            [(1.0, 5.8), (5.0, 5.5), (5.2, 8.0), (1.2, 8.2)],
            [(14.0, 9.8), (18.0, 10.0), (18.2, 6.5), (14.2, 6.8)],
            [(6.5, 1.5), (6.8, 5.5), (11.5, 5.8), (11.2, 2.0)],
        ]
        speeds = [0.5, 0.65, 0.8, 0.55, 0.7, 0.6]
        for i in range(n_humans):
            self.humans.append(Human(human_routes[i % len(human_routes)],
                                     speed=speeds[i % len(speeds)], rng=self.rng))

        # Başlangıç/hedef: senaryoya göre çeşitlendir
        starts = [(1.0, 1.0), (1.0, 10.5), (18.5, 1.0), (1.2, 1.2), (2.0, 10.0)]
        goals = [(18.5, 10.5), (18.5, 1.5), (1.5, 10.5), (17.0, 10.8), (18.0, 2.0)]
        self.start = np.array(starts[s % len(starts)], dtype=float)
        self.goal = np.array(goals[s % len(goals)], dtype=float)

    def _extract_landmarks(self):
        """Mobilya ve iç duvar köşelerini EKF-SLAM landmark'ları olarak çıkar."""
        pts = []
        for (x0, y0, w, h) in self.furniture:
            pts += [(x0, y0), (x0 + w, y0), (x0, y0 + h), (x0 + w, y0 + h)]
        # İç duvar uçları da landmark
        for (x0, y0, w, h) in self.walls[4:]:
            pts += [(x0, y0), (x0 + w, y0 + h)]
        self.landmarks = np.array(pts)

    def _build_static_grid(self):
        """Statik occupancy grid (0=boş, 1=dolu). Robot yarıçapı kadar şişirilmemiş ham grid."""
        grid = np.zeros((self.ny, self.nx), dtype=np.uint8)
        xs = (np.arange(self.nx) + 0.5) * self.res
        ys = (np.arange(self.ny) + 0.5) * self.res
        XX, YY = np.meshgrid(xs, ys)
        for (x0, y0, w, h) in self.walls + self.furniture:
            mask = (XX >= x0) & (XX <= x0 + w) & (YY >= y0) & (YY <= y0 + h)
            grid[mask] = 1
        return grid

    # ------------------------------------------------------------------
    # Simülasyon adımı
    # ------------------------------------------------------------------
    def step(self, dt, robot_pos=None):
        for h in self.humans:
            h.step(dt, robot_pos=robot_pos)
            # Sınırlar içinde tut
            h.pos[0] = np.clip(h.pos[0], 0.4, self.width - 0.4)
            h.pos[1] = np.clip(h.pos[1], 0.4, self.height - 0.4)

    # ------------------------------------------------------------------
    # Çarpışma kontrolü
    # ------------------------------------------------------------------
    def collision(self, x, y, robot_radius=0.25):
        if x < robot_radius or x > self.width - robot_radius:
            return True
        if y < robot_radius or y > self.height - robot_radius:
            return True
        for rect in self.walls + self.furniture:
            # Dikdörtgene en yakın nokta
            x0, y0, w, h = rect
            cx = np.clip(x, x0, x0 + w)
            cy = np.clip(y, y0, y0 + h)
            if (x - cx) ** 2 + (y - cy) ** 2 < robot_radius ** 2:
                return True
        for hm in self.humans:
            if (x - hm.pos[0]) ** 2 + (y - hm.pos[1]) ** 2 < (robot_radius + Human.RADIUS) ** 2:
                return True
        return False

    def static_collision(self, x, y, robot_radius=0.25):
        for rect in self.walls + self.furniture:
            x0, y0, w, h = rect
            cx = np.clip(x, x0, x0 + w)
            cy = np.clip(y, y0, y0 + h)
            if (x - cx) ** 2 + (y - cy) ** 2 < robot_radius ** 2:
                return True
        return False

    # ------------------------------------------------------------------
    # LIDAR simülasyonu
    # ------------------------------------------------------------------
    def lidar_scan(self, x, y, theta, n_beams=48, max_range=5.0, noise_std=0.02):
        """360° LIDAR taraması (vektörize). Menzil dizisi (n_beams,) döner."""
        angles = theta + np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
        dx = np.cos(angles)
        dy = np.sin(angles)
        best = np.full(n_beams, max_range)

        # Dikdörtgenler: slab yöntemi, tüm ışınlar için vektörize
        eps = 1e-12
        inv_dx = 1.0 / np.where(np.abs(dx) < eps, eps, dx)
        inv_dy = 1.0 / np.where(np.abs(dy) < eps, eps, dy)
        for (x0, y0, w, h) in self.walls + self.furniture:
            t1 = (x0 - x) * inv_dx
            t2 = (x0 + w - x) * inv_dx
            t3 = (y0 - y) * inv_dy
            t4 = (y0 + h - y) * inv_dy
            tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
            tmin = np.maximum(tmin, 0.0)
            tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
            hit = (tmax >= tmin) & (tmax >= 0)
            best = np.where(hit & (tmin < best), tmin, best)

        # İnsanlar: daire kesişimi vektörize
        for hm in self.humans:
            fx, fy = x - hm.pos[0], y - hm.pos[1]
            b = 2 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - Human.RADIUS ** 2
            disc = b * b - 4 * c
            ok = disc >= 0
            sq = np.sqrt(np.where(ok, disc, 0))
            t1 = (-b - sq) / 2
            t2 = (-b + sq) / 2
            t = np.where(t1 >= 0, t1, np.where(t2 >= 0, t2, np.inf))
            best = np.where(ok & (t < best), t, best)

        ranges = best + self.rng.normal(0, noise_std, n_beams)
        return np.clip(ranges, 0.05, max_range), angles

    def observe_landmarks(self, x, y, theta, max_range=4.0, fov=2 * np.pi,
                          range_noise=0.05, bearing_noise=0.03):
        """Görüş menzilindeki landmark'ların (id, menzil, açı) gözlemleri (gürültülü)."""
        obs = []
        for j, (lx, ly) in enumerate(self.landmarks):
            dx, dy = lx - x, ly - y
            r = np.hypot(dx, dy)
            if r > max_range or r < 0.1:
                continue
            b = np.arctan2(dy, dx) - theta
            b = (b + np.pi) % (2 * np.pi) - np.pi
            r_n = r + self.rng.normal(0, range_noise)
            b_n = b + self.rng.normal(0, bearing_noise)
            obs.append((j, r_n, b_n))
        return obs


# ----------------------------------------------------------------------------
# Robot modeli (diferansiyel sürüş)
# ----------------------------------------------------------------------------

class Robot:
    RADIUS = 0.25       # m
    MAX_V = 1.0         # m/s
    MIN_V = -0.2        # m/s (geri)
    MAX_W = 2.5         # rad/s
    MAX_ACC = 1.5       # m/s^2
    MAX_DW = 4.0        # rad/s^2

    def __init__(self, x, y, theta=0.0):
        self.x, self.y, self.theta = x, y, theta
        self.v, self.w = 0.0, 0.0
        self.battery = 100.0
        self.distance_traveled = 0.0

    @property
    def pose(self):
        return np.array([self.x, self.y, self.theta])

    def step(self, v_cmd, w_cmd, dt, rng=None):
        """Hız komutlarını ivme sınırlarıyla uygula; gürültülü odometri döndür."""
        v_cmd = np.clip(v_cmd, self.MIN_V, self.MAX_V)
        w_cmd = np.clip(w_cmd, -self.MAX_W, self.MAX_W)
        dv = np.clip(v_cmd - self.v, -self.MAX_ACC * dt, self.MAX_ACC * dt)
        dw = np.clip(w_cmd - self.w, -self.MAX_DW * dt, self.MAX_DW * dt)
        self.v += dv
        self.w += dw
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta = (self.theta + self.w * dt + np.pi) % (2 * np.pi) - np.pi
        self.distance_traveled += abs(self.v) * dt
        # Enerji modeli: hız + dönüş enerji harcar
        self.battery -= (0.02 * abs(self.v) + 0.01 * abs(self.w) + 0.002) * dt * 10
        # Gürültülü odometri (EKF girişi)
        if rng is not None:
            v_meas = self.v + rng.normal(0, 0.02)
            w_meas = self.w + rng.normal(0, 0.01)
        else:
            v_meas, w_meas = self.v, self.w
        return v_meas, w_meas
