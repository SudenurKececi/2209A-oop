# -*- coding: utf-8 -*-
"""
EKF-SLAM: Genişletilmiş Kalman Filtresi ile Eşzamanlı Konumlandırma ve Haritalama
==================================================================================
Durum vektörü: x = [xr, yr, theta, m1x, m1y, m2x, m2y, ...]
- Tahmin adımı: diferansiyel sürüş hareket modeli (gürültülü odometri girişi)
- Güncelleme adımı: landmark (menzil, açı) gözlemleri
- Sensör füzyonu: odometri + LIDAR landmark gözlemleri EKF içinde birleştirilir.

Ayrıca LIDAR taramalarından log-odds occupancy grid haritalama yapılır.
"""

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKFSLAM:
    def __init__(self, x0, y0, theta0, motion_noise=(0.02, 0.02), meas_noise=(0.08, 0.05)):
        # Durum: robot pozu; landmark'lar keşfedildikçe eklenir
        self.mu = np.array([x0, y0, theta0], dtype=float)
        self.Sigma = np.diag([0.01, 0.01, 0.01])
        self.R_v, self.R_w = motion_noise          # süreç gürültüsü (v, w)
        self.Q = np.diag([meas_noise[0] ** 2, meas_noise[1] ** 2])  # ölçüm gürültüsü
        self.lm_index = {}   # landmark id -> durum vektöründeki indeks

    @property
    def pose(self):
        return self.mu[:3].copy()

    @property
    def n_landmarks(self):
        return len(self.lm_index)

    # ------------------------------------------------------------------
    # Tahmin adımı (Prediction): x_k = f(x_{k-1}, u_k)
    # ------------------------------------------------------------------
    def predict(self, v, w, dt):
        th = self.mu[2]
        # Hareket modeli
        self.mu[0] += v * np.cos(th) * dt
        self.mu[1] += v * np.sin(th) * dt
        self.mu[2] = wrap(self.mu[2] + w * dt)

        n = len(self.mu)
        # Jacobian F (robot bloğu)
        F = np.eye(n)
        F[0, 2] = -v * np.sin(th) * dt
        F[1, 2] = v * np.cos(th) * dt
        # Süreç gürültüsü sadece robot pozuna
        Qp = np.zeros((n, n))
        Qp[0, 0] = (self.R_v * dt) ** 2 + (0.5 * v * dt) ** 2 * 0.01
        Qp[1, 1] = (self.R_v * dt) ** 2 + (0.5 * v * dt) ** 2 * 0.01
        Qp[2, 2] = (self.R_w * dt) ** 2
        self.Sigma = F @ self.Sigma @ F.T + Qp

    # ------------------------------------------------------------------
    # Güncelleme adımı (Update): landmark gözlemleri z = (r, b)
    # ------------------------------------------------------------------
    def update(self, observations):
        for (lm_id, r, b) in observations:
            if lm_id not in self.lm_index:
                self._add_landmark(lm_id, r, b)
                continue
            idx = self.lm_index[lm_id]
            lx, ly = self.mu[idx], self.mu[idx + 1]
            dx, dy = lx - self.mu[0], ly - self.mu[1]
            q = dx * dx + dy * dy
            r_hat = np.sqrt(q)
            b_hat = wrap(np.arctan2(dy, dx) - self.mu[2])
            z = np.array([r, b])
            z_hat = np.array([r_hat, b_hat])
            innov = z - z_hat
            innov[1] = wrap(innov[1])

            n = len(self.mu)
            H = np.zeros((2, n))
            H[0, 0] = -dx / r_hat
            H[0, 1] = -dy / r_hat
            H[1, 0] = dy / q
            H[1, 1] = -dx / q
            H[1, 2] = -1.0
            H[0, idx] = dx / r_hat
            H[0, idx + 1] = dy / r_hat
            H[1, idx] = -dy / q
            H[1, idx + 1] = dx / q

            S = H @ self.Sigma @ H.T + self.Q
            K = self.Sigma @ H.T @ np.linalg.inv(S)   # Kalman kazancı
            self.mu = self.mu + K @ innov
            self.mu[2] = wrap(self.mu[2])
            self.Sigma = (np.eye(n) - K @ H) @ self.Sigma

    def _add_landmark(self, lm_id, r, b):
        """Yeni landmark'ı durum vektörüne ekle (state augmentation)."""
        th = self.mu[2]
        lx = self.mu[0] + r * np.cos(th + b)
        ly = self.mu[1] + r * np.sin(th + b)
        n = len(self.mu)
        self.lm_index[lm_id] = n
        self.mu = np.append(self.mu, [lx, ly])
        Sigma_new = np.eye(n + 2) * 0.0
        Sigma_new[:n, :n] = self.Sigma
        Sigma_new[n, n] = 1.0    # yeni landmark başlangıç belirsizliği
        Sigma_new[n + 1, n + 1] = 1.0
        self.Sigma = Sigma_new

    def localization_error(self, true_pose):
        return float(np.hypot(self.mu[0] - true_pose[0], self.mu[1] - true_pose[1]))


class OccupancyMapper:
    """LIDAR taramalarından log-odds occupancy grid haritalama.

    Robot keşfettikçe harita güncellenir; Dynamic A* bu haritayı kullanır.
    """

    L_OCC = 0.85
    L_FREE = -0.4
    L_MIN, L_MAX = -4.0, 6.0

    def __init__(self, width, height, resolution):
        self.res = resolution
        self.nx = int(width / resolution)
        self.ny = int(height / resolution)
        self.logodds = np.zeros((self.ny, self.nx))

    def world_to_grid(self, x, y):
        return int(np.clip(x / self.res, 0, self.nx - 1)), int(np.clip(y / self.res, 0, self.ny - 1))

    DECAY = 0.97  # dinamik engellerin (insan) haritada kalıcılaşmasını önler

    def integrate_scan(self, pose, ranges, angles, max_range):
        # Zamansal sönümleme: hareketli nesnelerin izleri yavaşça silinir
        self.logodds *= self.DECAY
        x, y = pose[0], pose[1]
        gi, gj = self.world_to_grid(x, y)
        for r, a in zip(ranges, angles):
            hit = r < max_range - 0.1
            ex = x + r * np.cos(a)
            ey = y + r * np.sin(a)
            ei, ej = self.world_to_grid(ex, ey)
            # Bresenham ışın izleme: boş hücreler
            for (ci, cj) in self._bresenham(gi, gj, ei, ej)[:-1]:
                self.logodds[cj, ci] = np.clip(self.logodds[cj, ci] + self.L_FREE,
                                               self.L_MIN, self.L_MAX)
            if hit:
                self.logodds[ej, ei] = np.clip(self.logodds[ej, ei] + self.L_OCC,
                                               self.L_MIN, self.L_MAX)

    def occupancy_grid(self, threshold=0.0):
        """1 = dolu, 0 = boş/bilinmiyor."""
        return (self.logodds > threshold).astype(np.uint8)

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells
