# -*- coding: utf-8 -*-
"""
Dynamic Window Approach (DWA) — Lokal Yol Planlayıcı
=====================================================
Robotun kinematik/dinamik kısıtlarına göre erişilebilir (v, w) hız uzayında
örnekleme yapar, her adayı kısa ufukta simüle eder ve maliyet fonksiyonu ile
en iyi komutu seçer:

    G(v,w) = alpha * hedef_yönelimi + beta * engel_mesafesi + gamma * hız_verimliliği
             + delta * global_yol_takibi

alpha, beta, gamma, delta ağırlıkları PPO ajanı tarafından ortam durumuna göre
uyarlanabilir (hibrit uyarlanabilir yaklaşımın çekirdeği).
"""

import numpy as np


def radius_safe(robot):
    """Acil yavaşlama tetikleme mesafesi."""
    return robot.RADIUS + 0.35


class DWAPlanner:
    def __init__(self, dt=0.1, predict_time=2.0, n_v=7, n_w=15):
        self.dt = dt
        self.predict_time = predict_time
        self.n_v = n_v
        self.n_w = n_w
        # Varsayılan (klasik, sabit) ağırlıklar
        self.weights = dict(heading=1.0, clearance=1.2, velocity=0.6, path=0.8)

    def set_weights(self, heading, clearance, velocity, path):
        self.weights = dict(heading=heading, clearance=clearance,
                            velocity=velocity, path=path)

    # ------------------------------------------------------------------
    def command(self, robot, goal, obstacle_pts, global_path=None, human_states=None):
        """En iyi (v, w) komutunu seç.

        obstacle_pts : (N,2) LIDAR'dan elde edilen engel noktaları (dünya koord.)
        human_states : [(pos, vel), ...] dinamik engel tahmini için
        """
        # Dinamik pencere: mevcut hız +- ivme sınırları
        v_min = max(robot.MIN_V, robot.v - robot.MAX_ACC * self.dt * 3)
        v_max = min(robot.MAX_V, robot.v + robot.MAX_ACC * self.dt * 3)
        w_min = max(-robot.MAX_W, robot.w - robot.MAX_DW * self.dt * 3)
        w_max = min(robot.MAX_W, robot.w + robot.MAX_DW * self.dt * 3)

        vs = np.linspace(v_min, v_max, self.n_v)
        ws = np.linspace(w_min, w_max, self.n_w)

        best_score, best_cmd, best_traj = -np.inf, (0.0, 0.0), None
        n_steps = int(self.predict_time / self.dt)

        # İnsan gelecek konum tahmini (sabit hız modeli)
        pred_humans = []
        if human_states:
            for (pos, vel) in human_states:
                pred_humans.append((np.array(pos), np.array(vel)))

        obstacle_pts = np.asarray(obstacle_pts) if len(obstacle_pts) else np.zeros((0, 2))

        for v in vs:
            for w in ws:
                traj = self._simulate(robot, v, w, n_steps)
                clearance = self._clearance(traj, obstacle_pts, pred_humans, robot.RADIUS)
                if clearance <= 0.0:      # çarpışma -> aday elenir
                    continue
                heading = self._heading_score(traj, goal, robot.MAX_V * self.predict_time)
                # Yalnızca ileri hareket ödüllendirilir (geri vites kurtarma
                # manevraları için serbesttir ama puan almaz)
                vel_score = max(v, 0.0) / robot.MAX_V
                # Yol takibi: yola yakınlık x ilerleme çarpanı (duran robot yol
                # üzerinde olsa bile tam puan alamaz -> dar kapıda donma önlenir)
                path_score = self._path_score(traj, global_path) * \
                    (0.2 + 0.8 * min(1.0, max(v, 0.0) / (0.5 * robot.MAX_V)))
                # Engel mesafesi 0.4 m üzerinde doygunlaşır: "çok uzak" ile
                # "yeterince uzak" aynı puanı alır (klasik DWA yaklaşımı)
                score = (self.weights['heading'] * heading +
                         self.weights['clearance'] * min(clearance, 0.4) / 0.4 +
                         self.weights['velocity'] * vel_score +
                         self.weights['path'] * path_score)
                if score > best_score:
                    best_score, best_cmd, best_traj = score, (v, w), traj

        if best_traj is None:
            # Tüm adaylar çarpışıyor: acil dur + yerinde dön (kurtarma davranışı)
            return 0.0, robot.MAX_W * 0.6, None
        # Acil güvenlik: çok yakın engel varsa hızı kırp
        if len(obstacle_pts):
            d_now = float(np.min(np.linalg.norm(obstacle_pts -
                                                np.array([robot.x, robot.y]), axis=1)))
            if d_now < radius_safe(robot):
                best_cmd = (min(best_cmd[0], 0.25), best_cmd[1])
        return best_cmd[0], best_cmd[1], best_traj

    # ------------------------------------------------------------------
    def _simulate(self, robot, v, w, n_steps):
        x, y, th = robot.x, robot.y, robot.theta
        traj = np.empty((n_steps, 2))
        for i in range(n_steps):
            x += v * np.cos(th) * self.dt
            y += v * np.sin(th) * self.dt
            th += w * self.dt
            traj[i] = (x, y)
        return traj

    SAFETY_MARGIN = 0.12  # m — EKF/sensör belirsizliğine karşı güvenlik payı

    @staticmethod
    def _clearance(traj, obstacle_pts, pred_humans, radius):
        """Yörünge boyunca en yakın engel mesafesi (çarpışmada <=0). Vektörize."""
        min_d = np.inf
        sub = traj  # tüm noktalar (yoğun kontrol)
        if len(obstacle_pts):
            diff = obstacle_pts[None, :, :] - sub[:, None, :]
            min_d = float(np.min(np.sqrt(np.sum(diff ** 2, axis=2))))
        # Dinamik engeller: sabit hız modeli ile gelecekteki konumları
        dt = 0.1
        if pred_humans:
            steps = np.arange(len(traj))[:, None] * dt          # (K,1)
            for (pos, vel) in pred_humans:
                hp = pos[None, :] + vel[None, :] * steps        # (K,2)
                d = np.min(np.sqrt(np.sum((sub - hp) ** 2, axis=1))) - 0.30
                min_d = min(min_d, float(d))
        return min_d - radius - DWAPlanner.SAFETY_MARGIN

    @staticmethod
    def _heading_score(traj, goal, max_progress):
        d_end = np.linalg.norm(goal - traj[-1])
        d_start = np.linalg.norm(goal - traj[0])
        # Hedefe yaklaşma / mümkün olan azami ilerleme (normalize, [-1, 1])
        return np.clip((d_start - d_end) / (max_progress + 1e-6), -1.0, 1.0)

    @staticmethod
    def _path_score(traj, global_path):
        """Global A* yoluna sadakat: yol yoksa 0."""
        if global_path is None or len(global_path) < 2:
            return 0.0
        d = np.min(np.linalg.norm(global_path[None, :, :] - traj[::4, None, :], axis=2), axis=1)
        return float(np.clip(1.0 - np.mean(d) / 2.0, 0.0, 1.0))
