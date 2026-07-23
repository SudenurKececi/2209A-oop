# -*- coding: utf-8 -*-
"""
Canlı ve İnteraktif Simülasyon Görüntüleyici
=============================================
Sol panel: dinamik ofis ortamı (gerçek dünya)
Sağ panel: robotun EKF-SLAM ile keşfettiği harita + konum tahmini

FARE:
  SOL TIK           : yeni hedef (robot rotayı anında yeniden planlar)
  SHIFT + SOL TIK   : hedefi GÖREV LİSTESİNE ekle (sırayla ziyaret eder)
  SAĞ TIK           : haritaya engel (kutu) ekle
  ORTA TIK          : tıklanan yere yürüyen insan ekle

KLAVYE (imleç haritanın üzerindeyken):
  H : imlecin olduğu yere insan ekle        D : imlece en yakın insanı sil
  B : bataryayı %25'e düşür (şarj senaryosunu tetikler)

Şarj istasyonu: batarya %30'un altına inince robot görevini erteleyip
istasyona (sarı kare) gider, %95'e şarj olur ve görevine kaldığı yerden döner.

Kullanım:  python main.py live [senaryo_no 1-10] [yöntem]
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Backend seçimi: bozuk Qt kurulumlarını atlamak için önce Tk denenir.
# ---------------------------------------------------------------------------
import matplotlib


def _pick_backend():
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg", force=True)
        return "TkAgg"
    except Exception:
        pass
    for bk in ("QtAgg", "Qt5Agg", "WXAgg", "GTK3Agg"):
        try:
            matplotlib.use(bk, force=True)
            import matplotlib.pyplot as _p
            _p.figure(); _p.close("all")
            return bk
        except Exception:
            continue
    # Hiçbir GUI backend yok: bozuk ayarı temizle (Agg ile en azından
    # kayıt modları çalışır)
    matplotlib.use("Agg", force=True)
    return None


_BACKEND = _pick_backend()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ppo import PPOAgent
from simulation import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

CHARGE_STATION = None   # run_live içinde env.start olarak atanır


class LiveViewer:
    def __init__(self, render_every=3, interactive=True, charge_station=None):
        self.render_every = render_every
        self.interactive = interactive
        self.charge_station = charge_station
        self.step = 0
        self.fig = None
        self.trail_x, self.trail_y = [], []
        self.closed = False
        self.status_msg = ""
        self.method_name = ""
        self._robot_pos = None
        # bekleyen fare/klavye komutları
        self.pending = {}
        # görev listesi (kuyruk)
        self.goal_queue = []
        self.queue_artists = []

    # ------------------------------------------------------------------
    def _setup(self, env):
        plt.ion()
        self.fig = plt.figure(figsize=(15, 7.2))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.12)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        try:
            self.fig.canvas.manager.set_window_title(
                "Hibrit Yol Planlama — Canlı Simülasyon")
        except Exception:
            pass
        self.fig.canvas.mpl_connect('close_event',
                                    lambda e: setattr(self, 'closed', True))
        if self.interactive:
            self.fig.canvas.mpl_connect('button_press_event', self._on_click)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # ---------------- Sol panel: gerçek dünya ----------------
        ax = self.ax
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect("equal")
        ax.set_title(f"{self.method_name}  |  SOL TIK: hedef  SHIFT+TIK: listeye ekle  "
                     f"SAĞ TIK: engel  ORTA/H: insan  D: insan sil  B: batarya",
                     fontsize=9)
        for (x0, y0, w, h) in env.walls:
            ax.add_patch(mpatches.Rectangle((x0, y0), w, h, color="#333333", zorder=2))
        self.furn_patches = []
        for (x0, y0, w, h) in env.furniture:
            p = mpatches.Rectangle((x0, y0), w, h, color="#8d6e63", zorder=2)
            ax.add_patch(p)
        self.n_furniture = len(env.furniture)

        if self.charge_station is not None:
            sx, sy = self.charge_station
            ax.add_patch(mpatches.Rectangle((sx - 0.35, sy - 0.35), 0.7, 0.7,
                                            color="#f4a261", zorder=3))
            ax.text(sx, sy, "⚡", ha="center", va="center", fontsize=13, zorder=4)

        self.goal_marker, = ax.plot(*env.goal, "r*", markersize=20, zorder=7,
                                    label="Hedef")
        self.path_ln, = ax.plot([], [], "-", color="#457b9d", lw=1.8, alpha=0.85,
                                label="Global yol (Dynamic A*)")
        self.trail_ln, = ax.plot([], [], "-", color="#2a9d8f", lw=1.6, alpha=0.8,
                                 label="Robot izi")
        self.lidar_sc = ax.scatter([], [], s=4, c="#999999", zorder=3, label="LIDAR")
        self.robot_dot, = ax.plot([], [], "o", color="#2a9d8f", markersize=13, zorder=6)
        self.heading_ln, = ax.plot([], [], "-", color="#1b5e20", lw=2.5, zorder=6)
        self.human_dots = []
        self.info_txt = ax.text(0.35, env.height - 0.7, "", fontsize=10, zorder=8,
                                bbox=dict(facecolor="white", alpha=0.8,
                                          edgecolor="none"))
        self.msg_txt = ax.text(0.35, 0.35, "", fontsize=10, zorder=8, color="#b23a48",
                               bbox=dict(facecolor="white", alpha=0.8,
                                         edgecolor="none"))
        ax.legend(loc="lower right", fontsize=7)

        # ---------------- Sağ panel: EKF-SLAM haritası ----------------
        ax2 = self.ax2
        ax2.set_title("Robotun Keşfettiği Harita (EKF-SLAM)", fontsize=10)
        self.map_im = ax2.imshow(np.zeros((10, 10)), cmap="Blues", origin="lower",
                                 extent=[0, env.width, 0, env.height],
                                 vmin=0, vmax=1, interpolation="nearest")
        self.est_trail_ln, = ax2.plot([], [], "-", color="#2a9d8f", lw=1.2,
                                      alpha=0.8, label="EKF poz tahmini")
        self.est_dot, = ax2.plot([], [], "o", color="#e76f51", markersize=8,
                                 zorder=5, label="Tahmini konum")
        self.true_dot, = ax2.plot([], [], "x", color="#1b5e20", markersize=8,
                                  zorder=5, label="Gerçek konum")
        self.lm_sc = ax2.scatter([], [], s=18, marker="+", c="#b23a48",
                                 zorder=4, label="Landmark tahminleri")
        ax2.set_xlim(0, env.width)
        ax2.set_ylim(0, env.height)
        ax2.set_aspect("equal")
        ax2.legend(loc="lower right", fontsize=7)
        self.est_trail = [[], []]
        self.fig.tight_layout()
        self._env = env

    # ------------------------------------------------------------------
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        env = self._env
        if event.button == 1:
            if env.static_collision(x, y, 0.3):
                self.status_msg = f"({x:.1f},{y:.1f}) engel içinde — başka nokta seç"
                return
            if event.key == "shift":       # SHIFT+SOL TIK -> görev listesine ekle
                self.goal_queue.append((x, y))
                m = self.ax.text(x, y, str(len(self.goal_queue)), fontsize=11,
                                 ha="center", va="center", zorder=7, color="white",
                                 bbox=dict(boxstyle="circle", facecolor="#457b9d",
                                           edgecolor="none"))
                self.queue_artists.append(m)
                self.status_msg = f"Görev listesine eklendi (#{len(self.goal_queue)})"
            else:                          # SOL TIK -> anlık hedef
                self.pending["goal"] = (x, y)
                self.status_msg = f"Yeni hedef: ({x:.1f}, {y:.1f})"
        elif event.button == 2:            # ORTA TIK -> insan ekle
            self.pending["add_human"] = (x, y)
            self.status_msg = f"İnsan eklendi: ({x:.1f}, {y:.1f})"
        elif event.button == 3:            # SAĞ TIK -> engel ekle
            rb = self._robot_pos
            if rb is not None and np.hypot(x - rb[0], y - rb[1]) < 1.0:
                self.status_msg = "Engel robota çok yakın — daha uzağa koy"
            else:
                self.pending["obstacle"] = (x, y)
                self.status_msg = f"Yeni engel: ({x:.1f}, {y:.1f})"

    def _on_key(self, event):
        if event.key is None:
            return
        k = event.key.lower()
        if k == "b":
            self.pending["set_battery"] = 25.0
            self.status_msg = "Batarya %25'e düşürüldü — şarj senaryosu başlıyor"
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if k == "h":
            self.pending["add_human"] = (x, y)
            self.status_msg = f"İnsan eklendi: ({x:.1f}, {y:.1f})"
        elif k == "d":
            if self._env.humans:
                self.pending["remove_human"] = (x, y)
                self.status_msg = "İmlece en yakın insan silindi"
            else:
                self.status_msg = "Silinecek insan yok"

    # ------------------------------------------------------------------
    def __call__(self, env=None, robot=None, t=None, ranges=None, angles=None,
                 global_path=None, est=None, mapper=None, ekf=None,
                 charge_state="normal"):
        if self.closed:
            raise KeyboardInterrupt
        if self.fig is None:
            self._setup(env)
        self.step += 1
        self._robot_pos = (robot.x, robot.y)
        self.trail_x.append(robot.x)
        self.trail_y.append(robot.y)
        self.est_trail[0].append(est[0])
        self.est_trail[1].append(est[1])

        # Görev listesi: hedefe varıldıysa sıradakini gönder
        cmd = dict(self.pending)
        self.pending.clear()
        d_goal = np.hypot(env.goal[0] - robot.x, env.goal[1] - robot.y)
        if ("goal" not in cmd and self.goal_queue and d_goal < 0.45
                and charge_state == "normal"):
            cmd["goal"] = self.goal_queue.pop(0)
            art = self.queue_artists.pop(0)
            art.remove()
            # kalan numaraları güncelle
            for i, a in enumerate(self.queue_artists):
                a.set_text(str(i + 1))
            self.status_msg = f"Sıradaki göreve gidiliyor ({len(self.goal_queue)} kaldı)"

        if self.step % self.render_every == 0:
            self._render(env, robot, t, ranges, angles, global_path,
                         est, mapper, ekf, charge_state, d_goal)
        return cmd if cmd else None

    # ------------------------------------------------------------------
    def _render(self, env, robot, t, ranges, angles, global_path, est,
                mapper, ekf, charge_state, d_goal):
        # Yeni eklenen mobilyalar
        while self.n_furniture < len(env.furniture):
            (x0, y0, w, h) = env.furniture[self.n_furniture]
            self.ax.add_patch(mpatches.Rectangle((x0, y0), w, h,
                                                 color="#6d4c41", zorder=2))
            self.n_furniture += 1
        # İnsan sayısı değiştiyse noktaları yeniden kur
        if len(self.human_dots) != len(env.humans):
            for d in self.human_dots:
                d.remove()
            self.human_dots = [self.ax.plot([], [], "o", color="#e63946",
                                            markersize=10, zorder=5)[0]
                               for _ in env.humans]

        self.goal_marker.set_data([env.goal[0]], [env.goal[1]])
        self.robot_dot.set_data([robot.x], [robot.y])
        self.heading_ln.set_data([robot.x, robot.x + 0.5 * np.cos(robot.theta)],
                                 [robot.y, robot.y + 0.5 * np.sin(robot.theta)])
        self.trail_ln.set_data(self.trail_x, self.trail_y)
        if global_path is not None and len(global_path):
            self.path_ln.set_data(global_path[:, 0], global_path[:, 1])
        pts = np.stack([robot.x + ranges * np.cos(angles),
                        robot.y + ranges * np.sin(angles)], axis=1)
        self.lidar_sc.set_offsets(pts[ranges < 4.9])
        for dot, h in zip(self.human_dots, env.humans):
            dot.set_data([h.pos[0]], [h.pos[1]])

        durum = {"to_station": "ŞARJA GİDİYOR",
                 "charging": f"ŞARJ OLUYOR (%{robot.battery:.0f})"}.get(
            charge_state, "HEDEFTE — tıkla" if d_goal < 0.45 else
            f"hedefe {d_goal:.1f} m")
        gorev = f" | görev listesi: {len(self.goal_queue)}" if self.goal_queue else ""
        self.info_txt.set_text(f"t={t:5.1f}s  hız={robot.v:.2f} m/s  {durum}  "
                               f"batarya=%{robot.battery:.0f}  "
                               f"insan={len(env.humans)}{gorev}")
        self.msg_txt.set_text(self.status_msg)

        # Sağ panel: keşfedilen harita (her 5 render'da bir)
        if mapper is not None and self.step % (self.render_every * 5) == 0:
            self.map_im.set_data((mapper.logodds > 1.5).astype(float))
        self.est_dot.set_data([est[0]], [est[1]])
        self.true_dot.set_data([robot.x], [robot.y])
        self.est_trail_ln.set_data(self.est_trail[0], self.est_trail[1])
        if ekf is not None and ekf.n_landmarks:
            lms = ekf.mu[3:].reshape(-1, 2)
            self.lm_sc.set_offsets(lms)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def run_live(scenario=0, method="hybrid", seed=None, interactive=True):
    if _BACKEND is None:
        print("HATA: Grafik arayüzü başlatılamadı.")
        print("Çözüm: 'pip install PyQt5' deneyin veya Python'u python.org'dan")
        print("(tkinter dahil) yeniden kurun.")
        return
    print(f"Grafik arayüzü: {_BACKEND}")

    agent = PPOAgent(seed=1)
    model = os.path.join(RESULTS, "ppo_model.npz")
    if method == "hybrid" and os.path.exists(model):
        agent.load(model)
        print("Eğitilmiş PPO modeli yüklendi.")

    # Şarj istasyonu: robotun başlangıç (dok) noktası
    from environment import OfficeEnvironment
    tmp_env = OfficeEnvironment(scenario=scenario, seed=seed)
    station = tuple(tmp_env.start)

    viewer = LiveViewer(interactive=interactive, charge_station=station)
    viewer.method_name = {"hybrid": "Hibrit (A*+DWA+PPO)",
                          "classic": "Klasik (A*+DWA)",
                          "apf": "APF (Potansiyel Alan)"}.get(method, method)
    print(f"Senaryo {scenario+1} | {viewer.method_name} başlatılıyor...")
    print("SOL TIK: hedef | SHIFT+TIK: görev listesi | SAĞ TIK: engel")
    print("ORTA TIK / H: insan ekle | D: insan sil | B: batarya düşür (şarj demosu)")
    try:
        res, _ = run_episode(scenario=scenario, method=method,
                             agent=agent if method == "hybrid" else None,
                             seed=seed, on_step=viewer,
                             max_time=3600.0 if interactive else 120.0,
                             hold_on_goal=interactive,
                             charge_station=station)
        durum = "HEDEFE ULAŞTI" if res.success else \
                ("ÇARPIŞMA!" if res.collision else "SÜRE AŞIMI")
        print(f"\nSonuç: {durum}  süre={res.time:.1f}s  yol={res.path_length:.1f}m")
        if viewer.fig is not None and not viewer.closed:
            plt.ioff()
            print("Pencereyi kapatınca program sonlanır.")
            plt.show()
    except KeyboardInterrupt:
        print("\nSimülasyon durduruldu. Görüşmek üzere!")


if __name__ == "__main__":
    s = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
    m = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    run_live(scenario=max(0, min(9, s)), method=m)
