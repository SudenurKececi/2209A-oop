# -*- coding: utf-8 -*-
"""
Yöntem Yarışı Modu
==================
İki yöntemi AYNI senaryoda, AYNI koşullarda yan yana yarıştırır ve
eş zamanlı animasyonla gösterir. Örn: hibrit yöntem ile APF'nin farkını
canlı izlemek için idealdir.

Kullanım:
    python main.py race                 -> Senaryo 1: hibrit vs APF
    python main.py race 3               -> Senaryo 3: hibrit vs APF
    python main.py race 3 hybrid classic -> Senaryo 3: hibrit vs klasik
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from live_view import _pick_backend, _BACKEND  # backend'i Tk'ye zorlar

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ppo import PPOAgent
from simulation import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

ADLAR = {"hybrid": "Hibrit (A*+DWA+PPO)",
         "classic": "Klasik (A*+DWA)",
         "apf": "APF (Potansiyel Alan)"}


def _compute(scenario, method, seed, agent):
    print(f"  {ADLAR.get(method, method)} koşuluyor...", flush=True)
    res, env = run_episode(scenario=scenario, method=method,
                           agent=agent if method == "hybrid" else None,
                           seed=seed, record=True, max_time=90.0)
    return res, env


def _draw_static(ax, env, title):
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    for (x0, y0, w, h) in env.walls:
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h, color="#333333"))
    for (x0, y0, w, h) in env.furniture:
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h, color="#8d6e63"))
    ax.plot(*env.start, "gs", markersize=9)
    ax.plot(*env.goal, "r*", markersize=15)
    robot, = ax.plot([], [], "o", color="#2a9d8f", markersize=11, zorder=6)
    trail, = ax.plot([], [], "-", color="#2a9d8f", lw=1.4, alpha=0.8)
    humans = [ax.plot([], [], "o", color="#e63946", markersize=8, zorder=5)[0]
              for _ in env.humans]
    status = ax.text(0.35, env.height - 0.7, "", fontsize=10,
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    return dict(robot=robot, trail=trail, humans=humans, status=status,
                tx=[], ty=[])


def run_race(scenario=0, m1="hybrid", m2="apf", seed=7, save_gif=False):
    agent = PPOAgent(seed=1)
    model = os.path.join(RESULTS, "ppo_model.npz")
    if os.path.exists(model):
        agent.load(model)

    print(f"Senaryo {scenario+1} yarışı hazırlanıyor (aynı tohum, aynı ortam):")
    res1, env1 = _compute(scenario, m1, seed, agent)
    res2, env2 = _compute(scenario, m2, seed, agent)

    tr1, tr2 = res1.trajectory, res2.trajectory
    t_max = max(tr1[-1][3], tr2[-1][3])

    if not save_gif:
        plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    try:
        fig.canvas.manager.set_window_title("Yöntem Yarışı")
    except Exception:
        pass
    closed = {"v": False}
    fig.canvas.mpl_connect('close_event', lambda e: closed.update(v=True))

    def sonuc(res):
        return "ULAŞTI" if res.success else ("ÇARPTI" if res.collision else "KALDI")

    a1 = _draw_static(axes[0], env1, f"{ADLAR.get(m1, m1)}")
    a2 = _draw_static(axes[1], env2, f"{ADLAR.get(m2, m2)}")
    fig.suptitle(f"Senaryo {scenario+1} — YÖNTEM YARIŞI", fontsize=12)
    fig.tight_layout()

    def update_panel(art, traj, res, t):
        # t anındaki kareyi bul
        idx = min(int(t / 0.1), len(traj) - 1)
        x, y, th, tt, hums = traj[idx]
        art["robot"].set_data([x], [y])
        art["tx"].append(x); art["ty"].append(y)
        art["trail"].set_data(art["tx"], art["ty"])
        for dot, (hx, hy) in zip(art["humans"], hums):
            dot.set_data([hx], [hy])
        if t >= traj[-1][3]:
            art["status"].set_text(f"{sonuc(res)}  {res.time:.1f} s, "
                                   f"{res.path_length:.1f} m")
        else:
            art["status"].set_text(f"t = {t:.1f} s")

    frames = np.arange(0, t_max + 0.3, 0.6 if save_gif else 0.3)
    if save_gif:
        from matplotlib.animation import FuncAnimation, PillowWriter

        def anim_update(i):
            update_panel(a1, tr1, res1, frames[i])
            update_panel(a2, tr2, res2, frames[i])
            return []
        anim = FuncAnimation(fig, anim_update, frames=len(frames), blit=False)
        out = os.path.join(RESULTS, f"yaris_s{scenario+1}_{m1}_vs_{m2}.gif")
        anim.save(out, writer=PillowWriter(fps=10))
        plt.close(fig)
        print(f"Yarış GIF kaydedildi: {out}")
        return out

    for t in frames:
        if closed["v"]:
            print("Yarış penceresi kapatıldı.")
            return
        update_panel(a1, tr1, res1, t)
        update_panel(a2, tr2, res2, t)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.03)

    # Kazanan
    def skor(r):
        return (0 if r.success else 1, r.time)
    kazanan = ADLAR.get(m1, m1) if skor(res1) <= skor(res2) else ADLAR.get(m2, m2)
    fig.suptitle(f"Senaryo {scenario+1} — KAZANAN: {kazanan}", fontsize=13,
                 color="#2a9d8f")
    print(f"\n{ADLAR.get(m1, m1):24s}: {sonuc(res1)}  {res1.time:.1f}s  {res1.path_length:.1f}m")
    print(f"{ADLAR.get(m2, m2):24s}: {sonuc(res2)}  {res2.time:.1f}s  {res2.path_length:.1f}m")
    print(f"KAZANAN: {kazanan}")
    plt.ioff()
    print("Pencereyi kapatınca program sonlanır.")
    plt.show()


if __name__ == "__main__":
    s = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
    m1 = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    m2 = sys.argv[3] if len(sys.argv) > 3 else "apf"
    run_race(scenario=max(0, min(9, s)), m1=m1, m2=m2)
