# -*- coding: utf-8 -*-
"""
Simülasyon Animasyonu (GIF)
===========================
Robotun dinamik ofis ortamında hedefe gidişini animasyonlu GIF olarak kaydeder.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from ppo import PPOAgent
from simulation import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def make_gif(scenario=0, method="hybrid", seed=7, out_name=None, fps=12, frame_skip=3):
    agent = PPOAgent(seed=1)
    model = os.path.join(RESULTS, "ppo_model.npz")
    if method == "hybrid" and os.path.exists(model):
        agent.load(model)

    res, env = run_episode(scenario=scenario, method=method,
                           agent=agent if method == "hybrid" else None,
                           seed=seed, record=True, max_time=90.0)
    traj = res.trajectory
    if not traj:
        print("Yorunge kaydi bos!")
        return None

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=100)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    status = "BASARILI" if res.success else ("CARPISMA" if res.collision else "SURE ASIMI")
    ax.set_title(f"Senaryo {scenario+1} | {method} | {status} | "
                 f"sure={res.time:.1f}s yol={res.path_length:.1f}m")

    # Statik ortam
    for (x0, y0, w, h) in env.walls:
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h, color="#333333"))
    for (x0, y0, w, h) in env.furniture:
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h, color="#8d6e63"))
    ax.plot(*env.start, "gs", markersize=10, label="Başlangıç")
    ax.plot(*env.goal, "r*", markersize=16, label="Hedef")

    # Dinamik ögeler
    robot_dot, = ax.plot([], [], "o", color="#2a9d8f", markersize=11, zorder=5)
    heading_ln, = ax.plot([], [], "-", color="#1b5e20", lw=2, zorder=5)
    trail_ln, = ax.plot([], [], "-", color="#2a9d8f", lw=1.4, alpha=0.7)
    human_dots = [ax.plot([], [], "o", color="#e63946", markersize=9, zorder=4)[0]
                  for _ in env.humans]
    time_txt = ax.text(0.35, env.height - 0.65, "", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)

    frames = traj[::frame_skip]
    trail_x, trail_y = [], []

    def update(i):
        x, y, th, t, hums = frames[i]
        robot_dot.set_data([x], [y])
        heading_ln.set_data([x, x + 0.45 * np.cos(th)], [y, y + 0.45 * np.sin(th)])
        trail_x.append(x); trail_y.append(y)
        trail_ln.set_data(trail_x, trail_y)
        for dot, (hx, hy) in zip(human_dots, hums):
            dot.set_data([hx], [hy])
        time_txt.set_text(f"t = {t:.1f} s")
        return [robot_dot, heading_ln, trail_ln, time_txt] + human_dots

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    out_name = out_name or f"sim_senaryo{scenario+1}_{method}.gif"
    out_path = os.path.join(RESULTS, out_name)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF kaydedildi: {out_path} ({status})")
    return out_path


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    make_gif(scenario=0, method="hybrid", seed=7)
    make_gif(scenario=4, method="hybrid", seed=11)
    make_gif(scenario=0, method="apf", seed=7)
