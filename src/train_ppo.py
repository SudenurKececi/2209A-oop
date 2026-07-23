# -*- coding: utf-8 -*-
"""
PPO Eğitim Döngüsü
==================
Hibrit mimarinin uyarlanabilir katmanı: PPO ajanı, farklı ofis senaryolarında
görevler koşarak DWA ağırlıklarını uyarlamayı öğrenir.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ppo import PPOAgent
from simulation import run_episode


def train(n_iterations=12, episodes_per_iter=6, seed=42, save_path=None, log_path=None,
          resume=False, start_iter=0):
    agent = PPOAgent(seed=seed + start_iter)
    if resume and save_path and os.path.exists(save_path):
        agent.load(save_path)
        print(f"Model yuklendi, iter {start_iter}'den devam ediliyor.")
    rng = np.random.default_rng(seed + 7 * start_iter)
    history = []

    for it in range(start_iter, start_iter + n_iterations):
        buf = dict(states=[], actions=[], logps=[], rewards=[], dones=[], values=[])
        ep_stats = []
        for ep in range(episodes_per_iter):
            scenario = int(rng.integers(0, 10))
            ep_seed = int(rng.integers(0, 100000))
            res, _ = run_episode(scenario=scenario, method="hybrid", agent=agent,
                                 seed=ep_seed, ppo_buffer=buf, deterministic=False,
                                 max_time=70.0)
            ep_stats.append((res.success, res.collision, res.time, res.path_length))
        if len(buf['rewards']) > 10:
            agent.update(buf)
        sr = np.mean([s[0] for s in ep_stats])
        cr = np.mean([s[1] for s in ep_stats])
        mt = np.mean([s[2] for s in ep_stats])
        history.append((it, sr, cr, mt, float(np.mean(buf['rewards'])) if buf['rewards'] else 0))
        line = (f"[iter {it+1:02d}] basari={sr:.2f} carpisma={cr:.2f} "
                f"ort_sure={mt:.1f}s ort_odul={history[-1][4]:.3f}")
        print(line, flush=True)
        if save_path:
            agent.save(save_path)   # her iterasyonda kaydet (kesintiye dayanıklı)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{it} {sr:.4f} {cr:.4f} {mt:.2f} {history[-1][4]:.4f}\n")

    return agent, history


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    t0 = time.time()
    train(n_iterations=n_iter, start_iter=start, resume=(start > 0),
          save_path=os.path.join(root, "results", "ppo_model.npz"),
          log_path=os.path.join(root, "results", "ppo_training_log.txt"))
    print(f"Egitim suresi: {time.time()-t0:.0f}s")
