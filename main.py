# -*- coding: utf-8 -*-
"""
TÜBİTAK 2209-A — Dinamik Ofis Ortamlarında Mobil Robotlar İçin
Hibrit Uyarlanabilir Yapay Zeka Tabanlı Optimum Yol Planlaması
================================================================
Ana çalıştırma betiği.

Kullanım:
    python main.py live         -> CANLI + İNTERAKTİF mod (tıkla-hedef, görev
                                   listesi, insan ekle/sil, şarj istasyonu,
                                   EKF-SLAM harita paneli)
    python main.py race         -> YÖNTEM YARIŞI: iki yöntemi yan yana yarıştır
                                   (örn: python main.py race 3 hybrid apf)
    python main.py train        -> PPO ajanını eğitir (results/ppo_model.npz)
    python main.py benchmark    -> 10 senaryoda 3 yöntemi karşılaştırır + grafikler
    python main.py animate      -> Örnek simülasyon GIF'leri üretir
    python main.py demo         -> Tek senaryo hızlı test (konsol çıktısı)
    python main.py all          -> Hepsini sırayla çalıştırır

Gereksinimler: numpy, scipy, matplotlib, pillow
    pip install numpy scipy matplotlib pillow
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def cmd_train():
    from train_ppo import train
    train(save_path=os.path.join(RESULTS, "ppo_model.npz"),
          log_path=os.path.join(RESULTS, "ppo_training_log.txt"))


def cmd_benchmark():
    from benchmark import run_benchmark, print_summary, make_charts
    rows = run_benchmark(n_runs=10, model_path=os.path.join(RESULTS, "ppo_model.npz"))
    print_summary(rows)
    make_charts(rows)


def cmd_animate():
    from animate import make_gif
    make_gif(scenario=0, method="hybrid", seed=7)
    make_gif(scenario=4, method="hybrid", seed=11)
    make_gif(scenario=0, method="apf", seed=7)


def cmd_live():
    from live_view import run_live
    s = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
    m = sys.argv[3] if len(sys.argv) > 3 else "hybrid"
    run_live(scenario=max(0, min(9, s)), method=m)


def cmd_race():
    from race import run_race
    s = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
    m1 = sys.argv[3] if len(sys.argv) > 3 else "hybrid"
    m2 = sys.argv[4] if len(sys.argv) > 4 else "apf"
    run_race(scenario=max(0, min(9, s)), m1=m1, m2=m2)


def cmd_demo():
    from ppo import PPOAgent
    from simulation import run_episode
    agent = PPOAgent(seed=1)
    model = os.path.join(RESULTS, "ppo_model.npz")
    if os.path.exists(model):
        agent.load(model)
    for method in ("hybrid", "classic", "apf"):
        res, _ = run_episode(scenario=0, method=method,
                             agent=agent if method == "hybrid" else None, seed=7)
        print(f"{method:8s} -> {res.as_dict()}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "all":
        cmd_train(); cmd_benchmark(); cmd_animate()
    else:
        {"train": cmd_train, "benchmark": cmd_benchmark, "live": cmd_live,
         "race": cmd_race, "animate": cmd_animate,
         "demo": cmd_demo}.get(cmd, cmd_demo)()
