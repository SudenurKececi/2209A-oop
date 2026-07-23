# -*- coding: utf-8 -*-
"""
Benchmark: Hibrit Uyarlanabilir Yöntem vs Geleneksel Yöntemler
===============================================================
10 farklı ofis senaryosunda 3 yöntem karşılaştırılır:
  1. hybrid  : Dynamic A* + DWA + PPO (önerilen hibrit uyarlanabilir)
  2. classic : Dynamic A* + DWA (sabit ağırlık, geleneksel)
  3. apf     : Yapay Potansiyel Alan (geleneksel taban çizgisi)

Metrikler: başarı oranı, çarpışma oranı, görev süresi, yol uzunluğu,
enerji tüketimi, min. engel mesafesi, yeniden planlama sayısı, EKF konum hatası.
Çıktılar: results/benchmark_results.csv + karşılaştırma grafikleri.
"""

import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ppo import PPOAgent
from simulation import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

METHOD_NAMES = {"hybrid": "Hibrit (A*+DWA+PPO)",
                "classic": "Klasik (A*+DWA)",
                "apf": "APF (Potansiyel Alan)"}


def run_benchmark(n_runs=10, scenarios=range(10), model_path=None):
    os.makedirs(RESULTS, exist_ok=True)
    agent = PPOAgent(seed=1)
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"PPO modeli yuklendi: {model_path}")

    rows = []
    t0 = time.time()
    for method in ("hybrid", "classic", "apf"):
        for s in scenarios:
            for run in range(n_runs):
                seed = 1000 * s + run
                res, _ = run_episode(scenario=s, method=method,
                                     agent=agent if method == "hybrid" else None,
                                     seed=seed, max_time=90.0)
                d = res.as_dict()
                d.update(method=method, scenario=s, run=run, seed=seed)
                rows.append(d)
        done = sum(1 for r in rows if r['method'] == method)
        sr = np.mean([r['success'] for r in rows if r['method'] == method])
        print(f"{METHOD_NAMES[method]}: {done} kosum, basari orani={sr:.2%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    csv_path = os.path.join(RESULTS, "benchmark_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Sonuclar: {csv_path}")
    return rows


def make_charts(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})

    methods = ["hybrid", "classic", "apf"]
    colors = {"hybrid": "#2a9d8f", "classic": "#457b9d", "apf": "#e76f51"}

    def agg(metric, only_success=False):
        out = []
        for m in methods:
            vals = [r[metric] for r in rows if r['method'] == m and
                    (not only_success or r['success'])]
            out.append(vals)
        return out

    # ---- Grafik 1: Başarı ve çarpışma oranları ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sr = [100 * np.mean([r['success'] for r in rows if r['method'] == m]) for m in methods]
    cr = [100 * np.mean([r['collision'] for r in rows if r['method'] == m]) for m in methods]
    labels = [METHOD_NAMES[m] for m in methods]
    bars = axes[0].bar(labels, sr, color=[colors[m] for m in methods])
    axes[0].set_ylabel("Başarı Oranı (%)")
    axes[0].set_title("Görev Başarı Oranı")
    axes[0].set_ylim(0, 105)
    for b, v in zip(bars, sr):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center")
    bars = axes[1].bar(labels, cr, color=[colors[m] for m in methods])
    axes[1].set_ylabel("Çarpışma Oranı (%)")
    axes[1].set_title("Çarpışma Oranı")
    for b, v in zip(bars, cr):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}%", ha="center")
    for ax in axes:
        ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig1_basari_carpisma.png"))
    plt.close(fig)

    # ---- Grafik 2: Süre / yol uzunluğu / enerji (başarılı koşumlar) ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title, unit in zip(
            axes, ["time", "path_length", "energy"],
            ["Görev Süresi", "Yol Uzunluğu", "Enerji Tüketimi"],
            ["s", "m", "birim"]):
        data = agg(metric, only_success=True)
        bp = ax.boxplot(data, labels=[m.split()[0] for m in labels], patch_artist=True)
        for patch, m in zip(bp['boxes'], methods):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.7)
        ax.set_title(f"{title} ({unit})")
        ax.grid(alpha=0.3)
    fig.suptitle("Başarılı Görevlerde Verimlilik Karşılaştırması")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig2_verimlilik.png"))
    plt.close(fig)

    # ---- Grafik 3: Senaryo bazlı başarı ----
    fig, ax = plt.subplots(figsize=(11, 4))
    scenarios = sorted(set(r['scenario'] for r in rows))
    x = np.arange(len(scenarios))
    w = 0.27
    for i, m in enumerate(methods):
        vals = [100 * np.mean([r['success'] for r in rows
                               if r['method'] == m and r['scenario'] == s])
                for s in scenarios]
        ax.bar(x + (i - 1) * w, vals, w, label=METHOD_NAMES[m], color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s+1}" for s in scenarios])
    ax.set_xlabel("Senaryo")
    ax.set_ylabel("Başarı Oranı (%)")
    ax.set_title("Senaryo Bazlı Başarı Oranları (10 ofis senaryosu)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig3_senaryo_basari.png"))
    plt.close(fig)

    # ---- Grafik 4: Min. engel mesafesi + EKF hata ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = agg("min_clearance", only_success=True)
    bp = axes[0].boxplot(data, labels=[m.split()[0] for m in labels], patch_artist=True)
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(colors[m]); patch.set_alpha(0.7)
    axes[0].set_title("Min. Engel Mesafesi (m) — Güvenlik")
    axes[0].grid(alpha=0.3)
    data = agg("loc_error_mean")
    bp = axes[1].boxplot(data, labels=[m.split()[0] for m in labels], patch_artist=True)
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(colors[m]); patch.set_alpha(0.7)
    axes[1].set_title("EKF-SLAM Ort. Konum Hatası (m)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig4_guvenlik_konum.png"))
    plt.close(fig)

    print("Grafikler results/ klasorune kaydedildi.")


def print_summary(rows):
    methods = ["hybrid", "classic", "apf"]
    print("\n===== OZET TABLO =====")
    hdr = f"{'Yontem':<22}{'Basari':>8}{'Carpisma':>10}{'Sure(s)':>9}{'Yol(m)':>8}{'Enerji':>8}{'Replan':>8}"
    print(hdr)
    summary = {}
    for m in methods:
        R = [r for r in rows if r['method'] == m]
        S = [r for r in R if r['success']]
        row = dict(
            success=100 * np.mean([r['success'] for r in R]),
            collision=100 * np.mean([r['collision'] for r in R]),
            time=np.mean([r['time'] for r in S]) if S else float('nan'),
            path=np.mean([r['path_length'] for r in S]) if S else float('nan'),
            energy=np.mean([r['energy'] for r in S]) if S else float('nan'),
            replans=np.mean([r['replans'] for r in R]),
        )
        summary[m] = row
        print(f"{METHOD_NAMES[m]:<22}{row['success']:>7.1f}%{row['collision']:>9.1f}%"
              f"{row['time']:>9.1f}{row['path']:>8.1f}{row['energy']:>8.2f}{row['replans']:>8.1f}")
    return summary


def run_chunk(method, s0, s1, n_runs=10, run0=0):
    """Tek yöntem, senaryo aralığı [s0, s1) için koşum; CSV'ye ekler."""
    os.makedirs(RESULTS, exist_ok=True)
    agent = PPOAgent(seed=1)
    model = os.path.join(RESULTS, "ppo_model.npz")
    if method == "hybrid" and os.path.exists(model):
        agent.load(model)
    csv_path = os.path.join(RESULTS, "benchmark_results.csv")
    fields = ['success', 'collision', 'timeout', 'time', 'path_length', 'energy',
              'min_clearance', 'replans', 'loc_error_mean', 'method', 'scenario', 'run', 'seed']
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            wr.writeheader()
        for s in range(s0, s1):
            for run in range(run0, n_runs):
                seed = 1000 * s + run
                res, _ = run_episode(scenario=s, method=method,
                                     agent=agent if method == "hybrid" else None,
                                     seed=seed, max_time=90.0)
                d = res.as_dict()
                d.update(method=method, scenario=s, run=run, seed=seed)
                wr.writerow(d)
            print(f"{method} S{s+1} tamam", flush=True)


def load_rows():
    csv_path = os.path.join(RESULTS, "benchmark_results.csv")
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ('success', 'collision', 'timeout', 'scenario', 'run', 'replans'):
                r[k] = int(r[k])
            for k in ('time', 'path_length', 'energy', 'min_clearance', 'loc_error_mean'):
                r[k] = float(r[k])
            rows.append(r)
    # Yinelenen kayıtları temizle (kesintili koşumlardan)
    seen = {}
    for r in rows:
        seen[(r['method'], r['scenario'], r['run'])] = r
    return list(seen.values())


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "run":
        run_chunk(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                  n_runs=int(sys.argv[5]) if len(sys.argv) > 5 else 10,
                  run0=int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    elif len(sys.argv) >= 2 and sys.argv[1] == "charts":
        rows = load_rows()
        print_summary(rows)
        make_charts(rows)
    else:
        model = os.path.join(RESULTS, "ppo_model.npz")
        rows = run_benchmark(n_runs=10, model_path=model)
        print_summary(rows)
        make_charts(rows)
