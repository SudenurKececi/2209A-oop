# Dinamik Ofis Ortamlarında Mobil Robotlar İçin Hibrit Uyarlanabilir Yapay Zeka Tabanlı Optimum Yol Planlaması

TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Desteği Programı kapsamında
geliştirilen simülasyon projesi.

**Araştırmacı:** Sudenur KEÇECİ — Fırat Üniversitesi
**Danışman:** Prof. Dr. Mehmet KARAKÖSE

## Proje Özeti

Dinamik ofis ortamlarında (statik mobilyalar + hareketli insanlar) görev yapan mobil
robotlar için araştırma önerisi formunda taahhüt edilen hibrit mimari eksiksiz
gerçeklenmiştir:

| Katman | Yöntem | Dosya |
|---|---|---|
| Konumlandırma + Haritalama | **EKF-SLAM** (landmark tabanlı) + log-odds occupancy grid | `src/ekf_slam.py` |
| Sensör Füzyonu | Gürültülü odometri + LIDAR landmark gözlemleri (Kalman kazancı ile) | `src/ekf_slam.py` |
| Global Yol Planlama | **Dynamic A\*** (değişiklik/takılma tetiklemeli yeniden planlama, Öklidyen heuristic) | `src/planners.py` |
| Lokal Planlama / Engel Kaçınma | **DWA** (Dynamic Window Approach, dinamik engel tahmini ile) | `src/dwa.py` |
| Uyarlanabilir Katman | **PPO** (derin pekiştirmeli öğrenme) — DWA maliyet ağırlıklarını ortama göre uyarlar | `src/ppo.py` |
| Karşılaştırma Taban Çizgisi | Yapay Potansiyel Alan (APF) | `src/planners.py` |

## Kurulum

```bash
pip install numpy matplotlib pillow
```

(Yalnızca bu üç kütüphane gerekir; PPO dahil tüm algoritmalar saf NumPy ile yazılmıştır.)

## Kullanım

```bash
python main.py live         # CANLI + İNTERAKTİF mod (aşağıya bakın)
python main.py live 3       # 3. senaryoyu canlı izle (1-10 arası)
python main.py live 3 apf   # 3. senaryoda APF yöntemini izle (hybrid/classic/apf)
python main.py race         # YÖNTEM YARIŞI: hibrit vs APF yan yana
python main.py race 3 hybrid classic   # istediğin iki yöntemi yarıştır
python main.py demo         # Tek senaryoda 3 yöntemi hızlı test et
python main.py train        # PPO ajanını eğit (results/ppo_model.npz)
python main.py benchmark    # 10 senaryo x 10 koşum x 3 yöntem + grafikler
python main.py animate      # Simülasyon GIF'leri üret
python main.py all          # Hepsi sırayla
```

## Canlı Mod Kontrolleri (`python main.py live`)

| Kontrol | İşlev |
|---|---|
| SOL TIK | Yeni hedef — robot rotayı anında yeniden planlar |
| SHIFT + SOL TIK | Hedefi **görev listesine** ekler; robot sırayla ziyaret eder |
| SAĞ TIK | Haritaya yeni engel (kutu) ekler — Dynamic A* canlı yeniden planlar |
| ORTA TIK veya **H** | İmlecin olduğu yere yürüyen insan ekler |
| **D** | İmlece en yakın insanı siler |
| **B** | Bataryayı %25'e düşürür → robot görevi erteleyip **şarj istasyonuna** (⚡) gider, %95'e şarj olup göreve döner |

Sağ panelde robotun EKF-SLAM ile keşfettiği harita, konum tahmini ve landmark
tahminleri canlı olarak görüntülenir.

## Deney Sonuçları (10 senaryo × 10 koşum = yöntem başına 100 doğrulama)

| Yöntem | Başarı | Çarpışma | Ort. Süre (s) | Ort. Yol (m) | Ort. Enerji |
|---|---|---|---|---|---|
| **Hibrit (A\*+DWA+PPO)** | **%100** | **%0** | **29.1** | **20.4** | **5.28** |
| Klasik (A\*+DWA sabit) | %100 | %0 | 31.4 | 21.1 | 5.57 |
| APF (geleneksel) | %50 | %0 | 36.7 | 23.9 | 8.91 |

- Hibrit yöntem klasik ayara göre **%7.3 daha hızlı**, **%3.5 daha kısa yol**,
  **%5.2 daha az enerji**; süre değişkenliği yarı yarıya düşük
  (bkz. `results/fig6_uyarlanabilir_agirliklar.png`).
- Geleneksel APF, lokal minimum problemi nedeniyle senaryoların yarısında hedefe ulaşamaz.
- EKF-SLAM ortalama konum hatası: **0.19 m** (LIDAR gürültüsü 2 cm, odometri gürültülü).
- Form başarı ölçütü (≥%95 doğruluk, ≥10 senaryo, ≥100 doğrulama) **sağlanmıştır**.

## Klasör Yapısı

```
├── main.py                 # Ana çalıştırma betiği
├── src/
│   ├── environment.py      # Dinamik ofis ortamı, insanlar, LIDAR, robot modeli
│   ├── ekf_slam.py         # EKF-SLAM + occupancy grid haritalama
│   ├── planners.py         # Dynamic A* + APF
│   ├── dwa.py              # Dynamic Window Approach
│   ├── ppo.py              # PPO ajanı (saf NumPy aktör-kritik)
│   ├── simulation.py       # Simülasyon çekirdeği (tüm katmanların entegrasyonu)
│   ├── train_ppo.py        # PPO eğitim döngüsü
│   ├── benchmark.py        # Karşılaştırma deneyleri + grafikler
│   └── animate.py          # GIF animasyon üretici
└── results/
    ├── benchmark_results.csv         # 300 koşumun ham verisi
    ├── ppo_model.npz                 # Eğitilmiş PPO ağırlıkları
    ├── ppo_training_log.txt          # Eğitim günlüğü
    ├── fig1..fig6 *.png              # Karşılaştırma grafikleri
    └── sim_senaryo*.gif              # Simülasyon animasyonları
```
<img width="1810" height="983" alt="image" src="https://github.com/user-attachments/assets/a8e7306a-bf43-46be-ad5c-f5068e28d527" />

