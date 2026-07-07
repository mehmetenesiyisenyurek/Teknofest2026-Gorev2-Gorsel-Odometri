"""
Teknofest 2026 — Visual Odometry Pipeline
Yardımcı Fonksiyonlar (utils.py)

Hiçbir bileşene özgü olmayan genel yardımcı işlevler:
  - Video okuma ve kare iterasyonu
  - Eşleşme görselleştirme
  - Trajectory çizimi
  - JSON çıktı üretme (yarışma formatı)
  - ATE (Absolute Trajectory Error) hesaplama

Bireysel test:
    python -m src.utils --video /data/THYZ_2026_Ornek_Veri_1.MP4
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np



## Video İşlemleri

# Videoya ait özellikleri döndürür.
def load_video(path: str):
    """
    Video dosyasını açar, bilgilerini döndürür.

    Args:
        path: Video dosyasının yolu.

    Returns:
        cap:         cv2.VideoCapture objesi
        frame_count: Toplam kare sayısı (int)
        size:        (genişlik, yükseklik) tuple
        fps:         Saniyedeki kare sayısı (float)

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        RuntimeError:      Video açılamazsa.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video dosyası bulunamadı: {path}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Video yüklendi: {p.name}")
    print(f"   Boyut       : {width}×{height}")
    print(f"   Kare sayısı : {frame_count}")
    print(f"   FPS         : {fps:.2f}")
    print(f"   Süre        : {frame_count / fps:.1f}s")

    return cap, frame_count, (width, height), fps

# Videonun kare sayısını döndürür.
def iter_frames(cap):
    """
    Kareleri sırayla döndüren generator.

    Args:
        cap: cv2.VideoCapture objesi (load_video'dan).

    Yields:
        (frame_id, frame): Kare numarası (0-indexed) ve BGR numpy array.
    """
    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_id, frame
            frame_id += 1
    finally:
        cap.release()



# Görselleştirme

# Eşleşmeleri çizer
def draw_matches(img0, img1, pts0, pts1, max_draw=50, output_path=None):
    """
    İki karedeki eşleşmeleri görselleştirir.

    Args:
        img0:        İlk kare (numpy, HxW veya HxWx3).
        img1:        İkinci kare (numpy, HxW veya HxWx3).
        pts0:        İlk karedeki noktalar (numpy, Mx2).
        pts1:        İkinci karedeki noktalar (numpy, Mx2).
        max_draw:    En fazla kaç eşleşme çizilecek.
        output_path: Varsa dosyaya kaydeder.

    Returns:
        canvas: Yan yana iki kare + eşleşme çizgileri (numpy, HxWx3).
    """
    # Gri ise BGR'ye çevir
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)

    # Yan yana birleştir
    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1

    # Çizilecek eşleşme sayısını sınırla
    n = min(len(pts0), max_draw)
    if n < len(pts0):
        indices = np.random.choice(len(pts0), n, replace=False)
    else:
        indices = np.arange(n)

    for i in indices:
        x0, y0 = int(pts0[i, 0]), int(pts0[i, 1])
        x1, y1 = int(pts1[i, 0]) + w0, int(pts1[i, 1])

        # Rastgele renk
        color = tuple(int(c) for c in np.random.randint(64, 255, 3))

        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 3, color, -1, cv2.LINE_AA)

    # Eşleşme sayısını yazdır
    cv2.putText(
        canvas,
        f"Eslesme: {len(pts0)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if output_path:
        cv2.imwrite(str(output_path), canvas)
        print(f" Eşleşme görseli kaydedildi: {output_path}")

    return canvas


def plot_trajectory(positions, ground_truth=None, output_path="trajectory.png"):
    """
    Hesaplanan trajectory'yi 2D olarak çizer.

    Args:
        positions:    Pozisyon listesi, her biri (x, y, z) veya Nx3 numpy.
        ground_truth: Opsiyonel referans pozisyonları (aynı formatta).
        output_path:  PNG dosya yolu.
    """
    # Lazy import — matplotlib her yerde kurulu olmayabilir
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.array(positions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Sol: X-Y (kuşbakışı) ---
    ax = axes[0]
    ax.plot(positions[:, 0], positions[:, 1], "b-", linewidth=0.8, label="Tahmin")
    ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10, label="Başlangıç")
    ax.plot(positions[-1, 0], positions[-1, 1], "rs", markersize=10, label="Bitiş")

    if ground_truth is not None:
        gt = np.array(ground_truth)
        ax.plot(gt[:, 0], gt[:, 1], "g--", linewidth=0.8, alpha=0.7, label="Ground Truth")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory — Kuşbakışı (X-Y)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Sağ: İrtifa (Z) profili ---
    ax2 = axes[1]
    frames = np.arange(len(positions))
    ax2.plot(frames, positions[:, 2], "r-", linewidth=0.8, label="Z (irtifa)")

    if ground_truth is not None:
        gt = np.array(ground_truth)
        ax2.plot(np.arange(len(gt)), gt[:, 2], "g--", linewidth=0.8, alpha=0.7, label="GT Z")

    ax2.set_xlabel("Kare")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("İrtifa Profili")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"📊 Trajectory grafiği kaydedildi: {output_path}")



## JSON Çıktı (Yarışma Formatı)


def export_json(poses, total_frames, output_path="output.json"):
    """
    Tüm karelerin X, Y, Z pozisyonlarını yarışma formatında JSON'a yazar.

    Beklenen çıktı formatı:
        {
            "frame_0": {"x": 0.0, "y": 0.0, "z": 0.0},
            "frame_1": {"x": 0.12, "y": -0.05, "z": 0.02},
            ...
        }

    Args:
        poses:        Sözlük {frame_id: 4x4 numpy matris} veya
                      {frame_id: (x, y, z)} tuple/array.
        total_frames: Toplam kare sayısı.
        output_path:  JSON dosya yolu.
    """
    result = {}
    last_known = np.zeros(3)  # Son bilinen pozisyon

    for fid in range(total_frames):
        key = f"frame_{fid}"

        if fid in poses and poses[fid] is not None:
            pose = poses[fid]

            # 4×4 matris ise pozisyonu çıkar
            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                pos = pose[:3, 3].copy()
            elif isinstance(pose, np.ndarray) and pose.shape == (3,):
                pos = pose.copy()
            elif isinstance(pose, (list, tuple)) and len(pose) == 3:
                pos = np.array(pose, dtype=np.float64)
            else:
                pos = last_known.copy()

            last_known = pos.copy()
        else:
            # İşlenmemiş kare → son bilinen pozisyonu kopyala
            pos = last_known.copy()

        result[key] = {
            "x": round(float(pos[0]), 6),
            "y": round(float(pos[1]), 6),
            "z": round(float(pos[2]), 6),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f" JSON çıktı kaydedildi: {output_path} ({total_frames} kare)")



## Metrikler


def compute_ate(predicted, ground_truth):
    """
    ATE (Absolute Trajectory Error) hesaplar.

    Her karede tahmin ile gerçek arasındaki Öklid mesafesini hesaplar,
    ortalamasını alır.

    Args:
        predicted:    Tahmin edilen pozisyonlar (Nx3 numpy).
        ground_truth: Referans pozisyonlar (Nx3 numpy).

    Returns:
        ate: Ortalama hata (float, metre).
    """
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)

    if predicted.shape != ground_truth.shape:
        min_len = min(len(predicted), len(ground_truth))
        predicted = predicted[:min_len]
        ground_truth = ground_truth[:min_len]
        print(f"⚠️  Uzunluk uyumsuzluğu, ilk {min_len} kare kullanıldı.")

    errors = np.linalg.norm(predicted - ground_truth, axis=1)
    ate = float(np.mean(errors))

    print(f"📏 ATE: {ate:.4f} m (medyan: {np.median(errors):.4f} m, max: {np.max(errors):.4f} m)")
    return ate


## Zamanlama Yardımcısı


class Timer:
    """Basit zamanlayıcı. Pipeline adımlarını ölçmek için."""

    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return f"{self.name}: {self.elapsed:.3f}s"


## BİREYSEL TEST

if __name__ == "__main__":
    import argparse

    # Kök dizini sys.path'e ekle (config import edebilmek için)
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from config import DEFAULT_VIDEO_PATH

    parser = argparse.ArgumentParser(description="utils.py — Bireysel Test")
    parser.add_argument(
        "--video",
        type=str,
        default=DEFAULT_VIDEO_PATH,
        help="Test videosu yolu",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  utils.py — Bireysel Test")
    print("=" * 60)

    # --- Test 1: Video okuma ---
    print("\n Test 1: Video Okuma")
    cap, frame_count, size, fps = load_video(args.video)

    # --- Test 2: İlk 5 kareyi oku ---
    print("\n Test 2: İlk 5 Kare")
    cap2, _, _, _ = load_video(args.video)
    for fid, frame in iter_frames(cap2):
        if fid >= 5:
            break
        print(f"   Kare {fid}: shape={frame.shape}, dtype={frame.dtype}, "
              f"min={frame.min()}, max={frame.max()}")

    # İlk kareyi kaydet
    cap3, _, _, _ = load_video(args.video)
    _, first_frame = next(iter_frames(cap3))
    cv2.imwrite("debug_ilk_kare.png", first_frame)
    print("    İlk kare kaydedildi: debug_ilk_kare.png")

    # --- Test 3: JSON export ---
    print("\n Test 3: JSON Export (sentetik veri)")
    dummy_poses = {}
    for i in range(10):
        T = np.eye(4)
        T[0, 3] = i * 0.5   # X
        T[1, 3] = i * 0.3   # Y
        T[2, 3] = 50.0       # Z sabit
        dummy_poses[i] = T

    export_json(dummy_poses, total_frames=10, output_path="debug_output.json")

    # --- Test 4: Trajectory çizimi ---
    print("\n Test 4: Trajectory Çizimi (sentetik veri)")
    positions = [(i * 0.5, i * 0.3, 50.0) for i in range(10)]
    plot_trajectory(positions, output_path="debug_trajectory.png")

    # --- Test 5: ATE ---
    print("\n Test 5: ATE Hesaplama (sentetik veri)")
    pred = np.array(positions)
    gt = pred + np.random.normal(0, 0.1, pred.shape)  # Hafif gürültü ekle
    compute_ate(pred, gt)

    # --- Test 6: Timer ---
    print("\n Test 6: Timer")
    with Timer("Uyku testi") as t:
        time.sleep(0.1)
    print(f"   {t}")

    print(f"\n{'=' * 60}")
    print("   Tüm testler tamamlandı")
    print("=" * 60)