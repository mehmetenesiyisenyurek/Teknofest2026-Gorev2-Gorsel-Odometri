"""
Teknofest 2026 — Visual Odometry Pipeline
Ön İşleme Modülü (preprocessing.py)

Kameradan gelen ham kareyi alıp geometrik ve görsel olarak düzeltir.
Sonraki tüm adımlar bu düzeltilmiş kareyi kullanır.

İşlem zinciri:
    Ham BGR kare
      → undistort  (lens bozulmasını düzelt)
      → enhance    (CLAHE + Gaussian Blur)

İki çıktı üretir:
    1. enhanced   : Gri tonlamalı, düzeltilmiş → feature.py (DISK gri ister)
    2. undistorted: BGR renkli, düzeltilmiş     → depth.py  (Metric3D renkli ister)

Bireysel test:
    python -m src.preprocessing --video /data/THYZ_2026_Ornek_Veri_1.MP4
"""

import sys
from pathlib import Path

import cv2
import numpy as np


class Preprocessor:
    """
    Tüm ön işleme adımlarını tek sınıfta toplar.

    Harita hesaplama ve CLAHE objesi oluşturma maliyetli olduğundan,
    bir kez __init__'te yapılır ve her kareye tekrar kullanılır.
    """

    def __init__(self, K, dist_coeffs, image_size, clahe_clip=2.0,
                 clahe_tile=(8, 8), blur_kernel=3):
        """
        Ön işleme için gerekli nesneleri bir kez hazırlar.

        Args:
            K:            3×3 kamera iç parametre matrisi (numpy).
            dist_coeffs:  5 elemanlı distorsiyon katsayıları (numpy).
            image_size:   (genişlik, yükseklik) tuple.
            clahe_clip:   CLAHE clip limit.
            clahe_tile:   CLAHE tile grid boyutu.
            blur_kernel:  Gaussian Blur kernel boyutu (tek sayı).
        """
        self.K = K.copy()
        self.dist_coeffs = dist_coeffs.copy()
        self.image_size = image_size
        self.blur_kernel = blur_kernel

        w, h = image_size

        # ── Undistortion haritası (bir kez hesapla, her kareye uygula) ──
        # alpha=0: siyah kenarları kırpar, tüm pikseller geçerli olur
        self.K_new, self.roi = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h)
        )

        # Remap haritası: "her piksel düzeltilmiş görüntüde nereye gidecek"
        # cv2.remap(), cv2.undistort()'tan 3-5x daha hızlıdır
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, dist_coeffs, None, self.K_new, (w, h), cv2.CV_16SC2
        )

        # ── CLAHE objesi ──
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=clahe_tile,
        )

        print(f"🔧 Preprocessor hazır:")
        print(f"   Görüntü boyutu : {w}×{h}")
        print(f"   CLAHE           : clip={clahe_clip}, tile={clahe_tile}")
        print(f"   Blur kernel     : {blur_kernel}")

    def undistort(self, frame):
        """
        Lens bozulmasını düzeltir.

        Args:
            frame: BGR renkli kare (numpy, HxWx3).

        Returns:
            Düzeltilmiş BGR kare (numpy, HxWx3).

        Süre: ~1ms/kare.
        """
        return cv2.remap(
            frame, self.map1, self.map2, cv2.INTER_LINEAR
        )

    def enhance(self, frame):
        """
        Görüntünün kontrastını artırır ve gürültüsünü azaltır.

        Args:
            frame: BGR veya gri tonlamalı kare (numpy).

        Returns:
            Gri tonlamalı, iyileştirilmiş kare (numpy, HxW, uint8).

        Süre: ~2ms/kare.
        """
        # BGR ise gri tonlamaya çevir
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # CLAHE — her blokta ayrı histogram eşitleme, bloklar arası yumuşatma
        enhanced = self.clahe.apply(gray)

        # Hafif Gaussian Blur — sensör gürültüsünü bastırır, detay kaybetmez
        if self.blur_kernel > 1:
            enhanced = cv2.GaussianBlur(
                enhanced, (self.blur_kernel, self.blur_kernel), 0
            )

        return enhanced

    def preprocess(self, frame):
        """
        Tam ön işleme zincirini çalıştırır: undistort → enhance.

        Args:
            frame: Ham BGR kare (numpy, HxWx3).

        Returns:
            enhanced:    Gri tonlamalı, düzeltilmiş, iyileştirilmiş kare
                         → feature.py kullanacak (DISK gri ister)
            undistorted: BGR renkli, düzeltilmiş ama iyileştirilmemiş kare
                         → depth.py kullanacak (Metric3D renkli ister)
        """
        undistorted = self.undistort(frame)
        enhanced = self.enhance(undistorted)
        return enhanced, undistorted

    def get_K_new(self):
        """
        Düzeltme sonrası geçerli olan yeni K matrisini döndürür.

        Undistort yaptıktan sonra orijinal K artık geçerli değil.
        geometry.py'daki tüm hesaplamalarda bu K_new kullanılmalı.

        Returns:
            K_new: 3×3 numpy matris.
        """
        return self.K_new.copy()



## BİREYSEL TEST

if __name__ == "__main__":
    import argparse
    import time

    # Kök dizini sys.path'e ekle
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from config import (
        CAMERA_K, DIST_COEFFS, IMAGE_SIZE,
        CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, BLUR_KERNEL,
        DEFAULT_VIDEO_PATH, CAMERA_PROFILE,
    )
    from src.utils import load_video, iter_frames, Timer

    parser = argparse.ArgumentParser(description="preprocessing.py — Bireysel Test")
    parser.add_argument(
        "--video", type=str, default=DEFAULT_VIDEO_PATH,
        help="Test videosu yolu",
    )
    parser.add_argument(
        "--frames", type=int, default=5,
        help="Kaç kare işlenecek",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  preprocessing.py — Bireysel Test")
    print(f"  Kamera profili: {CAMERA_PROFILE}")
    print("=" * 60)

    # Preprocessor oluştur
    prep = Preprocessor(
        K=CAMERA_K,
        dist_coeffs=DIST_COEFFS,
        image_size=IMAGE_SIZE,
        clahe_clip=CLAHE_CLIP_LIMIT,
        clahe_tile=CLAHE_TILE_SIZE,
        blur_kernel=BLUR_KERNEL,
    )

    # K_new'i göster
    print(f"\n Orijinal K matrisi:")
    for row in CAMERA_K:
        print(f"   [{row[0]:>10.2f}  {row[1]:>8.2f}  {row[2]:>10.2f}]")

    K_new = prep.get_K_new()
    print(f"\n Yeni K matrisi (undistort sonrası):")
    for row in K_new:
        print(f"   [{row[0]:>10.2f}  {row[1]:>8.2f}  {row[2]:>10.2f}]")

    # Video aç
    cap, frame_count, size, fps = load_video(args.video)

    print(f"\n İlk {args.frames} kare işleniyor...")
    times_undistort = []
    times_enhance = []
    times_total = []

    for fid, frame in iter_frames(cap):
        if fid >= args.frames:
            break

        # Adım adım zamanla
        with Timer("undistort") as t1:
            undistorted = prep.undistort(frame)
        times_undistort.append(t1.elapsed)

        with Timer("enhance") as t2:
            enhanced = prep.enhance(undistorted)
        times_enhance.append(t2.elapsed)

        total = t1.elapsed + t2.elapsed
        times_total.append(total)

        print(f"   Kare {fid:>4d}: "
              f"ham={frame.shape} → "
              f"undist={undistorted.shape} → "
              f"enhanced={enhanced.shape}  "
              f"[{total * 1000:.1f}ms]")

        # İlk kareyi kaydet
        if fid == 0:
            cv2.imwrite("debug_ham.png", frame)
            cv2.imwrite("debug_undistorted.png", undistorted)
            cv2.imwrite("debug_enhanced.png", enhanced)

            # Fark görselleştirme: ham vs düzeltilmiş
            ham_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(ham_gray, enhanced)
            diff_colored = cv2.applyColorMap(diff * 3, cv2.COLORMAP_JET)
            cv2.imwrite("debug_fark.png", diff_colored)

            # Yan yana karşılaştırma
            ham_resized = cv2.resize(ham_gray, (enhanced.shape[1], enhanced.shape[0]))
            comparison = np.hstack([
                cv2.cvtColor(ham_resized, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                diff_colored,
            ])
            cv2.imwrite("debug_karsilastirma.png", comparison)

            print(f"\n    Kaydedilen dosyalar:")
            print(f"      debug_ham.png           — Orijinal kare")
            print(f"      debug_undistorted.png    — Lens düzeltme sonrası")
            print(f"      debug_enhanced.png       — CLAHE + Blur sonrası")
            print(f"      debug_fark.png           — Ham vs İşlenmiş farkı")
            print(f"      debug_karsilastirma.png  — Yan yana karşılaştırma")

    # Zamanlama özeti
    print(f"\n  Zamanlama Özeti ({len(times_total)} kare):")
    print(f"   Undistort : ort={np.mean(times_undistort)*1000:.1f}ms")
    print(f"   Enhance   : ort={np.mean(times_enhance)*1000:.1f}ms")
    print(f"   Toplam    : ort={np.mean(times_total)*1000:.1f}ms/kare")

    print(f"\n{'=' * 60}")
    print("   Preprocessing testi tamamlandı")
    print("=" * 60)