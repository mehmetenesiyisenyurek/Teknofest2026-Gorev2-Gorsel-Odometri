"""
Teknofest 2026 — Visual Odometry Pipeline
Derinlik/İrtifa Modülü (depth.py)

Metric3D v2 (vit_small) ile tek görüntüden metrik derinlik tahmini.

İKİ KULLANIM ALANI:
    1. İrtifa Tahmini (Birincil Yol):
        Derinlik haritasının medyanı → irtifa (metre)
        H ayrıştırmasından gelen normalize t'yi ölçeklemek için kullanılır.

    2. Derinlik Haritası (Yedek Yol — PnP):
        Piksel başına metre cinsinden derinlik → 2D noktaları 3D'ye backproject
        solvePnPRansac ile metrik yer değiştirme hesaplamak için kullanılır.

NOT: Yarışma formatında ölçek GT'den kalibre edilir.
Depth modelinin mutlak doğruluğu kritik değil —
göreceli irtifa değişimini (ΔZ) doğru yakalaması yeterli.

PERFORMANS:
    estimate_altitude_and_depth() metodu tek inference ile hem irtifa hem
    depth_map döndürür. Pipeline'da çift model çalıştırmayı önler.
    CPU'da ~14s, CUDA'da ~0.05s per frame.

Metric3D API:
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    Input:  RGB, resize (616, 1064), ImageNet normalization
    Output: depth_map → de-canonicalize → medyan → irtifa

Bireysel test:
    python -m src.depth --video /home/mei/Benim/PROJECTS/Teknofest26/src/data/THYZ_2026_Ornek_Veri_1.MP4
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class AltitudeEstimator:
    """
    Metric3D v2 ile irtifa ve derinlik tahmini.

    Akış:
        frame_bgr → RGB → resize (616×1064) → normalize
        → Metric3D → canonical depth → de-canonicalize (× fx_scaled/1000)
        → medyan → EMA yumuşatma → irtifa (metre)

    Pipeline Entegrasyonu:
        Birincil yol: altitude = estimator.estimate_altitude(frame)
        Yedek yol:    altitude, depth_map = estimator.estimate_altitude_and_depth(frame)
    """

    # Metric3D'nin beklediği girdi boyutu
    INPUT_HEIGHT = 616
    INPUT_WIDTH = 1064

    # ImageNet normalizasyon değerleri
    MEAN = torch.tensor([123.675, 116.28, 103.53])
    STD = torch.tensor([58.395, 57.12, 57.375])

    def __init__(self, device="cuda", fx=1389.7, ema_alpha=0.2,
                 min_altitude=5.0, max_altitude=500.0):
        """
        Args:
            device:       'cuda' veya 'cpu'.
            fx:           Kameranın focal length (piksel). De-canonicalize için gerekli.
            ema_alpha:    EMA katsayısı (0-1). Yüksek = yeni değere çok ağırlık.
            min_altitude: Minimum fiziksel irtifa sınırı (metre).
            max_altitude: Maksimum fiziksel irtifa sınırı (metre).
        """
        self.device = torch.device(device)
        self.fx_original = fx
        self.ema_alpha = ema_alpha
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude

        # EMA state
        self._smoothed_altitude = None

        # Model yükle
        print(f" Metric3D v2 yükleniyor ({device})...")
        self.model = torch.hub.load(
            'yvanyin/metric3d', 'metric3d_vit_small', pretrain=True
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"    Metric3D yüklendi → {device}")

        # Normalize tensörleri GPU'ya taşı
        self.mean = self.MEAN.to(self.device).view(1, 3, 1, 1)
        self.std = self.STD.to(self.device).view(1, 3, 1, 1)

    # ──────────────────────────────────────────────────────────
    # Ön İşleme
    # ──────────────────────────────────────────────────────────

    def _preprocess_for_metric3d(self, frame_bgr):
        """
        BGR frame'i Metric3D'nin beklediği formata dönüştürür.

        Adımlar:
            1. BGR → RGB
            2. Aspect ratio koruyarak resize (hedef: 616×1064)
            3. Sağa ve alta siyah padding
            4. Tensor'a çevir + ImageNet normalize
            5. De-canonicalize çarpanını hesapla

        Args:
            frame_bgr: (H, W, 3) BGR numpy array.

        Returns:
            input_tensor: (1, 3, 616, 1064) normalize edilmiş tensor.
            pad_info:     (pad_h, pad_w, new_h, new_w) — çıktıyı kırpmak için.
            scale:        fx_scaled / 1000 — de-canonicalize çarpanı.
        """
        # BGR → RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb.shape[:2]

        # ── Aspect ratio koruyarak resize ──
        # Metric3D 616×1064 bekliyor. Orijinal görüntüyü
        # bu aspect ratio'ya uyacak şekilde resize + pad yapıyoruz.
        target_h, target_w = self.INPUT_HEIGHT, self.INPUT_WIDTH

        # Scale factor: hedef boyuta sığdırmak için
        scale_h = target_h / h_orig
        scale_w = target_w / w_orig
        scale = min(scale_h, scale_w)

        new_h = int(h_orig * scale)
        new_w = int(w_orig * scale)

        rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad: sağa ve alta siyah padding
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        rgb_padded = cv2.copyMakeBorder(
            rgb_resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=0
        )

        # Tensor'a çevir: (H, W, 3) → (1, 3, H, W), float32
        tensor = torch.from_numpy(rgb_padded).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        # ImageNet normalize: (x - mean) / std
        tensor = (tensor - self.mean) / self.std

        # Focal length'i de-canonicalize için ölçekle
        # Metric3D canonical focal = 1000. Gerçek fx'i scale ile çarp.
        fx_scaled = self.fx_original * scale
        decanon_scale = fx_scaled / 1000.0

        return tensor, (pad_h, pad_w, new_h, new_w), decanon_scale

    # ──────────────────────────────────────────────────────────
    # Derinlik Haritası
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def estimate_depth_map(self, frame_bgr):
        """
        Tek bir frame'den metrik derinlik haritası üretir.

        Akış:
            preprocess → Metric3D inference → de-canonicalize
            → padding kırp → orijinal boyuta resize

        Args:
            frame_bgr: (H, W, 3) BGR numpy array.

        Returns:
            depth_map: (H_orig, W_orig) numpy array, metre cinsinden.
        """
        input_tensor, (pad_h, pad_w, new_h, new_w), decanon_scale = \
            self._preprocess_for_metric3d(frame_bgr)

        # ── Inference ──
        pred_depth, confidence = self.model.inference({'input': input_tensor})

        # ── De-canonicalize ──
        # Metric3D canonical space'de üretir (focal=1000 varsayımı).
        # Gerçek metrik derinlik = canonical_depth × (fx_scaled / 1000)
        pred_depth = pred_depth * decanon_scale

        # ── Padding'i kırp ──
        # pred_depth shape: (1, 1, 616, 1064)
        pred_depth = pred_depth[:, :, :new_h, :new_w]

        # ── Orijinal boyuta resize ──
        h_orig, w_orig = frame_bgr.shape[:2]
        pred_depth = F.interpolate(
            pred_depth, size=(h_orig, w_orig),
            mode='bilinear', align_corners=False
        )

        depth_map = pred_depth.squeeze().cpu().numpy()  # (H, W)

        return depth_map

    # ──────────────────────────────────────────────────────────
    # İrtifa Tahmini
    # ──────────────────────────────────────────────────────────

    def estimate_altitude(self, frame_bgr):
        """
        Tek bir frame'den drone irtifası tahmin eder.

        Derinlik haritasının medyanını alır (outlier'lara karşı robust)
        ve EMA ile yumuşatır.

        NOT: İçeride estimate_altitude_and_depth() çağırır.
        Eğer hem altitude hem depth_map gerekiyorsa,
        doğrudan estimate_altitude_and_depth() kullanın.

        Args:
            frame_bgr: (H, W, 3) BGR numpy array.

        Returns:
            altitude: Yumuşatılmış irtifa (metre, float).
        """
        altitude, _ = self.estimate_altitude_and_depth(frame_bgr)
        return altitude

    def estimate_altitude_and_depth(self, frame_bgr):
        """
        Tek bir frame'den hem irtifa hem derinlik haritası üretir.

        Metric3D inference'ı BİR KEZ çalıştırır ve her iki sonucu da döndürür.
        Pipeline'da hem irtifa (Homography ölçekleme) hem depth_map
        (PnP yedek yol) gerektiğinde çift inference'dan kaçınır.

        Akış:
            estimate_depth_map(frame) → depth_map
            medyan(depth_map) → raw_altitude
            fiziksel kısıtlar → clip
            EMA yumuşatma → altitude

        Args:
            frame_bgr: (H, W, 3) BGR numpy array.

        Returns:
            altitude:  Yumuşatılmış irtifa (metre, float).
            depth_map: (H, W) metre cinsinden derinlik haritası.
        """
        depth_map = self.estimate_depth_map(frame_bgr)

        # ── Medyan irtifa (robust) ──
        # Sıfır, negatif ve inf değerleri filtrele
        valid = depth_map[(depth_map > 0) & np.isfinite(depth_map)]

        if len(valid) == 0:
            # Hiç geçerli piksel yoksa son tahmini koru
            altitude = self._smoothed_altitude if self._smoothed_altitude else 50.0
            return altitude, depth_map

        raw_altitude = float(np.median(valid))

        # ── Fiziksel kısıtlar ──
        raw_altitude = np.clip(raw_altitude, self.min_altitude, self.max_altitude)

        # ── EMA yumuşatma ──
        altitude = self._smooth(raw_altitude)

        return altitude, depth_map

    # ──────────────────────────────────────────────────────────
    # EMA Yumuşatma
    # ──────────────────────────────────────────────────────────

    def _smooth(self, raw_altitude):
        """
        EMA yumuşatma + ani sıçrama koruması.

        Normal değişim: standart EMA (alpha=0.2)
        Ani sıçrama (>%50 sapma): düşürülmüş EMA (alpha=0.06)

        Args:
            raw_altitude: Ham irtifa tahmini (metre).

        Returns:
            Yumuşatılmış irtifa (metre).
        """
        if self._smoothed_altitude is None:
            self._smoothed_altitude = raw_altitude
            return raw_altitude

        # Ani sıçrama kontrolü: önceki değerden %50'den fazla sapma → ağırlık azalt
        ratio = raw_altitude / self._smoothed_altitude if self._smoothed_altitude > 0 else 1.0
        if ratio > 1.5 or ratio < 0.67:
            # Ani sıçrama — EMA katsayısını azalt
            alpha = self.ema_alpha * 0.3
        else:
            alpha = self.ema_alpha

        self._smoothed_altitude = alpha * raw_altitude + (1 - alpha) * self._smoothed_altitude

        return self._smoothed_altitude

    # ──────────────────────────────────────────────────────────
    # Yardımcı Metodlar
    # ──────────────────────────────────────────────────────────

    def get_last_altitude(self):
        """Son yumuşatılmış irtifayı döndürür."""
        return self._smoothed_altitude if self._smoothed_altitude else 50.0

    def reset(self):
        """EMA state'i sıfırlar (GT reset sonrası)."""
        self._smoothed_altitude = None


# ──────────────────────────────────────────────────────────────
# BİREYSEL TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from config import (
        CAMERA_K, DEFAULT_VIDEO_PATH, CAMERA_PROFILE,
    )
    from src.utils import load_video, iter_frames, Timer

    parser = argparse.ArgumentParser(description="depth.py — Bireysel Test")
    parser.add_argument(
        "--video", type=str, default=DEFAULT_VIDEO_PATH,
        help="Test videosu yolu",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Cihaz: cuda veya cpu",
    )
    parser.add_argument(
        "--max-frames", type=int, default=10,
        help="Test edilecek kare sayısı",
    )
    parser.add_argument(
        "--show-map", action="store_true",
        help="Derinlik haritasını görselleştir",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  depth.py — Bireysel Test (v2 — combined API)")
    print(f"  Cihaz: {args.device} | Profil: {CAMERA_PROFILE}")
    print("=" * 60)

    fx = float(CAMERA_K[0, 0])
    estimator = AltitudeEstimator(
        device=args.device, fx=fx,
        ema_alpha=0.2, min_altitude=5.0, max_altitude=500.0,
    )

    cap, frame_count, size, fps = load_video(args.video)

    print(f"\n🧪 BÖLÜM 1: İrtifa Tahmini (ilk {args.max_frames} kare)\n")

    altitudes = []
    times = []

    for fid, frame in iter_frames(cap):
        if fid >= args.max_frames:
            break

        with Timer(f"depth_frame_{fid}") as t:
            altitude = estimator.estimate_altitude(frame)

        altitudes.append(altitude)
        times.append(t.elapsed)

        print(f"   Kare {fid:>4d}: irtifa = {altitude:>7.2f}m  ({t.elapsed*1000:.0f}ms)")

    cap.release()

    if altitudes:
        print(f"\n📊 İstatistikler:")
        print(f"   Ortalama irtifa : {np.mean(altitudes):.2f}m")
        print(f"   Std sapma       : {np.std(altitudes):.2f}m")
        print(f"   Min / Max       : {np.min(altitudes):.2f}m / {np.max(altitudes):.2f}m")
        print(f"   Ort. süre       : {np.mean(times)*1000:.0f}ms/kare")

    # ── BÖLÜM 2: Combined API Doğrulaması ──
    print(f"\n{'─' * 50}")
    print(f"🧪 BÖLÜM 2: Combined API (estimate_altitude_and_depth)\n")

    # EMA'yı sıfırla
    estimator.reset()

    cap2, _, _, _ = load_video(args.video)

    for fid, frame in iter_frames(cap2):
        if fid >= 3:
            break

        with Timer("combined") as t1:
            alt_combined, dmap = estimator.estimate_altitude_and_depth(frame)

        print(f"   Kare {fid}:")
        print(f"     İrtifa:        {alt_combined:.2f}m")
        print(f"     Depth map:     shape={dmap.shape}, "
              f"min={dmap.min():.2f}m, max={dmap.max():.2f}m")
        print(f"     Geçerli %:     {(dmap > 0).sum() / dmap.size * 100:.1f}%")
        print(f"     Süre:          {t1.elapsed*1000:.0f}ms (tek inference)")

    cap2.release()

    print(f"\n{'=' * 60}")
    print("  ✅ Depth testi tamamlandı (v2 — combined API)")
    print("=" * 60)