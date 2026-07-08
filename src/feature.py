"""
Teknofest 2026 — Visual Odometry Pipeline
Özellik Çıkarımı ve Eşleştirme Modülü (feature.py)

İki karedeki "aynı fiziksel noktayı" bulmak. Pipeline'ın kalbi.
4 iş yapar: nokta tespit etme, eşleştirme, filtreleme, keyframe kararı.

Bileşenler:
    FeatureExtractor  — DISK + LightGlue ile özellik çıkarım ve eşleştirme
    is_keyframe()     — Karenin işlenmeye değer olup olmadığına karar verir
    VOInitializer     — Pipeline'ın ilk poz hesaplamasını güvenli başlatır

Bireysel test:
    python -m src.feature --video /home/mei/Benim/PROJECTS/Teknofest26/src/data/THYZ_2026_Ornek_Veri_1.MP4
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from lightglue import LightGlue, DISK
from lightglue.utils import numpy_image_to_torch, rbd


class FeatureExtractor:
    """
    DISK ile anahtar nokta tespiti, LightGlue ile eşleştirme,
    güvenilirlik ve akış tabanlı filtreleme.
    """

    def __init__(self, device="cuda", max_keypoints=2048,
                 confidence_threshold=0.7, flow_percentile=85):
        """
        DISK ve LightGlue modellerini yükler.

        Args:
            device:                'cuda' veya 'cpu'.
            max_keypoints:         Kare başına maksimum anahtar nokta sayısı.
            confidence_threshold:  Minimum eşleşme güven skoru.
            flow_percentile:       Medyan akış filtresinde tutulacak yüzdelik.
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.flow_percentile = flow_percentile

        print(f" DISK yükleniyor (max_keypoints={max_keypoints})...")
        self.extractor = DISK(max_num_keypoints=max_keypoints).eval().to(self.device)

        print(f" LightGlue yükleniyor (features='disk')...")
        self.matcher = LightGlue(features="disk").eval().to(self.device)

        print(f"    Modeller yüklendi → {self.device}")

    def extract(self, gray_frame):
        """
        Tek bir gri kareye DISK uygulayarak anahtar noktaları çıkarır.

        Args:
            gray_frame: Gri tonlamalı numpy array (HxW, uint8).

        Returns:
            feats: LightGlue'nun beklediği formatta dict.
                   Anahtarlar: keypoints (1,N,2), descriptors (1,N,128),
                   keypoint_scores (1,N), image_size (1,2)

        Süre: ~25ms/kare (GPU).
        """
        # Numpy → PyTorch tensor: (HxW, uint8) → (1, 1, H, W, float32, [0,1])
        img_tensor = numpy_image_to_torch(gray_frame).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(img_tensor)

        return feats

    def match(self, feats0, feats1):
        """
        İki karenin özelliklerini eşleştirir.

        Args:
            feats0: extract()'ın döndürdüğü dict (ilk kare).
            feats1: extract()'ın döndürdüğü dict (ikinci kare).

        Returns:
            pts0:   İlk karedeki eşleşen noktaların piksel koordinatları (M, 2) numpy.
            pts1:   İkinci karedeki karşılıkları (M, 2) numpy.
            scores: Her eşleşmenin güvenilirlik skoru (M,) numpy.

        Süre: ~50ms/kare çifti (GPU).
        """
        with torch.no_grad():
            result = self.matcher({"image0": feats0, "image1": feats1})

        # Batch boyutunu kaldır
        feats0_rb, feats1_rb, result_rb = [rbd(x) for x in [feats0, feats1, result]]

        matches = result_rb["matches"]      # (S, 2) — indeks çiftleri
        scores = result_rb["scores"]        # (S,)   — güven skorları

        # Eşleşme yoksa boş döndür
        if len(matches) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        # İndekslerle keypoint dizilerinden piksel koordinatlarını seç
        kpts0 = feats0_rb["keypoints"]   # (N, 2)
        kpts1 = feats1_rb["keypoints"]   # (N, 2)

        pts0 = kpts0[matches[..., 0]].cpu().numpy()   # (S, 2)
        pts1 = kpts1[matches[..., 1]].cpu().numpy()   # (S, 2)
        scores_np = scores.cpu().numpy()                # (S,)

        return pts0, pts1, scores_np

    def filter_by_confidence(self, pts0, pts1, scores):
        """
        Güvenilirlik skoru düşük olan eşleşmeleri atar.

        Args:
            pts0:   Nokta çiftleri (M, 2).
            pts1:   Nokta çiftleri (M, 2).
            scores: Güven skorları (M,).

        Returns:
            Filtrelenmiş (pts0, pts1, scores) — daha az eleman.
        """
        mask = scores >= self.confidence_threshold
        return pts0[mask], pts1[mask], scores[mask]

    def filter_by_flow(self, pts0, pts1):
        """
        Hareketli nesnelerden kaynaklanan eşleşmeleri atar.

        Medyan optik akış filtresi: kameranın genel hareketinden
        en çok sapan %15'i atar (muhtemelen hareketli nesneler).

        Args:
            pts0: Nokta çiftleri (M, 2).
            pts1: Nokta çiftleri (M, 2).

        Returns:
            Filtrelenmiş (pts0, pts1) — daha az eleman.
        """
        if len(pts0) < 4:
            return pts0, pts1

        # Her eşleşmenin hareket vektörü
        flow = pts1 - pts0  # (M, 2)

        # Medyan — "kameranın genel hareketi" (aykırı değerlere dayanıklı)
        median_flow = np.median(flow, axis=0)  # (2,)

        # Her noktanın medyandan sapması
        deviation = np.linalg.norm(flow - median_flow, axis=1)  # (M,)

        # En sapan noktaları at
        threshold = np.percentile(deviation, self.flow_percentile)
        mask = deviation <= threshold

        return pts0[mask], pts1[mask]

    def filter_matches(self, pts0, pts1, scores):
        """
        Güvenilirlik ve akış filtrelerini sırayla uygular.

        Args:
            pts0:   Ham eşleşme çiftleri (M, 2).
            pts1:   Ham eşleşme çiftleri (M, 2).
            scores: Güven skorları (M,).

        Returns:
            Temizlenmiş (pts0, pts1).
        """
        # 1. Güvenilirlik filtresi
        pts0, pts1, scores = self.filter_by_confidence(pts0, pts1, scores)

        # Çok az nokta kaldıysa akış filtresini atla (medyan güvenilmez)
        if len(pts0) < 10:
            return pts0, pts1

        # 2. Akış filtresi
        pts0, pts1 = self.filter_by_flow(pts0, pts1)

        return pts0, pts1



## Keyframe Kararı (sınıf dışı fonksiyon)

def is_keyframe(pts0, pts1, min_matches=30, min_motion_px=2.5):
    """
    Gelen karenin işlenmeye değer olup olmadığına karar verir.

    Args:
        pts0:          Filtrelenmiş eşleşme çiftleri (M, 2).
        pts1:          Filtrelenmiş eşleşme çiftleri (M, 2).
        min_matches:   Minimum eşleşme sayısı.
        min_motion_px: Minimum medyan piksel yer değişimi.

    Returns:
        (karar, sebep): bool ve açıklama string'i.
    """
    # Yeterli eşleşme var mı?
    if len(pts0) < min_matches:
        return False, f"insufficient_matches ({len(pts0)}<{min_matches})"

    # Piksel yer değişimlerini hesapla
    displacements = np.linalg.norm(pts1 - pts0, axis=1)  # (M,)
    median_disp = float(np.median(displacements))

    # Yeterli hareket var mı?
    if median_disp < min_motion_px:
        return False, f"low_motion ({median_disp:.1f}px<{min_motion_px}px)"

    return True, f"accepted (median={median_disp:.1f}px, matches={len(pts0)})"



## VO Başlatıcı

class VOInitializer:
    """
    Pipeline'ın ilk poz hesaplamasını güvenli şekilde başlatır.
    Yeterli hareket (parallax) ve geçerli geometri olmadan
    poz hesaplaması başlatmaz.
    """

    def __init__(self, parallax_threshold=15.0, min_matches=30):
        """
        Args:
            parallax_threshold: Başlatma için minimum medyan piksel yer değişimi.
            min_matches:        Minimum eşleşme sayısı.
        """
        self.parallax_threshold = parallax_threshold
        self.min_matches = min_matches
        self.initialized = False

    def try_initialize(self, pts0, pts1, K):
        """
        İlk karelerde yeterli hareket olup olmadığını kontrol eder.

        Args:
            pts0: Eşleşme çiftleri (M, 2).
            pts1: Eşleşme çiftleri (M, 2).
            K:    3×3 kamera iç parametre matrisi.

        Returns:
            bool — başlatıldı mı?
        """
        if self.initialized:
            return True

        # Yeterli eşleşme?
        if len(pts0) < self.min_matches:
            return False

        # Yeterli paralaks?
        displacements = np.linalg.norm(pts1 - pts0, axis=1)
        median_disp = float(np.median(displacements))

        if median_disp < self.parallax_threshold:
            return False

        # Triangülasyon kontrolü: Essential Matrix → recoverPose → 3D
        # noktaların kaçı kameranın önünde (pozitif derinlik)?
        E, mask_e = cv2.findEssentialMat(
            pts0, pts1, K,
            method=cv2.USAC_MAGSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None:
            return False

        # recoverPose: R, t hesapla + kaç nokta kameranın önünde
        n_infront, R, t, mask_rp = cv2.recoverPose(E, pts0, pts1, K)

        # %70'den fazlası öndeyse geometri doğru
        infront_ratio = n_infront / len(pts0)
        if infront_ratio < 0.70:
            return False

        self.initialized = True
        print(f"    VO başlatıldı: paralaks={median_disp:.1f}px, "
              f"önde={infront_ratio:.0%}")
        return True



## BİREYSEL TEST

if __name__ == "__main__":
    import argparse
    import time

    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from config import (
        CAMERA_K, DIST_COEFFS, IMAGE_SIZE,
        CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, BLUR_KERNEL,
        MAX_KEYPOINTS, CONFIDENCE_THRESHOLD, FLOW_PERCENTILE,
        MIN_MATCHES, MIN_MOTION_PX, PARALLAX_THRESHOLD,
        DEFAULT_VIDEO_PATH, CAMERA_PROFILE,
    )
    from src.utils import load_video, iter_frames, draw_matches, Timer
    from src.preprocessing import Preprocessor

    parser = argparse.ArgumentParser(description="feature.py — Bireysel Test")
    parser.add_argument(
        "--video", type=str, default=DEFAULT_VIDEO_PATH,
        help="Test videosu yolu",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Cihaz: cuda veya cpu",
    )
    parser.add_argument(
        "--frames", type=int, default=5,
        help="Kaç kare çifti test edilecek",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  feature.py — Bireysel Test")
    print(f"  Cihaz: {args.device} | Profil: {CAMERA_PROFILE}")
    print("=" * 60)

    # Preprocessor
    prep = Preprocessor(
        K=CAMERA_K, dist_coeffs=DIST_COEFFS, image_size=IMAGE_SIZE,
        clahe_clip=CLAHE_CLIP_LIMIT, clahe_tile=CLAHE_TILE_SIZE,
        blur_kernel=BLUR_KERNEL,
    )

    # Feature Extractor
    fe = FeatureExtractor(
        device=args.device,
        max_keypoints=MAX_KEYPOINTS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        flow_percentile=FLOW_PERCENTILE,
    )

    # VO Initializer
    initializer = VOInitializer(
        parallax_threshold=PARALLAX_THRESHOLD,
        min_matches=MIN_MATCHES,
    )

    # Video aç
    cap, frame_count, size, fps = load_video(args.video)
    K_new = prep.get_K_new()

    print(f"\n İlk {args.frames + 1} kare işleniyor...")

    prev_feats = None
    prev_enhanced = None

    for fid, frame in iter_frames(cap):
        if fid > args.frames:
            break

        # Ön işleme
        enhanced, undistorted = prep.preprocess(frame)

        # Özellik çıkarımı
        with Timer("extract") as t_ext:
            feats = fe.extract(enhanced)

        n_kpts = feats["keypoints"].shape[1]
        print(f"\n   Kare {fid}: {n_kpts} keypoint [{t_ext.elapsed*1000:.0f}ms]")

        # İlk kare — eşleştirecek önceki yok
        if prev_feats is None:
            prev_feats = feats
            prev_enhanced = enhanced
            continue

        # Eşleştirme
        with Timer("match") as t_match:
            pts0, pts1, scores = fe.match(prev_feats, feats)
        print(f"   Ham eşleşme : {len(pts0)} [{t_match.elapsed*1000:.0f}ms]")

        # Filtreleme
        with Timer("filter") as t_filt:
            pts0_clean, pts1_clean = fe.filter_matches(pts0, pts1, scores)
        print(f"   Temiz eşleşme: {len(pts0_clean)} [{t_filt.elapsed*1000:.0f}ms]")

        # Keyframe kararı
        kf, reason = is_keyframe(
            pts0_clean, pts1_clean,
            min_matches=MIN_MATCHES,
            min_motion_px=MIN_MOTION_PX,
        )
        emoji = "" if kf else "⏭️"
        print(f"   Keyframe     : {emoji} {reason}")

        # Initialization kontrolü
        if not initializer.initialized and kf:
            init_ok = initializer.try_initialize(pts0_clean, pts1_clean, K_new)
            if init_ok:
                print(f"   VO Init      :  Başlatıldı!")
            else:
                print(f"   VO Init      : ⏳ Henüz yeterli paralaks yok")

        # İlk eşleşmeyi görselleştir ve kaydet
        if fid == 1 and len(pts0_clean) > 0:
            canvas = draw_matches(
                prev_enhanced, enhanced,
                pts0_clean, pts1_clean,
                max_draw=80,
                output_path="debug_eslesmeler.png",
            )

        # Sonraki iterasyon için güncelle (keyframe ise)
        if kf:
            prev_feats = feats
            prev_enhanced = enhanced

    print(f"\n{'=' * 60}")
    print("   Feature testi tamamlandı")
    print("=" * 60)