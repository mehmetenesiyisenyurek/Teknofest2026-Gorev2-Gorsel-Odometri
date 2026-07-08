"""
Teknofest 2026 — Visual Odometry Pipeline
Geometri Modülü (geometry.py)

Eşleşen noktalardan kameranın nasıl hareket ettiğini hesaplar.
Pipeline'ın matematik kısmı.

Hafta 1: Sadece birincil yol (Homography) kodlanır.
Hafta 2: Yedek yol (Essential Matrix + PnP) eklenecek.

Birincil yol akışı:
    pts0, pts1
      → estimate_homography()   →  H, inlier_mask, inlier_ratio
      → is_planar()             →  düzlemsel mi?
      → decompose_homography()  →  R, t_norm, normal, success
      → check_rotation_only()   →  salt rotasyon mu?
      → compute_displacement()  →  dx, dy, dz (metre)

Bireysel test:
    python -m src.geometry --video /home/mei/Benim/PROJECTS/Teknofest26/src/data/THYZ_2026_Ornek_Veri_1.MP4
"""

import sys
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────
# Homography Tahmini
# ──────────────────────────────────────────────────────────────

def estimate_homography(pts0, pts1, magsac_threshold=3.0):
    """
    İki kare arasındaki düzlemsel dönüşümü (Homography) tahmin eder.

    Args:
        pts0:              İlk karedeki noktalar (M, 2) numpy.
        pts1:              İkinci karedeki noktalar (M, 2) numpy.
        magsac_threshold:  MAGSAC++ reprojection error eşiği (piksel).

    Returns:
        H:            3×3 Homography matrisi (numpy) veya None.
        inlier_mask:  Hangi eşleşmelerin modele uyduğu (M,) bool veya None.
        inlier_ratio: Uyumluluk oranı (float, 0-1).
    """
    if len(pts0) < 4:
        return None, None, 0.0

    H, mask = cv2.findHomography(
        pts0, pts1,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=magsac_threshold,
        maxIters=2000,
        confidence=0.999,
    )

    if H is None or mask is None:
        return None, None, 0.0

    inlier_mask = mask.ravel().astype(bool)
    inlier_ratio = float(np.sum(inlier_mask)) / len(inlier_mask)

    return H, inlier_mask, inlier_ratio


# ──────────────────────────────────────────────────────────────
# Düzlemsellik Kararı
# ──────────────────────────────────────────────────────────────

def is_planar(inlier_ratio, threshold=0.70):
    """
    Sahne düzlemsel mi değil mi sorusuna cevap verir.

    Args:
        inlier_ratio: Homography inlier oranı (float, 0-1).
        threshold:    Eşik değeri.

    Returns:
        bool: True ise birincil yol (Homography), False ise yedek yol.
    """
    return inlier_ratio >= threshold


# ──────────────────────────────────────────────────────────────
# Homography Ayrıştırma
# ──────────────────────────────────────────────────────────────

def decompose_homography(H, K):
    """
    H matrisini parçalayarak R (dönme) ve t (öteleme) çıkarır.

    cv2.decomposeHomographyMat 4 aday çözüm üretir.
    Fiziksel kısıtlarla (normal vektörü, determinant) doğru çözüm seçilir.

    Args:
        H: 3×3 Homography matrisi.
        K: 3×3 kamera iç parametre matrisi (undistort sonrası K_new).

    Returns:
        R:        3×3 rotasyon matrisi.
        t_norm:   3×1 normalize öteleme vektörü (irtifaya bölünmüş).
        normal:   3×1 zemin düzlemi normali.
        success:  bool — geçerli çözüm bulundu mu.
    """
    try:
        n_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return None, None, None, False

    if n_solutions == 0:
        return None, None, None, False

    # ── 4 aday çözümden doğrusunu seç ──
    best_score = -1.0
    best_R, best_t, best_n = None, None, None

    for i in range(n_solutions):
        R_cand = rotations[i]
        t_cand = translations[i]
        n_cand = normals[i]

        # Kontrol 1: Normal vektörü aşağıya bakıyor mu?
        # Kamera aşağı bakıyor → zemin normali kameraya doğru → n'in Z bileşeni > 0
        nz = float(n_cand[2, 0]) if n_cand.shape == (3, 1) else float(n_cand[2])
        if nz < 0.3:
            continue

        # Kontrol 2: Rotasyon matrisi geçerli mi? (det(R) ≈ 1)
        det = np.linalg.det(R_cand)
        if abs(det - 1.0) > 0.05:
            continue

        # Kontrol 3: R ortogonal mi? (R @ R^T ≈ I)
        ortho_err = np.linalg.norm(R_cand @ R_cand.T - np.eye(3))
        if ortho_err > 0.05:
            continue

        # Skor: normal vektörünün Z bileşeni en büyük olan = en "aşağı bakan"
        score = nz
        if score > best_score:
            best_score = score
            best_R = R_cand
            best_t = t_cand
            best_n = n_cand

    if best_R is None:
        return None, None, None, False

    # t'yi (3,) vektöre çevir
    t_norm = best_t.flatten()
    normal_vec = best_n.flatten()

    return best_R, t_norm, normal_vec, True


# ──────────────────────────────────────────────────────────────
# Salt Rotasyon Tespiti
# ──────────────────────────────────────────────────────────────

def check_rotation_only(R, pts0, pts1, K, threshold_px=1.5):
    """
    Kameranın sadece döndüğü (öteleme yapmadığı) durumu tespit eder.

    Eğer kamera sadece dönseydi noktalar nerede olurdu?
    Bu tahmin ile gerçek noktalar arasındaki fark küçükse → salt rotasyon.

    Args:
        R:            3×3 rotasyon matrisi.
        pts0:         İlk karedeki noktalar (M, 2).
        pts1:         İkinci karedeki noktalar (M, 2).
        K:            3×3 kamera matrisi.
        threshold_px: Medyan piksel hatası eşiği.

    Returns:
        bool: True ise salt rotasyon (öteleme yok).
    """
    if len(pts0) < 4:
        return False

    K_inv = np.linalg.inv(K)

    # Piksel → normalize koordinatlar: x_norm = K^-1 × [x, y, 1]^T
    ones = np.ones((len(pts0), 1))
    pts0_h = np.hstack([pts0, ones])  # (M, 3)

    # Normalize koordinatlara çevir
    pts0_norm = (K_inv @ pts0_h.T).T  # (M, 3)

    # R uygula: "sadece dönseydi noktalar nerede olurdu?"
    pts0_rotated = (R @ pts0_norm.T).T  # (M, 3)

    # Tekrar piksel koordinatlarına çevir
    pts0_reproj = (K @ pts0_rotated.T).T  # (M, 3)
    pts0_reproj = pts0_reproj[:, :2] / pts0_reproj[:, 2:3]  # (M, 2) — perspektif bölme

    # Döndürülmüş noktalar ile gerçek noktalar arasındaki fark
    errors = np.linalg.norm(pts0_reproj - pts1, axis=1)  # (M,)
    median_error = float(np.median(errors))

    return median_error < threshold_px


# ──────────────────────────────────────────────────────────────
# Yer Değiştirme Hesaplama
# ──────────────────────────────────────────────────────────────

def compute_displacement(t_normalized, altitude, prev_altitude):
    """
    Normalize ötelemeyi ve irtifayı kullanarak metre cinsinden ΔX, ΔY, ΔZ hesaplar.

    X, Y: Homography'den gelen normalize öteleme × irtifa
    Z:    İrtifa farkı (derinlik modelinden)

    Her ekseni kendi en güvenilir kaynağından beslemek hataları izole eder.

    Args:
        t_normalized:  3 elemanlı normalize öteleme vektörü (H ayrıştırmasından).
        altitude:      Mevcut irtifa (float, metre).
        prev_altitude: Önceki keyframe'in irtifası (float, metre).

    Returns:
        dx, dy, dz: Metre cinsinden yer değiştirme.
    """
    dx = float(t_normalized[0]) * altitude
    dy = float(t_normalized[1]) * altitude
    dz = altitude - prev_altitude

    return dx, dy, dz


# ──────────────────────────────────────────────────────────────
# BİREYSEL TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import torch
    from config import (
        CAMERA_K, DIST_COEFFS, IMAGE_SIZE,
        CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, BLUR_KERNEL,
        MAX_KEYPOINTS, CONFIDENCE_THRESHOLD, FLOW_PERCENTILE,
        MIN_MATCHES, MIN_MOTION_PX,
        HOMOGRAPHY_INLIER_THRESHOLD, ROTATION_ONLY_THRESHOLD_PX,
        MAGSAC_THRESHOLD,
        DEFAULT_VIDEO_PATH, CAMERA_PROFILE,
    )
    from src.utils import load_video, iter_frames, Timer
    from src.preprocessing import Preprocessor
    from src.feature import FeatureExtractor, is_keyframe

    parser = argparse.ArgumentParser(description="geometry.py — Bireysel Test")
    parser.add_argument(
        "--video", type=str, default=DEFAULT_VIDEO_PATH,
        help="Test videosu yolu",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Cihaz: cuda veya cpu",
    )
    parser.add_argument(
        "--max-frames", type=int, default=60,
        help="Maksimum taranacak kare sayısı",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  geometry.py — Bireysel Test")
    print(f"  Cihaz: {args.device} | Profil: {CAMERA_PROFILE}")
    print("=" * 60)

    # Modüller
    prep = Preprocessor(
        K=CAMERA_K, dist_coeffs=DIST_COEFFS, image_size=IMAGE_SIZE,
        clahe_clip=CLAHE_CLIP_LIMIT, clahe_tile=CLAHE_TILE_SIZE,
        blur_kernel=BLUR_KERNEL,
    )
    fe = FeatureExtractor(
        device=args.device, max_keypoints=MAX_KEYPOINTS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        flow_percentile=FLOW_PERCENTILE,
    )
    K_new = prep.get_K_new()

    cap, frame_count, size, fps = load_video(args.video)

    print(f"\n🧪 İlk keyframe çiftini arıyorum (maks {args.max_frames} kare)...\n")

    prev_feats = None
    prev_enhanced = None
    found_keyframe = False

    for fid, frame in iter_frames(cap):
        if fid >= args.max_frames:
            print(f"\n   ⚠️  {args.max_frames} kare tarandı, keyframe bulunamadı.")
            break

        enhanced, undistorted = prep.preprocess(frame)
        feats = fe.extract(enhanced)

        if prev_feats is None:
            prev_feats = feats
            prev_enhanced = enhanced
            continue

        pts0, pts1, scores = fe.match(prev_feats, feats)
        pts0_c, pts1_c = fe.filter_matches(pts0, pts1, scores)

        kf, reason = is_keyframe(pts0_c, pts1_c, MIN_MATCHES, MIN_MOTION_PX)
        if not kf:
            continue

        found_keyframe = True
        print(f"   ✅ Keyframe bulundu: kare 0 ↔ kare {fid} ({reason})")

        # ── Homography ──
        with Timer("homography") as t_h:
            H, mask, ratio = estimate_homography(pts0_c, pts1_c, MAGSAC_THRESHOLD)

        if H is None:
            print(f"   ❌ Homography bulunamadı")
            break

        print(f"\n📐 Homography Sonuçları:")
        print(f"   Inlier oranı  : {ratio:.2%}")
        print(f"   Düzlemsel mi?  : {'✅ Evet' if is_planar(ratio, HOMOGRAPHY_INLIER_THRESHOLD) else '❌ Hayır'}")
        print(f"   Süre          : {t_h.elapsed*1000:.1f}ms")

        # H matrisini göster
        print(f"\n   H matrisi:")
        for row in H:
            print(f"     [{row[0]:>12.6f}  {row[1]:>12.6f}  {row[2]:>12.6f}]")

        # ── H Ayrıştırma ──
        if is_planar(ratio, HOMOGRAPHY_INLIER_THRESHOLD):
            with Timer("decompose") as t_d:
                R, t_norm, normal, success = decompose_homography(H, K_new)

            if success:
                print(f"\n🔄 H Ayrıştırma Sonuçları:")
                print(f"   Başarılı      : ✅")
                print(f"   Süre          : {t_d.elapsed*1000:.1f}ms")

                # Rotasyon (Euler açılarına çevir)
                from scipy.spatial.transform import Rotation as Rot
                euler = Rot.from_matrix(R).as_euler("xyz", degrees=True)
                print(f"\n   Rotasyon (derece):")
                print(f"     Roll  (X): {euler[0]:>8.3f}°")
                print(f"     Pitch (Y): {euler[1]:>8.3f}°")
                print(f"     Yaw   (Z): {euler[2]:>8.3f}°")

                print(f"\n   Öteleme (normalize):")
                print(f"     tx: {t_norm[0]:>10.6f}")
                print(f"     ty: {t_norm[1]:>10.6f}")
                print(f"     tz: {t_norm[2]:>10.6f}")

                print(f"\n   Zemin normali:")
                print(f"     nx: {normal[0]:>10.6f}")
                print(f"     ny: {normal[1]:>10.6f}")
                print(f"     nz: {normal[2]:>10.6f}")

                # Salt rotasyon kontrolü
                rot_only = check_rotation_only(
                    R, pts0_c, pts1_c, K_new, ROTATION_ONLY_THRESHOLD_PX
                )
                print(f"\n   Salt rotasyon : {'✅ Evet (öteleme=0)' if rot_only else '❌ Hayır (öteleme var)'}")

                # Yer değiştirme (varsayılan irtifa=50m)
                dummy_alt = 50.0
                dx, dy, dz = compute_displacement(t_norm, dummy_alt, dummy_alt)
                print(f"\n   Yer değiştirme (irtifa={dummy_alt}m varsayımıyla):")
                print(f"     ΔX: {dx:>8.3f} m")
                print(f"     ΔY: {dy:>8.3f} m")
                print(f"     ΔZ: {dz:>8.3f} m")
            else:
                print(f"\n   ❌ H ayrıştırma başarısız — geçerli çözüm bulunamadı")

        break  # İlk keyframe çiftini bulduk, çık

    if not found_keyframe:
        # Sentetik veri ile test et
        print(f"\n🧪 Sentetik veri ile fonksiyon testi...")

        # Basit bir öteleme Homography'si oluştur
        H_synth = np.eye(3, dtype=np.float64)
        H_synth[0, 2] = 5.0   # 5 piksel sağa kayma
        H_synth[1, 2] = 3.0   # 3 piksel aşağı kayma

        print(f"   Sentetik H (5px sağ, 3px aşağı öteleme):")
        R, t_n, n_vec, ok = decompose_homography(H_synth, K_new)
        print(f"   Ayrıştırma: {'✅' if ok else '❌'}")
        if ok:
            print(f"   t_norm: [{t_n[0]:.6f}, {t_n[1]:.6f}, {t_n[2]:.6f}]")
            print(f"   normal: [{n_vec[0]:.6f}, {n_vec[1]:.6f}, {n_vec[2]:.6f}]")

    print(f"\n{'=' * 60}")
    print("  ✅ Geometry testi tamamlandı")
    print("=" * 60)