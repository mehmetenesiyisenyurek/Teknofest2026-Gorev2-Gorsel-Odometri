"""
Teknofest 2026 — Visual Odometry Pipeline
Geometri Modülü (geometry.py)

Eşleşen noktalardan kameranın nasıl hareket ettiğini hesaplar.

KRİTİK KONVANSIYON:
    OpenCV decomposeHomographyMat formülü: H = R + (1/d) · t · nᵀ
    Burada t = "sahne hareketi" (kameranın hareketinin tersi).
    Kameranın gerçek dünya hareketi = -t.
    Bu negasyon compute_displacement() içinde uygulanır.

    Normal vektörü n, OpenCV konvansiyonunda zemine doğru bakar (nz > 0).
    Aşağı bakan kamera için: n ≈ [0, 0, +1].

Bireysel test:
    python -m src.geometry --video /home/mei/Benim/PROJECTS/Teknofest26/src/data/THYZ_2026_Ornek_Veri_1.MP4
"""

import sys
from pathlib import Path

import cv2
import numpy as np



## Homography Tahmini

def estimate_homography(pts0, pts1, magsac_threshold=3.0):
    """
    İki kare arasındaki Homography'yi MAGSAC++ ile tahmin eder.

    Returns:
        H:            3×3 matris veya None.
        inlier_mask:  (M,) bool veya None.
        inlier_ratio: float, 0-1.
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



## Düzlemsellik Kararı

def is_planar(inlier_ratio, threshold=0.70):
    """Sahne düzlemsel mi?"""
    return inlier_ratio >= threshold


## Normal Vektörü Takibi

class NormalTracker:
    """
    Zemin normali tahminini zamanda takip eder (EMA).

    Küçük hareketlerde decompose normal tahmini gürültülüdür.
    Bu sınıf güvenilir tahminlere daha çok ağırlık vererek
    zamanla kararlı bir normal tahmini üretir.
    """

    def __init__(self, alpha=0.3, default_normal=None):
        self.alpha = alpha
        if default_normal is not None:
            self.smoothed = np.array(default_normal, dtype=np.float64)
        else:
            # OpenCV konvansiyonu: aşağı bakan kamera için n ≈ [0, 0, +1]
            self.smoothed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.count = 0

    def update(self, normal, median_displacement_px):
        """
        Yeni normal tahmini ile günceller.

        Args:
            normal:                 (3,) yeni normal tahmini.
            median_displacement_px: Eşleşmelerin medyan piksel yer değişimi.

        Returns:
            (3,) yumuşatılmış normal.
        """
        normal = normal / np.linalg.norm(normal)

        # Büyük hareket = güvenilir normal, küçük hareket = gürültülü
        confidence = np.clip((median_displacement_px - 2.0) / 8.0, 0.0, 1.0)
        effective_alpha = self.alpha * confidence

        if self.count == 0 and confidence > 0.3:
            self.smoothed = normal.copy()
        elif confidence > 0.1:
            self.smoothed = effective_alpha * normal + (1 - effective_alpha) * self.smoothed
            self.smoothed = self.smoothed / np.linalg.norm(self.smoothed)

        self.count += 1
        return self.smoothed.copy()

    def get(self):
        """Mevcut yumuşatılmış normali döndürür."""
        return self.smoothed.copy()


## Homography Ayrıştırma

def decompose_homography(H, K, pts0=None, pts1=None, inlier_mask=None):
    """
    H matrisini parçalayarak R ve t çıkarır.

    Disambiguasyon:
      1. filterByVisibleRefpoints — noktalar her iki kameranın önünde mi?
      2. nz > 0 — OpenCV konvansiyonunda normal zemine doğru bakar
      3. det(R) ≈ 1, R ortogonal — fiziksel geçerlilik
      4. [0,0,1]'e en yakın normal — en iyi çözüm

    ÖNEMLİ: Döndürülen t "sahne hareketi"dir. Kamera hareketi = -t.
    Bu dönüşüm compute_displacement() içinde yapılır.

    Returns:
        R:        3×3 rotasyon matrisi.
        t_scene:  (3,) sahne hareketi vektörü (kamera = -t).
        normal:   (3,) zemin normali (OpenCV: zemine doğru bakar).
        success:  bool.
    """
    try:
        n_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return None, None, None, False

    if n_solutions == 0:
        return None, None, None, False

    # ── Adım 1: Görünürlük filtresi ──
    visible_mask = None
    if pts0 is not None and pts1 is not None:
        if inlier_mask is not None:
            p0 = pts0[inlier_mask]
            p1 = pts1[inlier_mask]
        else:
            p0 = pts0
            p1 = pts1

        if len(p0) >= 4:
            try:
                p0_f32 = p0.astype(np.float32).reshape(-1, 1, 2)
                p1_f32 = p1.astype(np.float32).reshape(-1, 1, 2)
                visible_mask = cv2.filterHomographyDecompByVisibleRefpoints(
                    rotations, normals, p0_f32, p1_f32
                )
            except cv2.error:
                visible_mask = None

    # ── Adım 2: Skorlama ──
    best_score = -1.0
    best_R, best_t, best_n = None, None, None

    for i in range(n_solutions):
        R_cand = rotations[i]
        t_cand = translations[i]
        n_cand = normals[i]

        # Görünürlük filtresi
        if visible_mask is not None and len(visible_mask) == n_solutions:
            if not visible_mask[i]:
                continue

        n_flat = n_cand.flatten()

        # OpenCV konvansiyonu: n zemine doğru bakar → nz > 0
        nz = float(n_flat[2])
        if nz < 0.2:
            continue

        # det(R) ≈ 1
        if abs(np.linalg.det(R_cand) - 1.0) > 0.05:
            continue

        # R ortogonal
        if np.linalg.norm(R_cand @ R_cand.T - np.eye(3)) > 0.05:
            continue

        # [0,0,1]'e yakınlık
        score = float(np.dot(n_flat, np.array([0.0, 0.0, 1.0])))
        if score > best_score:
            best_score = score
            best_R = R_cand
            best_t = t_cand
            best_n = n_cand

    if best_R is None:
        return None, None, None, False

    return best_R, best_t.flatten(), best_n.flatten(), True


## Salt Rotasyon Tespiti

def check_rotation_only(R, pts0, pts1, K, threshold_px=1.5):
    """
    Kameranın sadece döndüğü durumu tespit eder.

    Returns:
        bool: True ise salt rotasyon.
    """
    if len(pts0) < 4:
        return False

    K_inv = np.linalg.inv(K)
    ones = np.ones((len(pts0), 1))
    pts0_h = np.hstack([pts0, ones])
    pts0_norm = (K_inv @ pts0_h.T).T
    pts0_rotated = (R @ pts0_norm.T).T
    pts0_reproj = (K @ pts0_rotated.T).T
    pts0_reproj = pts0_reproj[:, :2] / pts0_reproj[:, 2:3]

    errors = np.linalg.norm(pts0_reproj - pts1, axis=1)
    median_error = float(np.median(errors))

    return median_error < threshold_px



## Yer Değiştirme Hesaplama

def compute_displacement(t_scene, altitude, prev_altitude):
    """
    Metre cinsinden kamera yer değiştirmesi (ΔX, ΔY, ΔZ) hesaplar.

    KRİTİK: decomposeHomographyMat sahne hareketini döndürür.
    Kameranın gerçek dünya hareketi = -t_scene.

    Formül:  H = R + (1/d) · t · nᵀ
    t burada "sahne nasıl hareket etti" demek.
    Kamera sağa giderse → sahne sola gider → t_x < 0
    Bu yüzden kamera hareketi = -t.

    Args:
        t_scene:       (3,) sahne hareketi (decomposeHomographyMat çıktısı).
        altitude:      Mevcut irtifa (metre).
        prev_altitude: Önceki irtifa (metre).

    Returns:
        dx, dy, dz: Metre cinsinden KAMERA yer değiştirmesi.
    """
    # Negasyon: sahne hareketi → kamera hareketi
    dx = -float(t_scene[0]) * altitude
    dy = -float(t_scene[1]) * altitude
    dz = altitude - prev_altitude

    return dx, dy, dz



## BİREYSEL TEST

if __name__ == "__main__":
    import argparse

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
        "--max-frames", type=int, default=120,
        help="Maksimum taranacak kare sayısı",
    )
    args = parser.parse_args()

    # Ground truth (ilk 59 kare, yarışma verisi)
    GT_POSITIONS = {
        0: (0.04443, 0.00306, -0.00023),
        16: (0.08825, -0.10316, -0.05761),
    }

    print("=" * 60)
    print("  geometry.py — Bireysel Test (v5 — t negasyonu düzeltmesi)")
    print(f"  Cihaz: {args.device} | Profil: {CAMERA_PROFILE}")
    print("=" * 60)

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
    tracker = NormalTracker(alpha=0.3)

    cap, frame_count, size, fps = load_video(args.video)

    print(f"\n İlk 3 keyframe çiftini test ediyorum...\n")

    prev_feats = None
    keyframe_count = 0

    for fid, frame in iter_frames(cap):
        if fid >= args.max_frames or keyframe_count >= 3:
            break

        enhanced, undistorted = prep.preprocess(frame)
        feats = fe.extract(enhanced)

        if prev_feats is None:
            prev_feats = feats
            continue

        pts0, pts1, scores = fe.match(prev_feats, feats)
        pts0_c, pts1_c = fe.filter_matches(pts0, pts1, scores)

        kf, reason = is_keyframe(pts0_c, pts1_c, MIN_MATCHES, MIN_MOTION_PX)
        if not kf:
            continue

        keyframe_count += 1
        median_disp = float(np.median(np.linalg.norm(pts1_c - pts0_c, axis=1)))

        print(f"{'─' * 50}")
        print(f" Keyframe #{keyframe_count}: kare 0 ↔ kare {fid} ({reason})")

        # Homography
        H, mask, ratio = estimate_homography(pts0_c, pts1_c, MAGSAC_THRESHOLD)

        if H is None or not is_planar(ratio, HOMOGRAPHY_INLIER_THRESHOLD):
            prev_feats = feats
            continue

        print(f"   H inlier: {ratio:.2%}")

        # Tüm 4 çözüm (debug)
        from scipy.spatial.transform import Rotation as Rot
        try:
            n_sol, Rs, ts, ns = cv2.decomposeHomographyMat(H, K_new)
            vis = None
            try:
                p0_f32 = pts0_c.astype(np.float32).reshape(-1, 1, 2)
                p1_f32 = pts1_c.astype(np.float32).reshape(-1, 1, 2)
                vis = cv2.filterHomographyDecompByVisibleRefpoints(Rs, ns, p0_f32, p1_f32)
            except cv2.error:
                pass

            print(f"\n   4 aday çözüm:")
            for i in range(n_sol):
                n_f = ns[i].flatten()
                t_f = ts[i].flatten()
                vis_str = ""
                if vis is not None and len(vis) == n_sol:
                    vis_str = " VIS" if vis[i] else " ❌HID"
                print(f"     [{i}] nz={n_f[2]:>+.3f} t=[{t_f[0]:>+.4f},{t_f[1]:>+.4f},{t_f[2]:>+.4f}]{vis_str}")
        except cv2.error:
            pass

        # Seçilen çözüm
        R, t_scene, normal, success = decompose_homography(H, K_new, pts0_c, pts1_c, mask)

        if not success:
            prev_feats = feats
            continue

        normal_smooth = tracker.update(normal, median_disp)
        euler = Rot.from_matrix(R).as_euler("xyz", degrees=True)

        print(f"\n    Seçilen çözüm:")
        print(f"   R: roll={euler[0]:>+7.3f}° pitch={euler[1]:>+7.3f}° yaw={euler[2]:>+7.3f}°")
        print(f"   t_scene:  [{t_scene[0]:>+.6f}, {t_scene[1]:>+.6f}, {t_scene[2]:>+.6f}]")
        print(f"   -t (kamera): [{-t_scene[0]:>+.6f}, {-t_scene[1]:>+.6f}, {-t_scene[2]:>+.6f}]")
        print(f"   normal:   [{normal[0]:>+.4f}, {normal[1]:>+.4f}, {normal[2]:>+.4f}]")

        rot_only = check_rotation_only(R, pts0_c, pts1_c, K_new, ROTATION_ONLY_THRESHOLD_PX)
        print(f"   salt rot: {'' if rot_only else '❌'}")

        # Farklı irtifa varsayımlarıyla test
        print(f"\n    Yer değiştirme (farklı irtifalar):")
        for h_test in [20, 30, 50, 80]:
            dx, dy, dz = compute_displacement(t_scene, h_test, h_test)
            print(f"     h={h_test:>3d}m: ΔX={dx:>+8.4f}m  ΔY={dy:>+8.4f}m")

        # Ground truth karşılaştırma
        if fid in GT_POSITIONS and 0 in GT_POSITIONS:
            gt_dx = GT_POSITIONS[fid][0] - GT_POSITIONS[0][0]
            gt_dy = GT_POSITIONS[fid][1] - GT_POSITIONS[0][1]
            print(f"\n   📊 Ground Truth (kare 0→{fid}):")
            print(f"     GT:  ΔX={gt_dx:>+8.4f}m  ΔY={gt_dy:>+8.4f}m")

            # İrtifayı ground truth'tan tersine hesapla
            if abs(t_scene[0]) > 1e-6:
                h_from_x = abs(gt_dx / t_scene[0])
                print(f"     Tahmini irtifa (X'ten): ~{h_from_x:.0f}m")
            if abs(t_scene[1]) > 1e-6:
                h_from_y = abs(gt_dy / t_scene[1])
                print(f"     Tahmini irtifa (Y'den): ~{h_from_y:.0f}m")

        prev_feats = feats

    print(f"\n{'=' * 60}")
    print("   Geometry testi tamamlandı")
    print("=" * 60)