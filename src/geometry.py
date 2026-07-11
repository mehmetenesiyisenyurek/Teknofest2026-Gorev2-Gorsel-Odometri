"""
Teknofest 2026 — Visual Odometry Pipeline
Geometri Modülü (geometry.py)

Eşleşen noktalardan kameranın nasıl hareket ettiğini hesaplar.

İKİ YOL MİMARİSİ:
    Birincil Yol (Homography):
        Sahne düzlemsel → H ayrıştır → R, t/h, n → irtifa ile ölçekle
        Avantaj: Düzlemsel sahnelerde en kararlı çözüm

    Yedek Yol (Essential Matrix + PnP):
        Sahne 3D yapılar içeriyor → Essential'dan R + PnP'den metrik t
        Avantaj: Düzlemsel olmayan sahnelerde de çalışır

    Yol seçimi: GRIC (Geometric Robust Information Criterion, Torr 1998)

KRİTİK KONVANSIYONLAR:
    Birincil Yol:
        OpenCV decomposeHomographyMat formülü: H = R + (1/d) · t · nᵀ
        Burada t = "sahne hareketi" (kameranın hareketinin tersi).
        Kameranın gerçek dünya hareketi = -t.
        Bu negasyon compute_displacement() içinde uygulanır.

        Normal vektörü n, OpenCV konvansiyonunda zemine doğru bakar (nz > 0).
        Aşağı bakan kamera için: n ≈ [0, 0, +1].

    Yedek Yol:
        solvePnPRansac: önceki kamera frame'indeki 3D noktaları mevcut kareye
        projekte eden R, t döndürür. Çıkan t, kamera yer değiştirmesini
        önceki kamera koordinatlarında, METRE cinsinden verir.
        Negasyon gerekmez — doğrudan accumulator'a verilir.

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


# ──────────────────────────────────────────────────────────────
# Düzlemsellik Kararı
# ──────────────────────────────────────────────────────────────

def is_planar(inlier_ratio, threshold=0.70):
    """Sahne düzlemsel mi?"""
    return inlier_ratio >= threshold


# ──────────────────────────────────────────────────────────────
# GRIC Model Seçimi (Torr 1998)
# ──────────────────────────────────────────────────────────────

def _symmetric_transfer_error(pts0, pts1, H):
    """
    Homography için simetrik transfer hatası (piksel² cinsinden).

    İleri: pts0 → H → pts1_pred, hata = ||pts1 - pts1_pred||²
    Geri:  pts1 → H⁻¹ → pts0_pred, hata = ||pts0 - pts0_pred||²
    Sonuç: (ileri + geri) / 2 → outlier'lara karşı simetrik ve dengeli.

    Args:
        pts0: (N, 2) ilk karedeki noktalar.
        pts1: (N, 2) ikinci karedeki noktalar.
        H:    3×3 homography matrisi.

    Returns:
        (N,) simetrik transfer hatası (piksel²).
    """
    n = len(pts0)
    ones = np.ones((n, 1))

    # İleri projeksiyon: pts0 → pts1_pred
    pts0_h = np.hstack([pts0, ones])
    pts1_pred = (H @ pts0_h.T).T
    pts1_pred = pts1_pred[:, :2] / pts1_pred[:, 2:3]
    err_fwd = np.sum((pts1 - pts1_pred) ** 2, axis=1)

    # Geri projeksiyon: pts1 → pts0_pred
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return err_fwd  # Tekil matris — sadece ileri hatayı kullan

    pts1_h = np.hstack([pts1, ones])
    pts0_pred = (H_inv @ pts1_h.T).T
    pts0_pred = pts0_pred[:, :2] / pts0_pred[:, 2:3]
    err_bwd = np.sum((pts0 - pts0_pred) ** 2, axis=1)

    return (err_fwd + err_bwd) / 2.0


def _sampson_distance(pts0, pts1, F):
    """
    Fundamental matris için Sampson mesafesi (piksel² cinsinden).

    Geometrik hatanın 1. derece Taylor yaklaşımı.
    Tam geometrik hataya çok yakın ama hesaplama maliyeti çok düşük.

    Args:
        pts0: (N, 2) ilk karedeki noktalar.
        pts1: (N, 2) ikinci karedeki noktalar.
        F:    3×3 fundamental matrisi.

    Returns:
        (N,) Sampson mesafesi (piksel²).
    """
    n = len(pts0)
    ones = np.ones((n, 1))

    p0_h = np.hstack([pts0, ones])  # (N, 3)
    p1_h = np.hstack([pts1, ones])  # (N, 3)

    # x₂ᵀ F x₁ — epipolar kısıt
    Fp0 = (F @ p0_h.T).T            # (N, 3) — epipolar çizgiler (ikinci kare)
    Ftp1 = (F.T @ p1_h.T).T         # (N, 3) — epipolar çizgiler (ilk kare)

    # Numeratör: (x₂ᵀ F x₁)²
    num = np.sum(p1_h * Fp0, axis=1) ** 2

    # Paydanın sıfır olmasını önle
    denom = (Fp0[:, 0] ** 2 + Fp0[:, 1] ** 2 +
             Ftp1[:, 0] ** 2 + Ftp1[:, 1] ** 2)
    denom = np.maximum(denom, 1e-12)

    return num / denom


def compute_gric(residuals_sq, sigma, n_points, k, d, r=4):
    """
    GRIC skoru hesaplar (Torr 1998).

    GRIC = Σ ρ(eᵢ²/σ²) + λ₁·d·n + λ₂·k

    Düşük skor = veriye daha iyi ve daha sade uyum.

    Args:
        residuals_sq: (N,) her noktanın karesel residual hatası (piksel²).
        sigma:        Piksel gürültü standart sapması (genellikle ~1.0 px).
        n_points:     Toplam eşleşme sayısı.
        k:            Model parametre sayısı (H=8, F=7).
        d:            Model manifold boyutu (H: d=2, F: d=3).
        r:            Veri boyutu (2D-2D eşleşme: r=4).

    Returns:
        float: GRIC skoru.
    """
    sigma2 = sigma * sigma

    # Outlier cezası: r - d boyut modelce açıklanamıyor
    T2 = 2.0 * (r - d)

    # Robust maliyet: inlier → gerçek hata, outlier → sabit ceza
    rho = np.minimum(residuals_sq / sigma2, T2)

    # Model karmaşıklık cezaları
    lambda1 = np.log(r)
    lambda2 = np.log(r * n_points) / 2.0

    return float(np.sum(rho) + lambda1 * d * n_points + lambda2 * k)


def select_model_gric(pts0, pts1, K, H, sigma=1.0):
    """
    GRIC ile Homography ve Fundamental Matrix arasında seçim yapar.

    Her iki modelin veriye uyumunu ve karmaşıklığını karşılaştırarak
    istatistiksel olarak en uygun modeli seçer.

    Args:
        pts0:  (N, 2) ilk karedeki eşleşme noktaları.
        pts1:  (N, 2) ikinci karedeki eşleşme noktaları.
        K:     3×3 kamera iç parametre matrisi.
        H:     3×3 homography matrisi (daha önce hesaplanmış).
        sigma: Piksel gürültü standart sapması.

    Returns:
        model:  "homography" veya "essential".
        gric_h: Homography GRIC skoru.
        gric_f: Fundamental GRIC skoru.
    """
    n = len(pts0)

    # ── Homography residual'ları ──
    res_h = _symmetric_transfer_error(pts0, pts1, H)

    # ── Fundamental Matrix hesapla ve residual'ları al ──
    F, mask_f = cv2.findFundamentalMat(
        pts0, pts1,
        method=cv2.FM_8POINT,
    )

    if F is None or F.shape != (3, 3):
        # F hesaplanamazsa H'yi seç
        return "homography", 0.0, float("inf")

    res_f = _sampson_distance(pts0, pts1, F)

    # ── GRIC skorları ──
    # Homography:    k=8 parametre, d=2 (düzlemsel manifold)
    # Fundamental:   k=7 parametre, d=3 (epipolar manifold)
    gric_h = compute_gric(res_h, sigma, n, k=8, d=2)
    gric_f = compute_gric(res_f, sigma, n, k=7, d=3)

    model = "homography" if gric_h <= gric_f else "essential"
    return model, gric_h, gric_f


# ──────────────────────────────────────────────────────────────
# Normal Vektörü Takibi
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Homography Ayrıştırma (Birincil Yol)
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Salt Rotasyon Tespiti
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Yer Değiştirme Hesaplama (Birincil Yol)
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Yedek Yol: Essential Matrix + PnP
# ──────────────────────────────────────────────────────────────

def backproject_points(pts, depth_map, K):
    """
    2D piksel koordinatlarını derinlik haritası ile 3D kamera
    koordinatlarına dönüştürür.

    Formül:
        X = (u - cx) · d / fx
        Y = (v - cy) · d / fy
        Z = d

    Args:
        pts:       (N, 2) piksel koordinatları (u, v).
        depth_map: (H, W) metre cinsinden derinlik haritası.
        K:         3×3 kamera iç parametre matrisi.

    Returns:
        pts3d:     (N, 3) 3D koordinatlar (kamera frame).
        valid:     (N,) bool maskesi — geçerli derinliğe sahip noktalar.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h_img, w_img = depth_map.shape[:2]

    pts3d = np.zeros((len(pts), 3), dtype=np.float64)
    valid = np.zeros(len(pts), dtype=bool)

    for i, (u, v) in enumerate(pts):
        u_int = int(round(u))
        v_int = int(round(v))

        # Sınır kontrolü
        if u_int < 0 or u_int >= w_img or v_int < 0 or v_int >= h_img:
            continue

        d = float(depth_map[v_int, u_int])

        # Geçersiz derinlik kontrolü
        if d <= 0 or not np.isfinite(d):
            continue

        pts3d[i, 0] = (u - cx) * d / fx
        pts3d[i, 1] = (v - cy) * d / fy
        pts3d[i, 2] = d
        valid[i] = True

    return pts3d, valid


def estimate_essential(pts0, pts1, K):
    """
    Essential Matrix ile iki kare arasındaki R ve t yön vektörünü hesaplar.

    Essential Matrix derinlik bilgisine ihtiyaç duymaz, bu yüzden
    rotasyon tahmini derinlik gürültüsünden bağımsız ve kararlıdır.

    NOT: Döndürülen t birim vektördür — ölçek bilgisi yoktur.
    Metrik ölçek PnP'den alınır.

    Args:
        pts0: (N, 2) ilk karedeki noktalar.
        pts1: (N, 2) ikinci karedeki noktalar.
        K:    3×3 kamera iç parametre matrisi.

    Returns:
        R:       3×3 rotasyon matrisi.
        t_unit:  (3,) birim öteleme vektörü (sadece yön).
        success: bool.
    """
    if len(pts0) < 5:
        return None, None, False

    E, mask_e = cv2.findEssentialMat(
        pts0, pts1, K,
        method=cv2.USAC_MAGSAC,
        prob=0.999,
        threshold=1.0,
    )

    if E is None:
        return None, None, False

    n_infront, R, t, mask_rp = cv2.recoverPose(E, pts0, pts1, K)

    # Noktaların en az %50'si kameranın önünde olmalı
    infront_ratio = n_infront / len(pts0)
    if infront_ratio < 0.50:
        return None, None, False

    return R, t.flatten(), True


def estimate_pose_pnp(pts0, pts1, K, depth_map0):
    """
    Yedek yol: Essential Matrix'ten R + PnP'den metrik t.

    Hibrit yapının avantajı:
        - R: Essential Matrix'ten → derinlik gürültüsünden bağımsız, kararlı
        - t: PnP'den → doğrudan metre cinsinden, ölçek bilgisi var

    Akış:
        1. pts0'ı depth_map0 ile 3D'ye backproject et
        2. Essential Matrix → R_essential (güvenilir rotasyon)
        3. solvePnPRansac → R_pnp, t_pnp (metrik ölçekli)
        4. R_essential ve t_pnp'yi birleştir

    KONVANSIYON: Çıkan R ve t, önceki kamera frame'inden mevcut kamera
    frame'ine dönüşümü temsil eder. t, önceki kamera koordinatlarında
    metre cinsinden yer değiştirmedir.

    Args:
        pts0:       (N, 2) önceki karedeki noktalar.
        pts1:       (N, 2) mevcut karedeki noktalar.
        K:          3×3 kamera iç parametre matrisi.
        depth_map0: (H, W) önceki karenin derinlik haritası (metre).

    Returns:
        R:       3×3 rotasyon matrisi (Essential'dan).
        t_metric: (3,) metre cinsinden yer değiştirme (PnP'den).
        success: bool.
    """
    # ── 1. 3D backprojection ──
    pts3d, valid = backproject_points(pts0, depth_map0, K)

    n_valid = int(np.sum(valid))
    if n_valid < 10:
        return None, None, False

    pts3d_v = pts3d[valid]
    pts0_v = pts0[valid]
    pts1_v = pts1[valid]

    # ── 2. Essential Matrix → R (derinlik bağımsız, kararlı) ──
    R_essential, _, e_success = estimate_essential(pts0_v, pts1_v, K)
    if not e_success:
        R_essential = None  # PnP'den R kullanılacak

    # ── 3. PnP → R_pnp, t_pnp (metrik ölçekli) ──
    pts3d_f32 = pts3d_v.astype(np.float32)
    pts1_f32 = pts1_v.astype(np.float32).reshape(-1, 1, 2)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d_f32, pts1_f32, K.astype(np.float32), None,
        iterationsCount=2000,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None or len(inliers) < 6:
        return None, None, False

    R_pnp, _ = cv2.Rodrigues(rvec)
    t_pnp = tvec.flatten()

    # ── 4. Hibrit birleştirme ──
    # R → Essential'dan (derinlik gürültüsüne bağlı değil, daha kararlı)
    # t → PnP'den (doğrudan metre cinsinden)
    R_final = R_essential if R_essential is not None else R_pnp

    # PnP konvansiyonu: p_cam = R @ p_world + t
    # Kamera yer değiştirmesi (önceki cam frame'inde): -R.T @ t
    t_camera = -R_final.T @ t_pnp

    return R_final, t_camera, True


# ──────────────────────────────────────────────────────────────
# BİREYSEL TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import torch
    from scipy.spatial.transform import Rotation as Rot
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
    print("  geometry.py — Bireysel Test (v6 — GRIC + Yedek Yol)")
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

    print(f"\n🧪 BÖLÜM 1: Birincil Yol + GRIC (gerçek video)\n")

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
        print(f"🎯 Keyframe #{keyframe_count}: kare 0 ↔ kare {fid} ({reason})")

        # Homography
        H, mask, ratio = estimate_homography(pts0_c, pts1_c, MAGSAC_THRESHOLD)

        if H is None or not is_planar(ratio, HOMOGRAPHY_INLIER_THRESHOLD):
            prev_feats = feats
            continue

        print(f"   H inlier: {ratio:.2%}")

        # ── GRIC Model Seçimi ──
        sigma_gric = MAGSAC_THRESHOLD / 3.0
        model, gric_h, gric_f = select_model_gric(pts0_c, pts1_c, K_new, H, sigma=sigma_gric)
        print(f"\n   📊 GRIC Analizi:")
        print(f"     GRIC_H = {gric_h:.1f}  |  GRIC_F = {gric_f:.1f}")
        print(f"     Seçilen model: {model.upper()}")
        print(f"     Fark: {abs(gric_h - gric_f):.1f} ({'H daha iyi' if gric_h < gric_f else 'F daha iyi'})")

        # Tüm 4 aday çözüm (debug)
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
                    vis_str = " ✅VIS" if vis[i] else " ❌HID"
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

        print(f"\n   🏆 Seçilen çözüm:")
        print(f"   R: roll={euler[0]:>+7.3f}° pitch={euler[1]:>+7.3f}° yaw={euler[2]:>+7.3f}°")
        print(f"   t_scene:  [{t_scene[0]:>+.6f}, {t_scene[1]:>+.6f}, {t_scene[2]:>+.6f}]")
        print(f"   -t (kamera): [{-t_scene[0]:>+.6f}, {-t_scene[1]:>+.6f}, {-t_scene[2]:>+.6f}]")
        print(f"   normal:   [{normal[0]:>+.4f}, {normal[1]:>+.4f}, {normal[2]:>+.4f}]")

        rot_only = check_rotation_only(R, pts0_c, pts1_c, K_new, ROTATION_ONLY_THRESHOLD_PX)
        print(f"   salt rot: {'✅' if rot_only else '❌'}")

        # Farklı irtifa varsayımlarıyla test
        print(f"\n   📏 Yer değiştirme (farklı irtifalar):")
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

    # ── BÖLÜM 2: Yedek Yol (Sentetik Veri) ──
    print(f"\n{'═' * 60}")
    print(f"🧪 BÖLÜM 2: Yedek Yol — Sentetik Doğrulama\n")

    # Sentetik kamera: basit focal length
    K_syn = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # ── Test 2a: Backprojection ──
    print("🧪 Test 2a: Backprojection Doğrulaması")

    # Bilinen 3D noktalar
    pts3d_true = np.array([
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 10.0],
        [1.0, 1.0, 10.0],
    ], dtype=np.float64)

    # 3D → 2D projeksiyon (piksel koordinatlarını hesapla)
    pts2d_syn = np.zeros((4, 2))
    for i in range(4):
        x, y, z = pts3d_true[i]
        pts2d_syn[i, 0] = K_syn[0, 0] * x / z + K_syn[0, 2]
        pts2d_syn[i, 1] = K_syn[1, 1] * y / z + K_syn[1, 2]

    # Sentetik derinlik haritası (640x480, tüm pikseller 10m)
    depth_syn = np.full((480, 640), 10.0, dtype=np.float64)

    pts3d_recon, valid = backproject_points(pts2d_syn, depth_syn, K_syn)

    all_valid = np.all(valid)
    recon_err = np.max(np.abs(pts3d_recon[valid] - pts3d_true[valid]))
    print(f"   Tüm noktalar geçerli: {'✅' if all_valid else '❌'}")
    print(f"   Maks. rekonstrüksiyon hatası: {recon_err:.6f}m {'✅' if recon_err < 0.01 else '❌'}")

    # ── Test 2b: Essential Matrix ──
    print(f"\n🧪 Test 2b: Essential Matrix (sentetik hareket)")

    # Bilinen hareket: 5° yaw + [0.5, 0, 0] öteleme
    R_true = Rot.from_euler("z", 5, degrees=True).as_matrix()
    t_true = np.array([0.5, 0.0, 0.0])

    # 50 rastgele 3D nokta oluştur (derinlik 8-15m arası)
    np.random.seed(42)
    n_pts = 50
    pts3d_rand = np.random.uniform(-3, 3, (n_pts, 3))
    pts3d_rand[:, 2] = np.random.uniform(8, 15, n_pts)

    # Kare 0: doğrudan projeksiyon
    pts0_syn = (K_syn @ pts3d_rand.T).T
    pts0_syn = pts0_syn[:, :2] / pts0_syn[:, 2:3]

    # Kare 1: R, t uygulanmış projeksiyon
    pts3d_moved = (R_true @ pts3d_rand.T).T + t_true
    pts1_syn = (K_syn @ pts3d_moved.T).T
    pts1_syn = pts1_syn[:, :2] / pts1_syn[:, 2:3]

    R_est, t_est, e_ok = estimate_essential(pts0_syn.astype(np.float64), pts1_syn.astype(np.float64), K_syn)
    if e_ok:
        R_err = np.linalg.norm(R_est - R_true, "fro")
        # t yönü karşılaştır (Essential sadece yön verir)
        t_true_unit = t_true / np.linalg.norm(t_true)
        t_dot = abs(float(np.dot(t_est, t_true_unit)))
        print(f"   R hatası (Frobenius): {R_err:.4f} {'✅' if R_err < 0.1 else '❌'}")
        print(f"   t yön benzerliği:     {t_dot:.4f} {'✅' if t_dot > 0.9 else '❌'}")
    else:
        print(f"   ❌ Essential Matrix başarısız!")

    # ── Test 2c: Tam PnP Yedek Yol ──
    print(f"\n🧪 Test 2c: Hibrit Essential + PnP (sentetik hareket)")

    # Sentetik derinlik haritası oluştur (her noktanın derinliğini piksel pozisyonuna yaz)
    depth_syn2 = np.zeros((480, 640), dtype=np.float64)
    for i in range(n_pts):
        u_int = int(round(pts0_syn[i, 0]))
        v_int = int(round(pts0_syn[i, 1]))
        if 0 <= u_int < 640 and 0 <= v_int < 480:
            depth_syn2[v_int, u_int] = pts3d_rand[i, 2]

    R_pnp, t_pnp, pnp_ok = estimate_pose_pnp(
        pts0_syn.astype(np.float64),
        pts1_syn.astype(np.float64),
        K_syn,
        depth_syn2,
    )

    if pnp_ok:
        # t_camera konvansiyonu: önceki kamera frame'inde yer değiştirme
        # Sentetik hareketi buna çevir: t_true önceki frame'de → doğrudan karşılaştır
        R_err_pnp = np.linalg.norm(R_pnp - R_true, "fro")
        t_err_pnp = np.linalg.norm(t_pnp - t_true)
        print(f"   R hatası (Frobenius): {R_err_pnp:.4f} {'✅' if R_err_pnp < 0.15 else '❌'}")
        print(f"   t hatası (metre):     {t_err_pnp:.4f}m {'✅' if t_err_pnp < 0.3 else '❌'}")
        print(f"   t tahmin: [{t_pnp[0]:>+.4f}, {t_pnp[1]:>+.4f}, {t_pnp[2]:>+.4f}]")
        print(f"   t gerçek: [{t_true[0]:>+.4f}, {t_true[1]:>+.4f}, {t_true[2]:>+.4f}]")
    else:
        print(f"   ❌ PnP başarısız!")

    print(f"\n{'=' * 60}")
    print("  ✅ Geometry testi tamamlandı (v6 — GRIC + Yedek Yol)")
    print("=" * 60)