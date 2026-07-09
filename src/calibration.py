"""
Teknofest 2026 — Visual Odometry Pipeline
Kalibrasyon Modülü (calibration.py)

GT ile VO arasındaki dönüşümü hesaplar:
    p_world = scale * R_calib @ p_vo + t_calib

Yarışma akışı:
    1. İlk 450 karede: add_sample(vo_pos, gt_pos)
    2. 450. karede:     calibrate() → s, R, t hesapla
    3. GT kesilince:    transform(vo_pos) → dünya koordinatı
    4. GT dönünce:      update_on_gt_return(vo_pos, gt_pos) → drift sıfırla

Procrustes analizi:
    - Ölçek:    s = ||GT_centered|| / ||VO_centered||
    - Rotasyon: SVD(GT^T @ VO) → U @ V^T
    - Öteleme:  t = mean(GT) - s * R @ mean(VO)

Bireysel test:
    python -m src.calibration
"""

import sys
from pathlib import Path

import numpy as np


class VOCalibrator:
    """
    VO → Dünya koordinat dönüşümü yöneticisi.

    p_world = scale * R_calib @ p_vo + t_calib
    """

    def __init__(self, min_samples=10):
        """
        Args:
            min_samples: Kalibrasyon için gereken minimum örnek sayısı.
        """
        self.min_samples = min_samples

        # Kalibrasyon parametreleri
        self.scale = 1.0
        self.R_calib = np.eye(3, dtype=np.float64)
        self.t_calib = np.zeros(3, dtype=np.float64)

        # Biriken örnekler
        self._vo_samples = []     # [(x, y, z), ...]
        self._gt_samples = []     # [(x, y, z), ...]

        # Durum
        self.is_calibrated = False
        self._calibration_error = None

    def add_sample(self, vo_pos, gt_pos):
        """
        Kalibrasyon fazında VO-GT çifti ekler.

        Args:
            vo_pos: (3,) VO pozisyonu (yerel koordinat).
            gt_pos: (3,) GT pozisyonu (dünya koordinat).
        """
        self._vo_samples.append(np.array(vo_pos, dtype=np.float64))
        self._gt_samples.append(np.array(gt_pos, dtype=np.float64))

    def get_sample_count(self):
        """Biriken örnek sayısını döndürür."""
        return len(self._vo_samples)

    def calibrate(self):
        """
        Biriken örneklerden optimal dönüşümü hesaplar.

        Procrustes analizi:
            1. Her iki veri setini merkezle
            2. Ölçek: ||GT|| / ||VO||
            3. Rotasyon: SVD(GT^T @ VO) → R = U @ V^T
            4. Öteleme: mean(GT) - s * R @ mean(VO)

        Returns:
            bool: Kalibrasyon başarılı mı.
        """
        n = len(self._vo_samples)
        if n < self.min_samples:
            print(f"   ⚠️  Kalibrasyon için yetersiz örnek: {n}/{self.min_samples}")
            return False

        vo_arr = np.array(self._vo_samples)  # (N, 3)
        gt_arr = np.array(self._gt_samples)  # (N, 3)

        # ── Merkezleme ──
        vo_mean = vo_arr.mean(axis=0)
        gt_mean = gt_arr.mean(axis=0)
        vo_c = vo_arr - vo_mean  # centered
        gt_c = gt_arr - gt_mean

        # ── Ölçek ──
        vo_norm = np.linalg.norm(vo_c)
        gt_norm = np.linalg.norm(gt_c)

        if vo_norm < 1e-10:
            print("   ⚠️  VO hareketi çok küçük, kalibrasyon başarısız")
            return False

        self.scale = gt_norm / vo_norm

        # ── Rotasyon (Procrustes — SVD) ──
        # R: vo_c'yi gt_c'ye döndüren matris
        # gt_c ≈ s * R @ vo_c
        # Minimize: ||gt_c - s * R @ vo_c||^2
        # Çözüm: H = vo_c^T @ gt_c, SVD(H) = U S V^T, R = V @ U^T
        H = vo_c.T @ gt_c  # (3, 3)
        U, S, Vt = np.linalg.svd(H)

        # Yansıma kontrolü (det(R) = +1 olmalı)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, d])

        self.R_calib = Vt.T @ sign_matrix @ U.T

        # ── Öteleme ──
        self.t_calib = gt_mean - self.scale * self.R_calib @ vo_mean

        # ── Hata hesapla ──
        gt_pred = self.scale * (self.R_calib @ vo_arr.T).T + self.t_calib
        errors = np.linalg.norm(gt_pred - gt_arr, axis=1)
        self._calibration_error = float(np.mean(errors))

        self.is_calibrated = True
        return True

    def transform(self, vo_pos):
        """
        VO pozisyonunu dünya koordinatına dönüştürür.

        p_world = scale * R_calib @ p_vo + t_calib

        Args:
            vo_pos: (3,) VO pozisyonu.

        Returns:
            (3,) dünya koordinatı.
        """
        vo_pos = np.array(vo_pos, dtype=np.float64)
        return self.scale * self.R_calib @ vo_pos + self.t_calib

    def update_on_gt_return(self, current_vo_pos, gt_pos):
        """
        GT geri döndüğünde çağrılır — drift'i sıfırlar.

        Strateji:
            1. Ötelemeyi güncelle: t = gt - s * R @ vo
            2. Ölçek ve rotasyonu koru (yeterli veri yoksa değiştirme)
            3. Yeni örnekleri biriktirmeye devam et

        Args:
            current_vo_pos: (3,) mevcut VO pozisyonu.
            gt_pos:         (3,) GT pozisyonu.
        """
        current_vo_pos = np.array(current_vo_pos, dtype=np.float64)
        gt_pos = np.array(gt_pos, dtype=np.float64)

        # Öteleme güncelle — drift'i sıfırlar
        self.t_calib = gt_pos - self.scale * self.R_calib @ current_vo_pos

        # Örnek ekle (gelecek kalibrasyon güncellemeleri için)
        self._vo_samples.append(current_vo_pos.copy())
        self._gt_samples.append(gt_pos.copy())

    def recalibrate(self, window_size=None):
        """
        Mevcut örneklerle yeniden kalibrasyon yapar.

        Args:
            window_size: Son N örneği kullan. None ise tümünü kullan.

        Returns:
            bool: Başarılı mı.
        """
        if window_size is not None and len(self._vo_samples) > window_size:
            # Geçici olarak son N örneği kullan
            old_vo = self._vo_samples
            old_gt = self._gt_samples
            self._vo_samples = self._vo_samples[-window_size:]
            self._gt_samples = self._gt_samples[-window_size:]
            result = self.calibrate()
            self._vo_samples = old_vo
            self._gt_samples = old_gt
            return result
        else:
            return self.calibrate()

    def get_calibration_info(self):
        """Kalibrasyon bilgilerini döndürür."""
        return {
            "is_calibrated": self.is_calibrated,
            "scale": self.scale,
            "R_calib": self.R_calib.tolist(),
            "t_calib": self.t_calib.tolist(),
            "num_samples": len(self._vo_samples),
            "mean_error": self._calibration_error,
        }



# BİREYSEL TEST

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 60)
    print("  calibration.py — Bireysel Test (sentetik veri)")
    print("=" * 60)

    # ── Test 1: Basit ölçek + öteleme ──
    print("\n Test 1: Ölçek=3.0, Öteleme=[10, 20, 5], Rotasyon=yok")

    cal = VOCalibrator(min_samples=5)

    # Sentetik GT: dünya koordinatları
    gt_positions = [
        [10.0, 20.0, 5.0],
        [10.3, 20.6, 5.1],
        [10.9, 21.5, 5.0],
        [11.8, 22.8, 4.9],
        [13.0, 24.5, 4.8],
        [14.5, 26.6, 4.7],
        [16.3, 29.1, 4.6],
        [18.4, 32.0, 4.5],
        [20.8, 35.3, 4.4],
        [23.5, 39.0, 4.3],
    ]

    # Sentetik VO: yerel koordinatlar (ölçek=3x küçük, öteleme farklı)
    # vo = (gt - [10, 20, 5]) / 3.0
    vo_positions = [
        [(g[0] - 10) / 3.0, (g[1] - 20) / 3.0, (g[2] - 5) / 3.0]
        for g in gt_positions
    ]

    for vo, gt in zip(vo_positions, gt_positions):
        cal.add_sample(vo, gt)

    success = cal.calibrate()
    print(f"   Kalibrasyon : {'✅ Başarılı' if success else '❌ Başarısız'}")
    print(f"   Ölçek       : {cal.scale:.4f} (beklenen: 3.0)")
    print(f"   Öteleme     : {cal.t_calib} (beklenen: [10, 20, 5])")
    print(f"   Ort. hata   : {cal._calibration_error:.6f}m")

    # Dönüşüm doğrula
    test_vo = np.array([1.0, 2.0, -0.1])
    test_world = cal.transform(test_vo)
    expected = np.array([13.0, 26.0, 4.7])
    error = np.linalg.norm(test_world - expected)
    print(f"\n   Dönüşüm testi:")
    print(f"     VO:       {test_vo}")
    print(f"     Dünya:    {test_world}")
    print(f"     Beklenen: {expected}")
    print(f"     Hata:     {error:.6f}m {'✅' if error < 0.01 else '❌'}")

    # ── Test 2: Ölçek + rotasyon (90° yaw) ──
    print(f"\n{'─' * 50}")
    print(" Test 2: Ölçek=2.0, 90° yaw rotasyonu")

    cal2 = VOCalibrator(min_samples=5)

    # 90° yaw: VO X → GT Y, VO Y → GT -X
    # GT = 2 * R_90 @ VO + [100, 200, 50]
    R_90 = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ], dtype=np.float64)

    vo_pts = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 2.0, -0.1],
        [4.0, 3.0, -0.2],
        [5.0, 5.0, -0.3],
        [6.0, 7.0, -0.4],
        [7.0, 10.0, -0.5],
    ]

    gt_pts = []
    for vo in vo_pts:
        vo_arr = np.array(vo)
        gt = 2.0 * R_90 @ vo_arr + np.array([100, 200, 50])
        gt_pts.append(gt.tolist())
        cal2.add_sample(vo, gt)

    success2 = cal2.calibrate()
    print(f"   Kalibrasyon : {'✅' if success2 else '❌'}")
    print(f"   Ölçek       : {cal2.scale:.4f} (beklenen: 2.0)")
    print(f"   Ort. hata   : {cal2._calibration_error:.6f}m")
    print(f"   R_calib:\n{cal2.R_calib}")
    print(f"   R_expected:\n{R_90}")

    R_error = np.linalg.norm(cal2.R_calib - R_90)
    print(f"   ||R - R_expected||: {R_error:.6f} {'✅' if R_error < 0.01 else '❌'}")

    # ── Test 3: GT reset (drift sıfırlama) ──
    print(f"\n{'─' * 50}")
    print(" Test 3: GT kesilme → drift → GT dönüş → reset")

    cal3 = VOCalibrator(min_samples=5)

    # Basit kalibrasyon: GT = 2 * VO + [0, 0, 0]
    for i in range(10):
        vo = [float(i), float(i) * 0.5, 0.0]
        gt = [float(i) * 2, float(i) * 1.0, 0.0]
        cal3.add_sample(vo, gt)

    cal3.calibrate()
    print(f"   Kalibrasyon: s={cal3.scale:.2f}")

    # GT kesildi → VO devam ediyor, drift birikiyor
    # VO'da drift var: gerçek pozisyon (20, 10, 0) ama VO (10.5, 5.5, 0) diyor
    drifted_vo = np.array([10.5, 5.5, 0.0])
    drifted_world = cal3.transform(drifted_vo)
    print(f"   Drift'li tahmin : {drifted_world}")

    # GT döndü: gerçek pozisyon (20, 10, 0)
    gt_return = np.array([20.0, 10.0, 0.0])
    drift = np.linalg.norm(drifted_world - gt_return)
    print(f"   Drift miktarı   : {drift:.4f}m")

    # Reset
    cal3.update_on_gt_return(drifted_vo, gt_return)
    after_reset = cal3.transform(drifted_vo)
    reset_error = np.linalg.norm(after_reset - gt_return)
    print(f"   Reset sonrası   : {after_reset}")
    print(f"   Reset hatası    : {reset_error:.6f}m {'✅' if reset_error < 0.001 else '❌'}")

    # Sonraki VO pozisyonu da doğru mu?
    next_vo = np.array([11.0, 6.0, 0.0])
    next_world = cal3.transform(next_vo)
    print(f"   Sonraki tahmin  : {next_world}")

    print(f"\n{'=' * 60}")
    print("   Calibration testi tamamlandı")
    print("=" * 60)