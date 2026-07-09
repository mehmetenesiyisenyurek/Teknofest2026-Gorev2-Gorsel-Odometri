"""
Teknofest 2026 — Visual Odometry Pipeline
Poz Biriktirme Modülü (pose.py)

Kare kare hesaplanan R ve (dx, dy, dz) değerlerini
global bir yörüngeye (trajectory) dönüştürür.

Sorumluluklar:
    - 4×4 homojen dönüşüm matrisi ile global poz takibi
    - Kamera koordinatından dünya koordinatına dönüşüm
    - Keyframe olmayan kareler için SLERP+LERP interpolasyon
    - GT döndüğünde pozisyon sıfırlama (reset_to_gt)
    - Tüm yörüngeyi dışa aktarma

Bireysel test:
    python -m src.pose
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


class PoseAccumulator:
    """
    Global poz biriktirici.

    Her keyframe'de:
        1. camera_to_world() ile (dx,dy,dz)'yi dünya koordinatına çevir
        2. accumulate() ile global pozu güncelle

    Skiplenen kareler:
        3. skip() ile işaretle
        4. interpolate_skipped() ile SLERP+LERP doldur
    """

    def __init__(self):
        """Başlangıç: orijinde, birim rotasyonla."""
        # Global dönüşüm matrisi (4×4)
        # [R  t]
        # [0  1]
        self.T_global = np.eye(4, dtype=np.float64)

        # Kayıtlı pozlar: {frame_id: {"T": 4x4, "type": "keyframe"|"skipped"|"interpolated"|"gt"}}
        self.poses = {}

        # Keyframe listesi (sıralı)
        self._keyframes = []

        # İlk kareyi kaydet
        self.poses[0] = {
            "T": self.T_global.copy(),
            "type": "keyframe",
        }
        self._keyframes.append(0)

    @staticmethod
    def make_T(R, t):
        """
        3×3 R ve (3,) t'den 4×4 homojen dönüşüm matrisi oluşturur.

        Args:
            R: 3×3 rotasyon matrisi.
            t: (3,) öteleme vektörü.

        Returns:
            4×4 numpy array.
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def camera_to_world(self, dx, dy, dz):
        """
        Kamera koordinatındaki yer değiştirmeyi dünya koordinatına çevirir.

        Mevcut global rotasyonu kullanarak:
            world_disp = R_global @ [dx, dy, dz]

        Args:
            dx, dy, dz: Kamera koordinatında yer değiştirme (metre).

        Returns:
            (3,) dünya koordinatında yer değiştirme.
        """
        R_global = self.T_global[:3, :3]
        camera_disp = np.array([dx, dy, dz], dtype=np.float64)
        world_disp = R_global @ camera_disp
        return world_disp

    def accumulate(self, R_local, dx, dy, dz, frame_id):
        """
        Yeni keyframe'in pozunu global poza ekler.

        Adımlar:
            1. (dx, dy, dz)'yi dünya koordinatına çevir
            2. Global rotasyonu güncelle: R_global = R_local @ R_global
            3. Global pozisyonu güncelle: pos += world_disp
            4. Kaydet

        Args:
            R_local:  3×3 yerel rotasyon matrisi (kare N-1 → N).
            dx, dy, dz: Kamera koordinatında yer değiştirme (metre).
            frame_id: Kare numarası.
        """
        # Dünya koordinatına çevir
        world_disp = self.camera_to_world(dx, dy, dz)

        # Global pozu güncelle
        # Pozisyon: mevcut pozisyon + dünya yer değiştirmesi
        self.T_global[:3, 3] += world_disp

        # Rotasyon: yerel rotasyonu global rotasyona uygula
        self.T_global[:3, :3] = R_local @ self.T_global[:3, :3]

        # Kaydet
        self.poses[frame_id] = {
            "T": self.T_global.copy(),
            "type": "keyframe",
        }
        self._keyframes.append(frame_id)

    def skip(self, frame_id):
        """
        Keyframe olmayan kareyi işaretler (sonra interpolasyon yapılacak).

        Args:
            frame_id: Kare numarası.
        """
        self.poses[frame_id] = {
            "T": None,
            "type": "skipped",
        }

    def interpolate_skipped(self):
        """
        Skiplenen karelerin pozlarını SLERP (rotasyon) + LERP (öteleme) ile doldurur.

        Her ardışık keyframe çifti arasındaki boşlukları doldurur.
        """
        if len(self._keyframes) < 2:
            return

        for i in range(len(self._keyframes) - 1):
            kf_start = self._keyframes[i]
            kf_end = self._keyframes[i + 1]

            if kf_end - kf_start <= 1:
                continue  # Ardışık keyframe'ler, interpolasyon gerekmez

            T_start = self.poses[kf_start]["T"]
            T_end = self.poses[kf_end]["T"]

            if T_start is None or T_end is None:
                continue

            # Rotasyon ve pozisyon çıkar
            R_start = T_start[:3, :3]
            R_end = T_end[:3, :3]
            pos_start = T_start[:3, 3]
            pos_end = T_end[:3, 3]

            # SLERP hazırlığı
            rots = Rot.from_matrix(np.stack([R_start, R_end]))
            slerp = Slerp([0.0, 1.0], rots)

            # Her skiplenen kare için interpolasyon
            n_gap = kf_end - kf_start
            for j in range(1, n_gap):
                fid = kf_start + j
                alpha = j / n_gap  # 0 < alpha < 1

                # SLERP: rotasyon
                R_interp = slerp(alpha).as_matrix()

                # LERP: pozisyon
                pos_interp = (1 - alpha) * pos_start + alpha * pos_end

                T_interp = self.make_T(R_interp, pos_interp)

                self.poses[fid] = {
                    "T": T_interp,
                    "type": "interpolated",
                }

    def reset_to_gt(self, frame_id, gt_position):
        """
        GT döndüğünde pozisyonu GT'ye eşitler ve drift'i sıfırlar.

        Rotasyonu korur (GT rotasyon vermiyorsa).

        Args:
            frame_id:    Kare numarası.
            gt_position: (3,) GT dünya pozisyonu.
        """
        gt_pos = np.array(gt_position, dtype=np.float64)

        # Global pozisyonu GT'ye ayarla, rotasyonu koru
        self.T_global[:3, 3] = gt_pos

        self.poses[frame_id] = {
            "T": self.T_global.copy(),
            "type": "gt",
        }
        self._keyframes.append(frame_id)

    def set_identity(self, frame_id):
        """
        İlk kareyi orijinde ayarlar.

        Args:
            frame_id: Kare numarası.
        """
        self.T_global = np.eye(4, dtype=np.float64)
        self.poses[frame_id] = {
            "T": self.T_global.copy(),
            "type": "keyframe",
        }
        if frame_id not in self._keyframes:
            self._keyframes.append(frame_id)

    def get_position(self, frame_id=None):
        """
        Belirtilen karenin dünya pozisyonunu döndürür.

        Args:
            frame_id: Kare numarası. None ise mevcut pozisyon.

        Returns:
            (3,) numpy array veya None.
        """
        if frame_id is None:
            return self.T_global[:3, 3].copy()

        if frame_id in self.poses and self.poses[frame_id]["T"] is not None:
            return self.poses[frame_id]["T"][:3, 3].copy()

        return None

    def get_rotation(self, frame_id=None):
        """
        Belirtilen karenin rotasyon matrisini döndürür.

        Args:
            frame_id: Kare numarası. None ise mevcut rotasyon.

        Returns:
            3×3 numpy array veya None.
        """
        if frame_id is None:
            return self.T_global[:3, :3].copy()

        if frame_id in self.poses and self.poses[frame_id]["T"] is not None:
            return self.poses[frame_id]["T"][:3, :3].copy()

        return None

    def get_trajectory(self):
        """
        Tüm kaydedilmiş pozisyonları sıralı olarak döndürür.

        Returns:
            frame_ids: Kare numaraları listesi (sıralı).
            positions: (N, 3) numpy array.
            types:     Tip listesi ("keyframe", "interpolated", "gt", "skipped").
        """
        sorted_ids = sorted(
            fid for fid, p in self.poses.items()
            if p["T"] is not None
        )

        if not sorted_ids:
            return [], np.empty((0, 3)), []

        positions = np.array([self.poses[fid]["T"][:3, 3] for fid in sorted_ids])
        types = [self.poses[fid]["type"] for fid in sorted_ids]

        return sorted_ids, positions, types

    def get_current_frame_id(self):
        """Son keyframe'in kare numarasını döndürür."""
        return self._keyframes[-1] if self._keyframes else 0

    def get_vo_position(self):
        """
        Mevcut VO pozisyonunu döndürür (kalibrasyon öncesi ham değer).
        calibration.py'a verilecek.
        """
        return self.T_global[:3, 3].copy()


# BİREYSEL TEST

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 60)
    print("  pose.py — Bireysel Test (sentetik veri)")
    print("=" * 60)

    # ── Test 1: Düz ileri hareket ──
    print("\n Test 1: 10 kare düz ileri hareket (dx=1m/kare)")

    acc = PoseAccumulator()
    R_identity = np.eye(3)

    for i in range(1, 11):
        acc.accumulate(R_identity, 1.0, 0.0, 0.0, frame_id=i)

    pos = acc.get_position()
    print(f"   Son pozisyon: {pos}")
    print(f"   Beklenen:     [10.0, 0.0, 0.0]")
    error = np.linalg.norm(pos - np.array([10.0, 0.0, 0.0]))
    print(f"   Hata: {error:.6f}m {'OK' if error < 0.001 else '❌'}")

    # ── Test 2: 90° yaw sonrası hareket ──
    print(f"\n{'─' * 50}")
    print(" Test 2: 90° yaw dönüşü sonrası 5m ileri")

    acc2 = PoseAccumulator()

    # Kare 1: 90° yaw (Z ekseni etrafında) + 0 öteleme
    # Yaw = saat yönünün tersi
    R_yaw90 = Rot.from_euler("z", 90, degrees=True).as_matrix()
    acc2.accumulate(R_yaw90, 0.0, 0.0, 0.0, frame_id=1)

    # Kare 2-6: kamera frame'inde X yönünde 1m
    # Ama global frame'de artık Y yönünde olmalı (90° döndük)
    R_id = np.eye(3)
    for i in range(2, 7):
        acc2.accumulate(R_id, 1.0, 0.0, 0.0, frame_id=i)

    pos2 = acc2.get_position()
    print(f"   Son pozisyon: {pos2}")
    print(f"   Beklenen:     ~[0.0, 5.0, 0.0] (90° yaw sonrası X→Y)")
    # 90° yaw sonrası kamera X ekseni = dünya Y ekseni
    expected2 = np.array([0.0, 5.0, 0.0])
    error2 = np.linalg.norm(pos2[:2] - expected2[:2])
    print(f"   XY hatası: {error2:.4f}m {'OK' if error2 < 0.1 else '❌'}")

    # ── Test 3: İnterpolasyon ──
    print(f"\n{'─' * 50}")
    print(" Test 3: Keyframe araları interpolasyon")

    acc3 = PoseAccumulator()

    # Kare 0: orijin (zaten set)
    # Kare 5: (5, 0, 0) — keyframe
    acc3.accumulate(np.eye(3), 5.0, 0.0, 0.0, frame_id=5)
    # Kare 10: (10, 5, 0) — keyframe
    acc3.accumulate(np.eye(3), 5.0, 5.0, 0.0, frame_id=10)

    # Kare 1-4 ve 6-9'u skiplenmiş olarak işaretle
    for fid in range(1, 10):
        if fid not in [5]:
            acc3.skip(fid)

    # İnterpolasyon
    acc3.interpolate_skipped()

    print(f"   Kare  0: {acc3.get_position(0)}")
    print(f"   Kare  2: {acc3.get_position(2)}  (interpolated)")
    print(f"   Kare  5: {acc3.get_position(5)}  (keyframe)")
    print(f"   Kare  7: {acc3.get_position(7)}  (interpolated)")
    print(f"   Kare 10: {acc3.get_position(10)} (keyframe)")

    # Kare 2.5: beklenen (2.5, 0, 0) → kare 2: (2, 0, 0)
    pos_2 = acc3.get_position(2)
    expected_2 = np.array([2.0, 0.0, 0.0])
    err_2 = np.linalg.norm(pos_2 - expected_2)
    print(f"   Kare 2 hatası: {err_2:.4f}m {'OK' if err_2 < 0.1 else '❌'}")

    # Kare 7: beklenen (%40: 5→10, 0→5) = (7, 2, 0)
    pos_7 = acc3.get_position(7)
    expected_7 = np.array([7.0, 2.0, 0.0])
    err_7 = np.linalg.norm(pos_7 - expected_7)
    print(f"   Kare 7 hatası: {err_7:.4f}m {'OK' if err_7 < 0.1 else '❌'}")

    # ── Test 4: GT reset ──
    print(f"\n{'─' * 50}")
    print(" Test 4: GT reset")

    acc4 = PoseAccumulator()

    # Birkaç kare ilerle
    for i in range(1, 6):
        acc4.accumulate(np.eye(3), 1.0, 0.5, 0.0, frame_id=i)

    pos_before = acc4.get_position()
    print(f"   VO pozisyonu (drift'li): {pos_before}")

    # GT döndü: gerçek pozisyon farklı (drift var)
    gt_pos = np.array([100.0, 200.0, 50.0])
    acc4.reset_to_gt(5, gt_pos)

    pos_after = acc4.get_position()
    print(f"   GT reset sonrası:        {pos_after}")
    print(f"   Beklenen:                {gt_pos}")
    reset_err = np.linalg.norm(pos_after - gt_pos)
    print(f"   Reset hatası: {reset_err:.6f}m {'✅' if reset_err < 0.001 else '❌'}")

    # Reset sonrası devam
    acc4.accumulate(np.eye(3), 1.0, 0.0, 0.0, frame_id=6)
    pos_next = acc4.get_position()
    print(f"   1m sonra:                {pos_next}")
    print(f"   Beklenen:                [101.0, 200.0, 50.0]")

    # ── Yörünge çıktısı ──
    print(f"\n{'─' * 50}")
    print(" Test 5: Yörünge export")
    fids, positions, types = acc4.get_trajectory()
    print(f"   Toplam kayıt: {len(fids)}")
    for fid, pos, typ in zip(fids, positions, types):
        print(f"     kare {fid:>3d}: [{pos[0]:>8.2f}, {pos[1]:>8.2f}, {pos[2]:>8.2f}]  ({typ})")

    print(f"\n{'=' * 60}")
    print("   Pose testi tamamlandı")
    print("=" * 60)