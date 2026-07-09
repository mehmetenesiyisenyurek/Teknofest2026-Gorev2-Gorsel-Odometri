"""
Teknofest 2026 — Visual Odometry Pipeline
Ana Pipeline Modülü (pipeline.py)

Tüm modülleri birleştirerek uçtan uca VO çalıştırır.

Faz Yönetimi:
    Faz 1 — Kalibrasyon (gps_health=1, kare < 450):
        VO çalıştır + GT ile kalibrasyon örnekleri biriktir + GT gönder
    Faz 2 — Tahmin (gps_health=0):
        VO çalıştır + kalibrasyon dönüşümü uygula + tahmin gönder
    Faz 3 — Reset (gps_health=1, kare >= 450):
        Pozisyonu GT'ye sıfırla + kalibrasyon güncelle + GT gönder

Bireysel test:
    python -m src.pipeline --video /home/mei/Benim/PROJECTS/Teknofest26/src/data/THYZ_2026_Ornek_Veri_1.MP4 --gt /home/mei/Benim/PROJECTS/Teknofest26/src/data/THYZ_2026_Ornek_Veri_1_translation.csv --max-frames 200

"""

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

import torch

if not torch.cuda.is_available():
    print("⚠️ Sistemde NVIDIA GPU bulunamadı. Pipeline testi için CPU yaması aktif ediliyor...")

    orig_linspace = torch.linspace


    def patched_linspace(*args, **kwargs):
        if 'device' in kwargs and kwargs['device'] == 'cuda':
            kwargs['device'] = 'cpu'
        return orig_linspace(*args, **kwargs)


    torch.linspace = patched_linspace
else:
    print("🚀 NVIDIA GPU (CUDA) algılandı! Model tam performans ekran kartında çalışacak.")


def load_ground_truth(gt_path):
    """
    Ground truth CSV/TSV dosyasını yükler.

    Beklenen format (tab veya virgül ayrımı):
        translation_x  translation_y  translation_z  frame_numbers
        0.04443        0.00306        -0.00023       frame_000000

    Args:
        gt_path: GT dosya yolu.

    Returns:
        dict: {frame_id: (x, y, z)} veya None.
    """
    if gt_path is None or not Path(gt_path).exists():
        return None

    gt = {}
    with open(gt_path, "r") as f:
        lines = f.readlines()

    # Başlık satırını atla
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # Tab veya virgül ayırıcı
        parts = line.replace(",", " ").split()
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 4:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            frame_str = parts[3]

            # "frame_000016" → 16
            if frame_str.startswith("frame_"):
                fid = int(frame_str.replace("frame_", ""))
            else:
                fid = int(frame_str)

            gt[fid] = (x, y, z)
        except (ValueError, IndexError):
            continue

    return gt if gt else None


def simulate_gps_health(frame_id, gt_data, cut_start=450, cut_intervals=None):
    """
    GPS sağlık durumunu simüle eder (offline test için).

    Args:
        frame_id:       Mevcut kare numarası.
        gt_data:        GT sözlüğü {fid: (x,y,z)}.
        cut_start:      GT'nin kesildiği ilk kare.
        cut_intervals:  [(start, end), ...] GT'nin kesildiği aralıklar.
                        None ise basit model: 0-449=GT var, 450+=GT yok.

    Returns:
        int: 1 = GT mevcut, 0 = GT kesildi.
    """
    if gt_data is None:
        return 0

    if cut_intervals is not None:
        for start, end in cut_intervals:
            if start <= frame_id < end:
                return 0
        return 1

    # Basit model: ilk cut_start kare GT var, sonra yok
    return 1 if frame_id < cut_start else 0


def run_pipeline(
    video_path,
    config_module,
    gt_path=None,
    max_frames=None,
    device="cpu",
    output_path=None,
    gps_cut_start=450,
    verbose=True,
):
    """
    Ana VO pipeline'ını çalıştırır.

    Args:
        video_path:     Video dosya yolu.
        config_module:  config modülü (import edilmiş).
        gt_path:        Ground truth dosya yolu (offline test).
        max_frames:     Maksimum kare sayısı (None = tümü).
        device:         'cuda' veya 'cpu'.
        output_path:    Sonuç JSON dosya yolu.
        gps_cut_start:  GT'nin kesildiği kare (simülasyon).
        verbose:        Ayrıntılı çıktı.

    Returns:
        dict: Sonuçlar {frame_id: (x, y, z)}.
    """
    # ── Modülleri import et ──
    from src.utils import load_video, iter_frames, Timer
    from src.preprocessing import Preprocessor
    from src.feature import FeatureExtractor, is_keyframe
    from src.geometry import (
        estimate_homography, is_planar, decompose_homography,
        check_rotation_only, compute_displacement, NormalTracker,
    )
    from src.depth import AltitudeEstimator
    from src.calibration import VOCalibrator
    from src.pose import PoseAccumulator

    cfg = config_module

    # ── Bileşenleri oluştur ──
    prep = Preprocessor(
        K=cfg.CAMERA_K, dist_coeffs=cfg.DIST_COEFFS, image_size=cfg.IMAGE_SIZE,
        clahe_clip=cfg.CLAHE_CLIP_LIMIT, clahe_tile=cfg.CLAHE_TILE_SIZE,
        blur_kernel=cfg.BLUR_KERNEL,
    )
    fe = FeatureExtractor(
        device=device, max_keypoints=cfg.MAX_KEYPOINTS,
        confidence_threshold=cfg.CONFIDENCE_THRESHOLD,
        flow_percentile=cfg.FLOW_PERCENTILE,
    )
    depth_est = AltitudeEstimator(
        device=device, fx=float(cfg.CAMERA_K[0, 0]),
        ema_alpha=0.2, min_altitude=5.0, max_altitude=500.0,
    )
    calibrator = VOCalibrator(min_samples=10)
    accumulator = PoseAccumulator()
    normal_tracker = NormalTracker(alpha=0.3)
    K_new = prep.get_K_new()

    # ── GT yükle ──
    gt_data = load_ground_truth(gt_path)
    if gt_data:
        print(f"   📄 GT yüklendi: {len(gt_data)} kare")
    else:
        print(f"   ⚠️  GT dosyası yok — kalibrasyonsuz mod")

    # ── Video ──
    cap, frame_count, size, fps = load_video(video_path)
    if max_frames is None:
        max_frames = frame_count

    # ── State ──
    prev_feats = None
    prev_frame = None
    prev_altitude = None
    results = {}  # {frame_id: (x, y, z)}
    phase = "calibration"
    keyframe_count = 0
    calibrated = False

    # ── İstatistikler ──
    stats = {
        "total_frames": 0,
        "keyframes": 0,
        "skipped": 0,
        "rotation_only": 0,
        "homography_fail": 0,
        "phase_changes": [],
    }

    print(f"\n🚀 Pipeline başlatılıyor...")
    print(f"   Maks kare: {max_frames} | GPS kesim: kare {gps_cut_start}")
    print(f"{'─' * 60}")

    t_start = time.time()

    for fid, frame in iter_frames(cap):
        if fid >= max_frames:
            break

        stats["total_frames"] += 1

        # ── 1. GPS durumu ──
        gps_health = simulate_gps_health(fid, gt_data, gps_cut_start)
        gt_pos = gt_data.get(fid) if gt_data else None

        # Faz belirle
        if gps_health == 1 and not calibrated:
            new_phase = "calibration"
        elif gps_health == 1 and calibrated:
            new_phase = "reset"
        else:
            new_phase = "prediction"

        if new_phase != phase:
            stats["phase_changes"].append((fid, new_phase))
            if verbose:
                print(f"\n   📌 Faz değişimi → {new_phase.upper()} (kare {fid})")
            phase = new_phase

        # ── 2. Ön işleme ──
        enhanced, undistorted = prep.preprocess(frame)

        # ── 3. Özellik çıkarımı ──
        feats = fe.extract(enhanced)

        if prev_feats is None:
            prev_feats = feats
            prev_frame = frame

            # İlk kare için irtifa
            prev_altitude = depth_est.estimate_altitude(frame)

            # İlk kare sonucu
            if gt_pos is not None:
                results[fid] = gt_pos
            else:
                results[fid] = (0.0, 0.0, 0.0)

            continue

        # ── 4. Eşleştirme ──
        pts0, pts1, scores = fe.match(prev_feats, feats)
        pts0_c, pts1_c = fe.filter_matches(pts0, pts1, scores)

        # ── 5. Keyframe kararı ──
        kf, reason = is_keyframe(pts0_c, pts1_c, cfg.MIN_MATCHES, cfg.MIN_MOTION_PX)

        if not kf:
            # Keyframe değil — skip
            accumulator.skip(fid)
            stats["skipped"] += 1

            # Sonucu belirle
            if phase == "calibration" and gt_pos is not None:
                results[fid] = gt_pos
            elif phase == "prediction" and calibrator.is_calibrated:
                # Son bilinen VO pozisyonunu dönüştür
                vo_pos = accumulator.get_vo_position()
                world_pos = calibrator.transform(vo_pos)
                results[fid] = tuple(world_pos)
            elif phase == "reset" and gt_pos is not None:
                results[fid] = gt_pos
            else:
                # Fallback: son bilinen pozisyon
                last_pos = accumulator.get_position()
                results[fid] = tuple(last_pos) if last_pos is not None else (0, 0, 0)

            continue

        # ── KEYFRAME İŞLEME ──
        keyframe_count += 1
        stats["keyframes"] += 1

        # 6. Homography
        H, mask, ratio = estimate_homography(pts0_c, pts1_c, cfg.MAGSAC_THRESHOLD)

        if H is None or not is_planar(ratio, cfg.HOMOGRAPHY_INLIER_THRESHOLD):
            stats["homography_fail"] += 1
            accumulator.skip(fid)
            prev_feats = feats
            prev_frame = frame

            # Sonuç: önceki pozisyon
            last_pos = accumulator.get_position()
            results[fid] = tuple(last_pos) if last_pos is not None else (0, 0, 0)
            continue

        # 7. H ayrıştırma
        R, t_scene, normal_raw, success = decompose_homography(
            H, K_new, pts0_c, pts1_c, mask
        )

        if not success:
            accumulator.skip(fid)
            prev_feats = feats
            prev_frame = frame
            last_pos = accumulator.get_position()
            results[fid] = tuple(last_pos) if last_pos is not None else (0, 0, 0)
            continue

        # Normal tracking
        median_disp = float(np.median(np.linalg.norm(pts1_c - pts0_c, axis=1)))
        normal_tracker.update(normal_raw, median_disp)

        # 8. Salt rotasyon kontrolü
        rot_only = check_rotation_only(
            R, pts0_c, pts1_c, K_new, cfg.ROTATION_ONLY_THRESHOLD_PX
        )

        if rot_only:
            stats["rotation_only"] += 1
            # Rotasyon uygula, öteleme=0
            accumulator.accumulate(R, 0.0, 0.0, 0.0, fid)
        else:
            # 9. İrtifa tahmini
            altitude = depth_est.estimate_altitude(frame)
            if prev_altitude is None:
                prev_altitude = altitude

            # 10. Yer değiştirme
            dx, dy, dz = compute_displacement(t_scene, altitude, prev_altitude)

            # 11. Poz biriktir
            accumulator.accumulate(R, dx, dy, dz, fid)

            prev_altitude = altitude

        # ── FAZ-SPESİFİK İŞLEMLER ──
        vo_pos = accumulator.get_vo_position()

        if phase == "calibration":
            if gt_pos is not None:
                # Kalibrasyon örneği ekle
                calibrator.add_sample(vo_pos, np.array(gt_pos))
                results[fid] = gt_pos

                # Yeterli örnek birikti mi?
                if calibrator.get_sample_count() >= 10 and not calibrator.is_calibrated:
                    if calibrator.calibrate():
                        calibrated = True
                        info = calibrator.get_calibration_info()
                        if verbose:
                            print(f"   ✅ Kalibrasyon tamamlandı! "
                                  f"Ölçek={info['scale']:.4f}, "
                                  f"Hata={info['mean_error']:.4f}m")
            else:
                results[fid] = tuple(vo_pos)

        elif phase == "prediction":
            if calibrator.is_calibrated:
                world_pos = calibrator.transform(vo_pos)
                results[fid] = tuple(world_pos)
            else:
                results[fid] = tuple(vo_pos)

        elif phase == "reset":
            if gt_pos is not None:
                accumulator.reset_to_gt(fid, np.array(gt_pos))
                calibrator.update_on_gt_return(vo_pos, np.array(gt_pos))
                results[fid] = gt_pos
            else:
                results[fid] = tuple(vo_pos)

        # Güncel keyframe'i sakla
        prev_feats = feats
        prev_frame = frame

        # Verbose çıktı (her 10 keyframe'de)
        if verbose and keyframe_count % 10 == 0:
            pos = accumulator.get_position()
            elapsed = time.time() - t_start
            fps_actual = stats["total_frames"] / elapsed if elapsed > 0 else 0
            print(f"   KF#{keyframe_count:>4d} kare={fid:>5d} "
                  f"pos=[{pos[0]:>8.2f},{pos[1]:>8.2f},{pos[2]:>8.2f}] "
                  f"({fps_actual:.1f} fps)")

    # ── İnterpolasyon ──
    accumulator.interpolate_skipped()

    # İnterpolasyon sonuçlarını güncelle
    fids_interp, positions_interp, types_interp = accumulator.get_trajectory()
    for fid_i, pos_i, typ_i in zip(fids_interp, positions_interp, types_interp):
        if typ_i == "interpolated" and fid_i not in results:
            if calibrator.is_calibrated:
                world_pos = calibrator.transform(pos_i)
                results[fid_i] = tuple(world_pos)
            else:
                results[fid_i] = tuple(pos_i)

    # ── Sonuçları kaydet ──
    elapsed = time.time() - t_start

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"📊 Pipeline Tamamlandı!")
        print(f"   Toplam kare   : {stats['total_frames']}")
        print(f"   Keyframe      : {stats['keyframes']}")
        print(f"   Skiplenen     : {stats['skipped']}")
        print(f"   Salt rotasyon : {stats['rotation_only']}")
        print(f"   H başarısız   : {stats['homography_fail']}")
        print(f"   Süre          : {elapsed:.1f}s ({stats['total_frames']/elapsed:.1f} fps)")

        if calibrator.is_calibrated:
            info = calibrator.get_calibration_info()
            print(f"   Kalibrasyon   : s={info['scale']:.4f}, "
                  f"hata={info['mean_error']:.4f}m, "
                  f"örnekler={info['num_samples']}")

    if output_path:
        output = {
            "predictions": {
                str(fid): {"x": pos[0], "y": pos[1], "z": pos[2]}
                for fid, pos in sorted(results.items())
            },
            "stats": stats,
            "calibration": calibrator.get_calibration_info() if calibrator.is_calibrated else None,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"   💾 Sonuçlar kaydedildi: {output_path}")

    # ── GT ile hata hesapla ──
    if gt_data and verbose:
        errors = []
        for fid, pred in results.items():
            if fid in gt_data:
                gt = gt_data[fid]
                err = np.sqrt((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2 + (pred[2]-gt[2])**2)
                errors.append(err)

        if errors:
            errors = np.array(errors)
            print(f"\n   📐 GT Karşılaştırma (tüm kareler):")
            print(f"     Ortalama hata : {np.mean(errors):.4f}m")
            print(f"     Medyan hata   : {np.median(errors):.4f}m")
            print(f"     Maks hata     : {np.max(errors):.4f}m")
            print(f"     Min hata      : {np.min(errors):.4f}m")

    return results


# ──────────────────────────────────────────────────────────────
# BİREYSEL TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import config as cfg

    parser = argparse.ArgumentParser(description="pipeline.py — VO Pipeline Test")
    parser.add_argument(
        "--video", type=str, default=cfg.DEFAULT_VIDEO_PATH,
        help="Video dosya yolu",
    )
    parser.add_argument(
        "--gt", type=str, default=None,
        help="Ground truth dosya yolu (CSV/TSV)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Cihaz: cuda veya cpu",
    )
    parser.add_argument(
        "--max-frames", type=int, default=200,
        help="Maksimum kare sayısı",
    )
    parser.add_argument(
        "--output", type=str, default="result.json",
        help="Sonuç dosya yolu",
    )
    parser.add_argument(
        "--gps-cut", type=int, default=450,
        help="GT kesim karesi (simülasyon)",
    )
    args = parser.parse_args()

    # torch import burada — GPU kontrolü için
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA mevcut değil, CPU kullanılıyor")
        args.device = "cpu"

    print("=" * 60)
    print("  pipeline.py — Visual Odometry Pipeline")
    print(f"  Cihaz: {args.device} | Profil: {cfg.CAMERA_PROFILE}")
    print(f"  Video: {Path(args.video).name}")
    print("=" * 60)

    results = run_pipeline(
        video_path=args.video,
        config_module=cfg,
        gt_path=args.gt,
        max_frames=args.max_frames,
        device=args.device,
        output_path=args.output,
        gps_cut_start=args.gps_cut,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print("  ✅ Pipeline testi tamamlandı")
    print("=" * 60)