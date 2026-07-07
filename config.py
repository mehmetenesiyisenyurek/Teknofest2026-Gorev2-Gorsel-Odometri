"""
Teknofest 2026 — Visual Odometry Pipeline
Merkezi Konfigürasyon Dosyası

Tüm pipeline boyunca kullanılan parametreler burada tanımlanır.
Parametre tuning yaparken SADECE bu dosya değiştirilir.

Kullanım:
    from config import CFG
    print(CFG.CAMERA_K)

Bireysel test:
    python config.py
"""

import numpy as np



# KAMERA PROFİLLERİ

# Aktif kamera profili — kullanılacak kamerayı seçer
# Seçenekler: "thermal", "rgb_4k", "rgb_1080p"
CAMERA_PROFILE = "rgb_1080p"




# --- Termal Kamera ---
_THERMAL = {
    "fx": 731.7965,
    "fy": 732.0172,
    "cx": 319.2367,
    "cy": 251.2424,
    "dist_coeffs": np.array([-0.3507, 0.1137, 0.0, 0.0, 0.0], dtype=np.float64),
    "image_size": (640, 512),  # (genişlik, yükseklik)
}

# --- RGB Kamera 4K ---
_RGB_4K = {
    "fx": 2792.2,
    "fy": 2795.2,
    "cx": 1988.0,
    "cy": 1562.2,
    "dist_coeffs": np.array([0.0798, -0.1867, 0.0, 0.0, 0.0], dtype=np.float64),
    "image_size": (4000, 3000),  # (genişlik, yükseklik)
}

# --- RGB Kamera 1080p ---
_RGB_1080P = {
    "fx": 1389.7,
    "fy": 1387.1,
    "cx": 954.007,
    "cy": 558.896,
    "dist_coeffs": np.array([0.1378, -0.2564, 0.0, 0.0, 0.0], dtype=np.float64),
    "image_size": (1920, 1080),  # (genişlik, yükseklik)
}

_PROFILES = {
    "thermal": _THERMAL,
    "rgb_4k": _RGB_4K,
    "rgb_1080p": _RGB_1080P,
}




## AKTİF KAMERA PARAMETRELERİ

_active = _PROFILES[CAMERA_PROFILE]

# K MATRİSİ
# 3×3 Kamera İç Parametre Matrisi
#   [[fx,  0, cx],
#    [ 0, fy, cy],
#    [ 0,  0,  1]]
CAMERA_K = np.array([
    [_active["fx"], 0.0,           _active["cx"]],
    [0.0,          _active["fy"],  _active["cy"]],
    [0.0,          0.0,            1.0          ],
], dtype=np.float64)

# 5 elemanlı DİSTRASYON KATSAYILARI [k1, k2, p1, p2, k3]
DIST_COEFFS = _active["dist_coeffs"]

# Görüntü boyutu (genişlik, yükseklik)
IMAGE_SIZE = _active["image_size"]





## ÖN İŞLEME PARAMETRELERİ

# CLAHE kontrastı ne kadar artıracak (düşük=az etki, yüksek=çok kontrast+gürültü)
CLAHE_CLIP_LIMIT = 2.0

# CLAHE blok boyutu — görüntü kaç parçaya bölünecek
CLAHE_TILE_SIZE = (8, 8)

# Gaussian Blur kernel boyutu (tek sayı olmalı, 3=hafif, 5=daha güçlü)
BLUR_KERNEL = 3


## ÖZELLİK ÇIKARIMI PARAMETRELERİ
# Bir karede en fazla kaç anahtar nokta tespit edilecek
MAX_KEYPOINTS = 2048

# Bir noktanın "ilginç" sayılması için minimum skor
DETECTION_THRESHOLD = 0.005


## FİLTRELEME PARAMETRELERİ

# LightGlue güven skoru eşiği — altındaki eşleşmeler atılır
CONFIDENCE_THRESHOLD = 0.7

# Medyan akış filtresinde üst yüzde kaçı atılacak (85 = en sapan %15'i at)
FLOW_PERCENTILE = 85


## KEYFRAME PARAMETRELERİ

# Keyframe sayılması için minimum medyan piksel yer değişimi
MIN_MOTION_PX = 2.5

# Minimum eşleşme sayısı — altında geometrik hesaplama güvenilmez
MIN_MATCHES = 30

# İlk poz hesaplaması için minimum piksel yer değişimi
PARALLAX_THRESHOLD = 15.0



## İRTİFA PARAMETRELERİ

# EMA filtre katsayısı (0.3 = %30 yeni + %70 eski)
DEPTH_EMA_ALPHA = 0.3

# Ardışık keyframe'ler arası maksimum irtifa değişim oranı (%30)
MAX_ALTITUDE_CHANGE_RATIO = 0.30


## GEOMETRİ PARAMETRELERİ

# Homography inlier oranı eşiği — üstünde sahne düzlemsel kabul edilir
HOMOGRAPHY_INLIER_THRESHOLD = 0.70

# Salt rotasyon tespiti için maksimum medyan piksel hatası
ROTATION_ONLY_THRESHOLD_PX = 1.5

# MAGSAC++ reprojection error eşiği (piksel)
MAGSAC_THRESHOLD = 3.0



# FİZİKSEL DOĞRULAMA PARAMETRELERİ (Verison - 2)
# ============================================================================
# Zemin normali ardışık karelerde en fazla bu kadar derece değişebilir
MAX_NORMAL_CHANGE_DEG = 15.0

# Hareket yönü ardışık karelerde en fazla bu kadar derece değişebilir
MAX_DIRECTION_CHANGE_DEG = 60.0



# FAIL-SAFE PARAMETRELERİ (Version - 2)

# Üst üste bu kadar kare tracking kaybedilirse reset
MAX_LOST_FRAMES = 15

# Minimum inlier oranı — altında tracking kayıp kabul edilir
MIN_INLIER_RATIO = 0.3



## DERİNLİK MODELİ PARAMETRELERİ

# Metric3D model adı (torch.hub)
DEPTH_MODEL_NAME = "metric3d_vit_small"

# Metric3D ViT modelleri için giriş boyutu (yükseklik, genişlik)
DEPTH_INPUT_SIZE = (616, 1064)

# Kanonik odak uzaklığı (Metric3D eğitim parametresi)
DEPTH_CANONICAL_FOCAL = 1000.0



## VİDEO YOLU (varsayılan test videosu)
DEFAULT_VIDEO_PATH = "/home/mei/Benim/PROJECTS/Teknofest26/src/data/THYZ_2026_Ornek_Veri_1.MP4"



## ÇIKTI PARAMETRELERİ
DEFAULT_OUTPUT_PATH = "output.json"
TRAJECTORY_PLOT_PATH = "trajectory.png"
LOG_INTERVAL = 100  # Her N karede bir log yazdır



## BİREYSEL TEST
if __name__ == "__main__":
    print("=" * 60)
    print("  Teknofest 2026 — Pipeline Konfigürasyonu")
    print("=" * 60)

    print(f"\n Aktif Kamera Profili: {CAMERA_PROFILE}")
    print(f"   Görüntü Boyutu     : {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}")
    print(f"   Odak Uzaklığı      : fx={_active['fx']:.1f}, fy={_active['fy']:.1f}")
    print(f"   Optik Merkez       : cx={_active['cx']:.1f}, cy={_active['cy']:.1f}")
    print(f"   Distorsiyon        : {DIST_COEFFS}")
    print(f"\n   K Matrisi:")
    for row in CAMERA_K:
        print(f"     [{row[0]:>10.2f}  {row[1]:>8.2f}  {row[2]:>10.2f}]")

    print(f"\n Ön İşleme:")
    print(f"   CLAHE Clip Limit   : {CLAHE_CLIP_LIMIT}")
    print(f"   CLAHE Tile Size    : {CLAHE_TILE_SIZE}")
    print(f"   Blur Kernel        : {BLUR_KERNEL}")

    print(f"\n Özellik Çıkarımı:")
    print(f"   Max Keypoints      : {MAX_KEYPOINTS}")
    print(f"   Detection Threshold: {DETECTION_THRESHOLD}")

    print(f"\n Filtreleme:")
    print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   Flow Percentile     : {FLOW_PERCENTILE}")

    print(f"\n Keyframe:")
    print(f"   Min Motion (px)    : {MIN_MOTION_PX}")
    print(f"   Min Matches        : {MIN_MATCHES}")
    print(f"   Parallax Threshold : {PARALLAX_THRESHOLD}")

    print(f"\n İrtifa:")
    print(f"   EMA Alpha          : {DEPTH_EMA_ALPHA}")
    print(f"   Max Altitude Change: %{MAX_ALTITUDE_CHANGE_RATIO * 100:.0f}")

    print(f"\n Geometri:")
    print(f"   H Inlier Threshold : {HOMOGRAPHY_INLIER_THRESHOLD}")
    print(f"   Rotation Only (px) : {ROTATION_ONLY_THRESHOLD_PX}")
    print(f"   MAGSAC Threshold   : {MAGSAC_THRESHOLD}")

    print(f"\n Derinlik Modeli:")
    print(f"   Model              : {DEPTH_MODEL_NAME}")
    print(f"   Giriş Boyutu       : {DEPTH_INPUT_SIZE}")
    print(f"   Kanonik Focal       : {DEPTH_CANONICAL_FOCAL}")

    print(f"\n Yollar:")
    print(f"   Video              : {DEFAULT_VIDEO_PATH}")
    print(f"   Çıktı              : {DEFAULT_OUTPUT_PATH}")
    print(f"   Trajectory         : {TRAJECTORY_PLOT_PATH}")

    print(f"\n{'=' * 60}")
    print("   Tüm parametreler yüklendi")
    print("=" * 60)