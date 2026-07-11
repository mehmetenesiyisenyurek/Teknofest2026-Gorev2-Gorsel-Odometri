"""
Teknofest 2026 — Visual Odometry Pipeline
Ana Giriş ve Çalıştırma Arayüzü (run.py)

Bu dosya, Visual Odometry sisteminin çalıştırılması, donanım kontrollerinin
yapılması ve çevrimdışı/çevrimiçi çalışma modlarının yönetilmesi için
sıfırdan tasarlanmış ana giriş noktasıdır.

Geliştirme prensipleri:
    - Donanım ve bağımlılıkların en başta taranması (Fail-Early).
    - Hata durumlarının kullanıcı dostu açıklama ve yönlendirmelerle raporlanması.
    - CLI argümanlarının merkezi config.py ile entegre varsayılan değerlerle beslenmesi.
    - Yarışma anında çalıştırılacak olan sunucu protokolü için açık mimari yapısı.

Test için:
python run.py --mode offline --video /home/mei/Benim/PROJECTS/Teknofest26/src/data/THYZ_2026_Ornek_Veri_1.MP4 --gt /home/mei/Benim/PROJECTS/Teknofest26/src/data/THYZ_2026_Ornek_Veri_1_translation.csv --max-frames 200 --gps-cut 50
"""

import os
import sys
import time
import argparse
from pathlib import Path

# src dizinini python modül arama yoluna güvenli şekilde ekle
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Merkezi konfigürasyon dosyasını en başta yükle
try:
    import config as cfg
except ImportError:
    print("=" * 60)
    print("❌ HATA: 'config.py' dosyası ana dizinde bulunamadı!")
    print("   Lütfen projenin kök dizininde olduğunuzdan emin olun.")
    print("=" * 60)
    sys.exit(1)


def check_environment():
    """
    Python sürümünü ve VO hattı için gerekli kritik kütüphaneleri denetler.
    Herhangi bir eksiklik durumunda projenin çökmesini önlemek için erken uyarı verir.
    """
    print("\n" + "=" * 60)
    print("🔍 Ortam ve Bağımlılık Denetimi")
    print("=" * 60)

    # 1. Python Sürümü Kontrolü
    print(f"   Python Sürümü   : {sys.version.split()[0]} (Önerilen: 3.8+)")
    if sys.version_info < (3, 8):
        print("   ⚠️  DİKKAT: Python sürümünüz 3.8'den eski. Uyumluluk sorunları yaşanabilir.")

    # 2. Kritik Paketlerin Varlığı
    required_packages = {
        "numpy": "NumPy",
        "cv2": "OpenCV (opencv-python)",
        "torch": "PyTorch",
        "scipy": "SciPy",
        "timm": "Timm (Deep Learning Backbones)",
    }

    missing_packages = []
    for import_name, display_name in required_packages.items():
        try:
            module = __import__(import_name)
            ver = getattr(module, "__version__", "Sürüm Bilgisi Yok")
            print(f"   ✅ {display_name:<16} : Yüklü (Versiyon: {ver})")
        except ImportError:
            print(f"   ❌ {display_name:<16} : YÜKLÜ DEĞİL!")
            missing_packages.append(display_name)

    # LightGlue kontrolü
    try:
        import lightglue
        print(f"   ✅ LightGlue        : Yüklü")
    except ImportError:
        print(f"   ❌ LightGlue        : YÜKLÜ DEĞİL!")
        missing_packages.append("LightGlue (pip install git+https://github.com/cvg/LightGlue.git)")

    if missing_packages:
        print("\n🚨 ÇALIŞMA ENGELLENDİ: Kritik bağımlılıklar eksik!")
        print("Lütfen pip aracılığıyla eksik paketleri yükleyin:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nPratik çözüm: pip install -r requirements.txt")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60 + "\n")


def log_hardware_diagnostics(device_arg):
    """
    Hesaplama yapılacak cihazın (CPU/GPU) yeteneklerini analiz eder ve raporlar.
    """
    print("=" * 60)
    print("🖥️  Donanım ve Hesaplama Cihazı Teşhisi")
    print("=" * 60)

    import torch

    # CPU Bilgileri
    print(f"   Mevcut CPU Çekirdek Sayısı : {os.cpu_count()}")

    # CUDA / GPU Bilgileri
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA (GPU) Desteği        : {'MEVCUT ✅' if cuda_available else 'YOK ❌'}")

    active_device = device_arg
    if device_arg == "cuda":
        if not cuda_available:
            print("   ⚠️  UYARI: GPU seçildi fakat CUDA aktif değil. CPU moduna geçiliyor.")
            active_device = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_cap = torch.cuda.get_device_capability(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   Aktif GPU Aygıtı           : {gpu_name}")
            print(f"   CUDA Compute Capability    : {gpu_cap[0]}.{gpu_cap[1]}")
            print(f"   Toplam VRAM                : {gpu_mem_gb:.2f} GB")

    print(f"   Çalışma Cihazı (Aygıt)     : {active_device.upper()}")
    print("=" * 60 + "\n")
    return active_device


def run_offline(args, active_device):
    """
    Offline simülasyon modunu çalıştırır. Yerel video üzerinde
    geliştirilen VO pipeline'ını koşturur ve çıktı üretir.
    """
    video_path = Path(args.video)
    gt_path = Path(args.gt) if args.gt else None
    output_path = Path(args.output)

    print("=" * 60)
    print("🎬 ÇEVRİMDIŞI (OFFLINE) TEST BAŞLATILIYOR")
    print("=" * 60)
    print(f"   Test Videosu   : {video_path.resolve()}")
    if gt_path:
        print(f"   Ground Truth   : {gt_path.resolve()}")
    else:
        print(f"   Ground Truth   : Sağlanmadı (ATE analizi atlanacak)")
    print(f"   Kayıt Hedefi   : {output_path.resolve()}")
    print(f"   GPS Kesme Nok. : {args.gps_cut}. kare")
    print("-" * 60)

    # Giriş dosyalarını kontrol et
    if not video_path.exists():
        print(f"❌ HATA: Video dosyası bulunamadı: {video_path}")
        sys.exit(1)

    if gt_path and not gt_path.exists():
        print(f"❌ HATA: Ground Truth dosyası bulunamadı: {gt_path}")
        sys.exit(1)

    # Çıktı klasörünü garanti altına al
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    try:
        # Pipeline modülünü gecikmeli import et
        from src.pipeline import run_pipeline

        # Pipeline'ı koştur
        run_pipeline(
            video_path=str(video_path),
            config_module=cfg,
            gt_path=str(gt_path) if gt_path else None,
            max_frames=args.max_frames,
            device=active_device,
            output_path=str(output_path),
            gps_cut_start=args.gps_cut,
            verbose=True,
        )

        elapsed = time.time() - t_start
        print(f"\n🎉 Çevrimdışı analiz başarıyla sonuçlandı! Toplam süre: {elapsed:.2f} saniye.")

    except KeyboardInterrupt:
        print("\n🛑 Kullanıcı tarafından kesildi (Ctrl+C). Kısmi veriler kaydedilmiş olabilir.")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Pipeline çalışırken beklenmedik bir hata oluştu:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_server(args):
    """
    Hafta 2 kapsamındaki yarışma sunucusu entegrasyon arayüzünün iskelet yapısı.
    """
    print("=" * 60)
    print("📡 SUNUCU ENTEGRASYON MODU (HAFTA 2)")
    print("=" * 60)
    print(f"   Sunucu Adresi  : {args.ip}:{args.port}")
    print(f"   Kamera Profili : {cfg.CAMERA_PROFILE} ({cfg.IMAGE_SIZE[0]}x{cfg.IMAGE_SIZE[1]})")
    print("-" * 60)

    print("\nℹ️  HAFTA 2 İLETİŞİM VE İŞLEME DÖNGÜSÜ AKIŞI:")
    print("   1. Sunucu ile TCP/IP protokolü üzerinden el sakinleşme sağlanır.")
    print("   2. Jüri sunucusundan her döngüde frame verisi ve GPS sağlık durumu alınır.")
    print("   3. gps_health_status == 1 (GPS aktif):")
    print("      - Gelen jüri konumu (x, y, z) referans kabul edilir.")
    print("      - Poz biriktirici (pose.py) ve Kalibratör (calibration.py) güncellenir.")
    print("      - Konum aynen sunucuya geri yansıtılır.")
    print("   4. gps_health_status == 0 (GPS kesik):")
    print("      - Son kare ile yeni kare arasındaki homografi çözülür.")
    print("      - Derinlik modelinden irtifa çekilir ve yer değiştirme hesaplanır.")
    print("      - Kalibre edilmiş dünya konumu sunucuya geri gönderilir.")

    print("\n🚨 BİLGİ: Soket API entegrasyonu, yarışma jürisinin teknik dokümanı")
    print("   paylaşmasıyla birlikte Hafta 2'de buraya eklenecektir.")

    raise NotImplementedError(
        "Soket entegrasyon kodu Hafta 2 kapsamında yarışma API dokümanı ile yazılacaktır."
    )


def main():
    # Argüman yönetimi ve parser kurulumu
    parser = argparse.ArgumentParser(
        description="Teknofest 2026 — Visual Odometry Pipeline Çalıştırıcı Arayüzü",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Temel Mod ve Donanım Parametreleri
    parser.add_argument(
        "--mode", type=str, default="offline", choices=["offline", "server"],
        help="Çalışma modu:\n"
             "  offline : Yerel video ve GT dosyaları ile çevrimdışı analiz.\n"
             "  server  : Jüri yarışma sunucusuna bağlanarak canlı veri işleme."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Model hesaplama aygıtı (varsayılan: cuda)"
    )

    # Offline Mod Parametreleri
    offline_opt = parser.add_argument_group("Çevrimdışı (Offline) Mod Seçenekleri")
    offline_opt.add_argument(
        "--video", type=str, default=cfg.DEFAULT_VIDEO_PATH,
        help="Analiz edilecek test videosunun yolu."
    )
    offline_opt.add_argument(
        "--gt", type=str, default=None,
        help="Doğrulama ve ATE hesaplaması için Ground Truth CSV/TSV dosyası."
    )
    offline_opt.add_argument(
        "--max-frames", type=int, default=None,
        help="Video içinde işlenecek maksimum kare sınırı (Belirtilmezse tümü)."
    )
    offline_opt.add_argument(
        "--output", type=str, default=cfg.DEFAULT_OUTPUT_PATH,
        help="Çıktı pozisyonlarının kaydedileceği JSON dosyası."
    )
    offline_opt.add_argument(
        "--gps-cut", type=int, default=450,
        help="Çevrimdışı simülasyonda GPS sinyalinin kesileceği kare numarası (varsayılan: 450)."
    )

    # Sunucu Modu Parametreleri
    server_opt = parser.add_argument_group("Sunucu (Server) Modu Seçenekleri")
    server_opt.add_argument(
        "--ip", type=str, default="127.0.0.1",
        help="Yarışma sunucusu IP adresi."
    )
    server_opt.add_argument(
        "--port", type=int, default=8080,
        help="Yarışma sunucusu port numarası."
    )

    args = parser.parse_args()

    # 1. Ortam kontrolü
    check_environment()

    # 2. Donanım tespiti ve loglama
    active_device = log_hardware_diagnostics(args.device)

    # 3. Moda göre yönlendirme
    if args.mode == "offline":
        run_offline(args, active_device)
    elif args.mode == "server":
        run_server(args)


if __name__ == "__main__":
    main()