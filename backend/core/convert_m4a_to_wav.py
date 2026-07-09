import os
import sys
import shutil

# Konfigurasi PATH untuk menambahkan binary ffmpeg dari imageio-ffmpeg
# Agar library audioread (yang digunakan librosa) dapat memproses format .m4a
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    
    # Audioread mencari "ffmpeg" atau "ffmpeg.exe". 
    # Karena imageio-ffmpeg menamainya "ffmpeg-win-x86_64-v7.1.exe", kita buat duplikatnya dengan nama "ffmpeg.exe"
    target_ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(target_ffmpeg_exe):
        print(f"[INFO] Membuat salinan ffmpeg.exe untuk kompatibilitas di: {target_ffmpeg_exe}")
        shutil.copy(ffmpeg_path, target_ffmpeg_exe)
    
    # Tambahkan ke variabel PATH sesi ini
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    print(f"[INFO] ffmpeg berhasil dideteksi dan dikonfigurasi di: {ffmpeg_dir}")
except Exception as e:
    print(f"[WARNING] Gagal mengonfigurasi ffmpeg otomatis: {e}")
    print("[WARNING] Pastikan imageio-ffmpeg terinstal dengan benar.")

import librosa
import soundfile as sf

def convert_m4a_to_wav(m4a_path, wav_path):
    """
    Mengonversi satu file .m4a ke .wav menggunakan librosa dan soundfile.
    Menjamin kualitas dan pitch suara tetap asli (tanpa resampling).
    """
    try:
        # Load audio (sr=None agar mempertahankan sample rate asli secara bit-perfect)
        # Ini menjamin tidak ada perubahan pitch atau penurunan kualitas suara.
        y, sr = librosa.load(m4a_path, sr=None)
        
        # Simpan ke format .wav (lossless PCM)
        sf.write(wav_path, y, sr)
        print(f"   [OK] BERHASIL: {os.path.basename(m4a_path)} -> {os.path.basename(wav_path)}")
        return True
    except Exception as e:
        # Menghindari karakter Unicode khusus agar tidak menyebabkan UnicodeEncodeError di Windows Console
        print(f"   [ERROR] GAGAL mengonversi {os.path.basename(m4a_path)}: {str(e)}")
        return False

def convert_recursive(parent_folder, delete_original=False):
    """
    Menelusuri folder utama dan mencari semua file .m4a di dalam folder utama
    maupun subfolder (seperti A, C, D, Dm, dst) lalu mengonversinya ke .wav.
    """
    if not os.path.exists(parent_folder):
        print(f"Folder parent tidak ditemukan: {parent_folder}")
        return

    print(f"Memindai folder: {parent_folder}")
    print("Mencari semua file .m4a di dalam subfolder...")
    
    total_found = 0
    success_count = 0
    m4a_files = []

    # 1. Kumpulkan semua file .m4a di semua subfolder
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith('.m4a'):
                full_path = os.path.join(root, file)
                m4a_files.append((root, file, full_path))
                total_found += 1

    if total_found == 0:
        print("Tidak ditemukan file .m4a di folder tersebut maupun subfoldernya.")
        return

    print(f"Ditemukan {total_found} file .m4a. Mulai mengonversi...")

    # 2. Lakukan konversi satu per satu
    for i, (root, file, full_path) in enumerate(m4a_files):
        # Buat nama file .wav di folder yang sama dengan file .m4a aslinya
        base_name = os.path.splitext(file)[0]
        wav_path = os.path.join(root, base_name + ".wav")
        
        print(f"[{i+1}/{total_found}] ", end="")
        if convert_m4a_to_wav(full_path, wav_path):
            success_count += 1
            # Hapus file .m4a lama jika parameter delete_original diaktifkan
            if delete_original:
                try:
                    os.remove(full_path)
                    print(f"      (File asli .m4a telah dihapus)")
                except Exception as e:
                    print(f"      (Gagal menghapus file asli: {e})")

    print("\n" + "="*40)
    print(f"SELESAI!")
    print(f"Total file ditemukan   : {total_found}")
    print(f"Berhasil dikonversi    : {success_count}")
    print(f"Gagal dikonversi       : {total_found - success_count}")
    print("="*40)

if __name__ == "__main__":
    # Tentukan folder utama dataset Anda yang berisi subfolder c, d, dm, dll.
    # Ubah path di bawah ini sesuai dengan lokasi folder dataset Anda.
    dataset_folder = r"g:\jammingdulu\dataset"
    
    # Set True jika Anda ingin otomatis menghapus file .m4a asli setelah sukses dikonversi ke .wav
    # Set False jika ingin tetap menyimpan file .m4a aslinya sebagai cadangan
    hapus_m4a_asli = False
    
    if os.path.exists(dataset_folder):
        convert_recursive(dataset_folder, delete_original=hapus_m4a_asli)
    else:
        print("=== Multi-Folder M4A to WAV Converter ===")
        print(f"Folder tidak ditemukan: {dataset_folder}")
        print("Silakan edit file ini dan sesuaikan variabel 'dataset_folder' dengan folder Anda.")
