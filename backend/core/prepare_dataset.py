import os
import glob
from audio_processing import process_and_save_for_training

# Konfigurasi Path Dataset
RAW_AUDIO_DIR = r"g:\jammingdulu\dataset\raw_audio"
SPECTROGRAM_DIR = r"g:\jammingdulu\dataset\spectrograms"
CHORDS = ['A', 'Am', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G']

def build_dataset():
    print("Memulai konversi rekaman Audio Mentah (.wav) ke Mel-Spectrogram Image...\n")
    total_processed = 0
    total_failed = 0

    for chord in CHORDS:
        # Tentukan letak direktori sumber dan tujuan
        input_folder = os.path.join(RAW_AUDIO_DIR, chord)
        output_folder = os.path.join(SPECTROGRAM_DIR, chord)

        # Pastikan direktori tujuan sudah ada
        os.makedirs(output_folder, exist_ok=True)

        # Cari semua file audio (.wav) di folder sumber
        wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
        
        if len(wav_files) == 0:
            print(f"[SKIP] Chord '{chord}': Tidak ada file .wav ditemukan di {input_folder}")
            continue

        print(f"Memproses {len(wav_files)} file untuk Chord '{chord}'...")
        
        for wav_path in wav_files:
            # Ambil nama file (misal: "my_chord_1.wav")
            filename = os.path.basename(wav_path)
            
            # Ubah ekstensinya menjadi .png (misal: "my_chord_1.png")
            img_filename = filename.replace(".wav", ".png")
            output_path = os.path.join(output_folder, img_filename)

            # Jika Anda pernah menjalankan skrip ini sebelumnya, file yang sudah ada akan dilewati (menghemat waktu)
            if os.path.exists(output_path):
                # print(f"  [SKIPPED] {img_filename} sudah ada.")
                continue

            # Jalankan konversi! (menggunakan fungsi yang kita buat di Part 1)
            success = process_and_save_for_training(wav_path, output_path)
            
            if success:
                total_processed += 1
            else:
                total_failed += 1

    print("\n" + "="*50)
    print("PROSES SELESAI!")
    print(f"Total file baru yang dikonversi : {total_processed}")
    print(f"Gagal memproses                 : {total_failed}")
    print("Sekarang Anda bisa menjalankan 'train_model.py'!")
    print("="*50)

if __name__ == "__main__":
    build_dataset()
