import matplotlib.pyplot as plt
import numpy as np
import os

# Data kelas chord
chords = ['A', 'Am', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G']
training_counts = [96] * 9
validation_counts = [24] * 9

# Lokasi X untuk grup
x = np.arange(len(chords))
width = 0.35  # Lebar bar

# Set style visual agar terlihat modern
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Membuat bar grouped
rects1 = ax.bar(x - width/2, training_counts, width, label='Training Set (80% / Total: 864)', color='#38bdf8') # Biru langit
rects2 = ax.bar(x + width/2, validation_counts, width, label='Validation Set (20% / Total: 216)', color='#00d285') # Hijau toska

# Menambahkan keterangan teks dan label
ax.set_ylabel('Jumlah Gambar Spektrogram', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlabel('Kelas Akord (Chord)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Distribusi Data Spektrogram untuk Training & Validation per Kelas', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(chords, fontsize=11, fontweight='semibold')
ax.set_ylim(0, 120)
ax.legend(fontsize=11, loc='upper right', frameon=True)

# Fungsi untuk memberi label nilai di atas bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# Tentukan folder penyimpanan
output_dir = r"g:\jammingdulu\backend\models"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "dataset_distribution.png")

# Simpan gambar
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"Grafik distribusi dataset berhasil disimpan di: {save_path}")
