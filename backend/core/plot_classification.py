import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set up figure
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

# Title
ax.text(50, 56, "DETAIL MODEL ARSITEKTUR BLOK KLASIFIKASI AKHIR", ha='center', va='center', fontsize=13, fontweight='bold', color='#0f172a')

# 1. Gambar representasi Blok Fitur 3D (16x16x128)
# Sisi depan
rect_front = patches.Rectangle((5, 18), 10, 15, linewidth=1.5, edgecolor='#1e293b', facecolor='#86efac', alpha=0.9, zorder=4)
ax.add_patch(rect_front)
# Sisi atas
polygon_top = patches.Polygon([[5, 33], [10, 38], [20, 38], [15, 33]], linewidth=1.5, edgecolor='#1e293b', facecolor='#a7f3d0', alpha=0.9, zorder=3)
ax.add_patch(polygon_top)
# Sisi samping
polygon_side = patches.Polygon([[15, 18], [20, 23], [20, 38], [15, 33]], linewidth=1.5, edgecolor='#1e293b', facecolor='#34d399', alpha=0.9, zorder=3)
ax.add_patch(polygon_side)

ax.text(12.5, 25.5, "Fitur Blok 3D\n16 x 16 x 128", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a', zorder=5)

# 2. Gambar Representasi Flatten (32.768 units)
# Gambar bar vertikal tinggi tipis
rect_flat = patches.Rectangle((32, 10), 3, 31, linewidth=1.5, edgecolor='#1e293b', facecolor='#fef9c3', zorder=4)
ax.add_patch(rect_flat)
ax.text(33.5, 25.5, "F\nL\nA\nT\nT\nE\nN", ha='center', va='center', fontsize=9, fontweight='bold', color='#854d0e', zorder=5)
ax.text(33.5, 6, "1D Vector\n(32.768 unit)", ha='center', va='top', fontsize=8, fontweight='bold', color='#0f172a')

# 3. Gambar Representasi Dense Layer (128 Neurons)
# Kita gambarkan 5 lingkaran dengan titik-titik di tengah untuk melambangkan 128 neuron
dense_y = [14, 20, 26, 32, 38]
for y in dense_y:
    circle = patches.Circle((58, y), 2, linewidth=1.5, edgecolor='#1e293b', facecolor='#fef08a', zorder=4)
    ax.add_patch(circle)
ax.text(58, 28, "...\n...\n...", ha='center', va='center', fontsize=12, fontweight='bold', color='#854d0e', zorder=3)
ax.text(58, 6, "Fully Connected\nDense (128 unit)\n+ Relu", ha='center', va='top', fontsize=8, fontweight='bold', color='#0f172a')

# 4. Gambar Representasi Output Layer (9 Neurons / Chords)
chords = ['A', 'Am', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G']
output_y = np.linspace(10, 42, 9)
for idx, y in enumerate(output_y):
    circle = patches.Circle((84, y), 1.6, linewidth=1.5, edgecolor='#1e293b', facecolor='#ffe4e6', zorder=4)
    ax.add_patch(circle)
    # Tulis nama chord di dalam lingkaran
    ax.text(84, y, chords[idx], ha='center', va='center', fontsize=7, fontweight='bold', color='#9f1239', zorder=5)
ax.text(84, 6, "Output Layer\nDense (9 unit)\n+ Softmax", ha='center', va='top', fontsize=8, fontweight='bold', color='#0f172a')

# --- Garis Koneksi & Panah ---

# Panah 1: Fitur 3D ke Flatten
ax.annotate('', xy=(31, 25.5), xytext=(21.5, 25.5),
            arrowprops=dict(arrowstyle="-|>", color='#475569', lw=2, mutation_scale=12))

# Garis Koneksi Padat 1: Flatten ke Dense (Gambarkan representasi garis silang)
for dy in dense_y:
    ax.plot([35, 56], [25.5, dy], color='#cbd5e1', lw=0.8, linestyle='-', zorder=1)

# Garis Koneksi Padat 2: Dense ke Output (Gambarkan representasi garis silang)
for dy in dense_y:
    for oy in output_y:
        ax.plot([60, 82.4], [dy, oy], color='#e2e8f0', lw=0.5, linestyle='-', zorder=1)

# --- Label Penjelas di Atas Bagan ---
# Keterangan proses Flattening
ax.text(26.5, 45, "Flattening\n(Perataan Dimensi)", ha='center', va='bottom', fontsize=8, fontweight='semibold', color='#475569')

# Keterangan Dropout & Batch Normalization
ax.text(58, 45, "Batch Normalization\n& Dropout (40%)", ha='center', va='bottom', fontsize=8, fontweight='semibold', color='#475569')

# Save plot
output_dir = r"g:\jammingdulu\backend\models"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "classification_head.png")
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
plt.close()

print(f"Classification head diagram successfully saved at: {save_path}")
