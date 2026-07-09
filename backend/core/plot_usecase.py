import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set up figure size
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title
ax.text(50, 95, "USE CASE DIAGRAM SISTEM GitarLive", ha='center', va='center', fontsize=14, fontweight='bold', color='#0f172a')

# 1. System Boundary Box
boundary = patches.Rectangle((28, 5), 44, 82, linewidth=2, edgecolor='#334155', facecolor='#f8fafc', zorder=1)
ax.add_patch(boundary)
ax.text(50, 84, "Sistem GitarLive (Aplikasi Web)", ha='center', va='center', fontsize=11, fontweight='bold', color='#1e293b')

# 2. Draw Left Actor (User / Pengguna)
# Head
circle_user = patches.Circle((12, 50), 3, linewidth=2, edgecolor='#0f172a', facecolor='none', zorder=3)
ax.add_patch(circle_user)
# Body
ax.plot([12, 12], [47, 37], color='#0f172a', lw=2, zorder=3)
# Arms
ax.plot([6, 12, 18], [42, 44, 42], color='#0f172a', lw=2, zorder=3)
# Legs
ax.plot([6, 12, 18], [28, 37, 28], color='#0f172a', lw=2, zorder=3)
# Label
ax.text(12, 23, "Pengguna\n(Pemain Gitar)", ha='center', va='top', fontsize=10, fontweight='bold', color='#0f172a')

# 3. Draw Right Actor (Backend FastAPI Server)
# Head
circle_sys = patches.Circle((88, 50), 3, linewidth=2, edgecolor='#0f172a', facecolor='none', zorder=3)
ax.add_patch(circle_sys)
# Body
ax.plot([88, 88], [47, 37], color='#0f172a', lw=2, zorder=3)
# Arms
ax.plot([82, 88, 94], [42, 44, 42], color='#0f172a', lw=2, zorder=3)
# Legs
ax.plot([82, 88, 94], [28, 37, 28], color='#0f172a', lw=2, zorder=3)
# Label
ax.text(88, 23, "Sistem Backend\n(FastAPI Server)", ha='center', va='top', fontsize=10, fontweight='bold', color='#0f172a')

# 4. Define Use Cases inside system boundary: (ID, Name, Y position)
use_cases = [
    ("UC1", "Memilih Target Chord", 74, "#eff6ff"),
    ("UC2", "Merekam Petikan Gitar", 60, "#eff6ff"),
    ("UC3", "Memutar Ulang Rekaman", 46, "#eff6ff"),
    ("UC4", "Meminta Koreksi Chord\n(WebSocket)", 32, "#eff6ff"),
    ("UC5", "Mengklasifikasikan Chord\n(CNN Model)", 18, "#ecfdf5"), # Warna hijau toska untuk inti AI
    ("UC6", "Melihat Hasil & Panduan Jari", 10, "#eff6ff")
]

# Draw Use Cases as Ovals/Ellipses
uc_patches = {}
for id_name, label, y, color in use_cases:
    ellipse = patches.Ellipse((50, y), 32, 8, linewidth=1.5, edgecolor='#1e3a8a', facecolor=color, zorder=2)
    ax.add_patch(ellipse)
    ax.text(50, y, label, ha='center', va='center', fontsize=8.5, fontweight='semibold', color='#1e293b', zorder=3)
    uc_patches[id_name] = y

# 5. Connect Actor User to Use Cases (UC1, UC2, UC3, UC4, UC6)
user_connections = ["UC1", "UC2", "UC3", "UC4", "UC6"]
for uc in user_connections:
    uc_y = uc_patches[uc]
    # Draw simple connector line
    ax.plot([17, 34], [43, uc_y], color='#475569', lw=1.2, zorder=2)

# 6. Connect Actor Backend to Use Cases (UC4, UC5)
backend_connections = ["UC4", "UC5"]
for uc in backend_connections:
    uc_y = uc_patches[uc]
    # Draw simple connector line
    ax.plot([83, 66], [43, uc_y], color='#475569', lw=1.2, zorder=2)

# 7. Draw <<include>> dependency between UC4 and UC5
# Menghubungkan meminta koreksi dengan proses klasifikasi oleh CNN
ax.annotate(
    '', 
    xy=(50, 22.4), xytext=(50, 27.6),
    arrowprops=dict(arrowstyle="->", color='#3b82f6', lw=1.5, linestyle='--', shrinkA=0, shrinkB=0),
    zorder=2
)
ax.text(51, 25, "<<include>>", ha='left', va='center', fontsize=7.5, color='#2563eb', fontweight='bold')

# Save plot
output_dir = r"g:\jammingdulu\backend\models"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "usecase_diagram.png")
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
plt.close()

print(f"Use Case diagram successfully saved at: {save_path}")
