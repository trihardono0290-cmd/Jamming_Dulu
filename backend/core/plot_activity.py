import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set up figure size (Portrait layout for vertical swimlanes)
fig, ax = plt.subplots(figsize=(11, 15), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title
ax.text(50, 97, "ACTIVITY DIAGRAM SISTEM GitarLive", ha='center', va='center', fontsize=14, fontweight='bold', color='#0f172a')

# 1. Swimlane Boundaries & Lines
# Swimlane 1: Pengguna (0 to 30)
# Swimlane 2: Frontend Web (30 to 68)
# Swimlane 3: Backend FastAPI (68 to 100)
ax.plot([30, 30], [5, 93], color='#475569', lw=1.5)
ax.plot([68, 68], [5, 93], color='#475569', lw=1.5)
ax.plot([5, 95], [87, 87], color='#475569', lw=2) # Header horizontal line

# Swimlane Headers
ax.text(15, 90, "PENGGUNA (USER)", ha='center', va='center', fontsize=11, fontweight='bold', color='#1e293b')
ax.text(49, 90, "APLIKASI WEB (FRONTEND)", ha='center', va='center', fontsize=11, fontweight='bold', color='#1e293b')
ax.text(84, 90, "SERVER BACKEND (FASTAPI)", ha='center', va='center', fontsize=11, fontweight='bold', color='#1e293b')

# 2. Activity Drawing Parameters
box_w = 24
box_h = 3.6
x_u = 15
x_f = 49
x_b = 84

# Helper function to draw standard rectangular boxes for activity states
def draw_act(ax, x, y, text, bg_color):
    rect = patches.Rectangle(
        (x - box_w/2, y - box_h/2), box_w, box_h, 
        linewidth=1.2, edgecolor='#334155', facecolor=bg_color, 
        zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=7.5, fontweight='semibold', color='#0f172a', zorder=4)

# --- Start Node (User) ---
start_y = 83
circle_start = patches.Circle((x_u, start_y), 1.2, facecolor='#0f172a', edgecolor='none', zorder=3)
ax.add_patch(circle_start)

# Arrow from Start to UC1
ax.annotate('', xy=(x_u, 78.5), xytext=(x_u, start_y), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# Activity 1: User memilih Target Chord (Y=76)
draw_act(ax, x_u, 76, "Memilih Target Chord\npada Menu", "#eff6ff")
ax.annotate('', xy=(x_u, 69.5), xytext=(x_u, 74.2), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# Activity 2: User menekan tombol Start (Y=67)
draw_act(ax, x_u, 67, "Menekan Tombol 'Start'", "#eff6ff")
# Arrow to Frontend
ax.annotate('', xy=(x_f - 2, 60.5), xytext=(x_u + 4, 65.5), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# Activity 3: Frontend mengaktifkan mic (Y=60)
draw_act(ax, x_f, 59, "Mengaktifkan Mic &\nMemulai Perekaman", "#f8fafc")
# Arrow to User
ax.annotate('', xy=(x_u + 2, 53.5), xytext=(x_f - 4, 57.5), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# Activity 4: User memetik gitar & tekan Stop (Y=51)
draw_act(ax, x_u, 51, "Memainkan Gitar &\nMenekan Tombol 'Stop'", "#eff6ff")
# Arrow to Frontend
ax.annotate('', xy=(x_f - 2, 44.5), xytext=(x_u + 4, 49.5), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# Activity 5: Frontend stop recording & compile WAV (Y=43)
draw_act(ax, x_f, 43, "Menghentikan Perekaman &\nMenyusun File WAV", "#f8fafc")
# Arrow to User
ax.annotate('', xy=(x_u + 2, 37.5), xytext=(x_f - 4, 41.5), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# Activity 6: User menekan tombol koreksi (Y=35)
draw_act(ax, x_u, 35, "Menekan Tombol\n'Koreksi Rekaman'", "#eff6ff")
# Arrow to Frontend
ax.annotate('', xy=(x_f - 2, 30.5), xytext=(x_u + 4, 33.5), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# Activity 7: Frontend kirim data audio via WebSocket (Y=29)
draw_act(ax, x_f, 29, "Mengirim Binary Audio\nvia WebSocket", "#f8fafc")
# Arrow to Backend
ax.annotate('', xy=(x_b - 2, 23.5), xytext=(x_f + 4, 27.5), arrowprops=dict(arrowstyle="-|>", color='#3b82f6', lw=1.5, ls='--'))

# Activity 8: Backend proses audio & ekstraksi Mel-Spec (Y=21)
draw_act(ax, x_b, 21, "Preprocessing Sinyal &\nEkstraksi Mel-Spectrogram", "#f0fdf4")
ax.annotate('', xy=(x_b, 14.5), xytext=(x_b, 19.2), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# Activity 9: Backend klasifikasi chord CNN & kirim hasil (Y=12)
draw_act(ax, x_b, 12, "Klasifikasi Model CNN &\nKirim Hasil via WebSocket", "#f0fdf4")
# Arrow back to Frontend
ax.annotate('', xy=(x_f + 4, 9.5), xytext=(x_b - 4, 11), arrowprops=dict(arrowstyle="-|>", color='#3b82f6', lw=1.5, ls='--'))

# Activity 10: Frontend terima data & membandingkan hasil (Y=8)
draw_act(ax, x_f, 8, "Mencocokkan Hasil Prediksi\ndengan Target Chord", "#f8fafc")
ax.annotate('', xy=(x_f, 15), xytext=(x_f, 9.8), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# --- Decision Diamond (Frontend) (Y=17) ---
# Diamond points: (x, y)
diamond_pts = [[x_f, 19.5], [x_f + 3, 17], [x_f, 14.5], [x_f - 3, 17]]
diamond = patches.Polygon(diamond_pts, linewidth=1.5, edgecolor='#334155', facecolor='#fef08a', zorder=3)
ax.add_patch(diamond)
ax.text(x_f, 17, "Sama?", ha='center', va='center', fontsize=7, fontweight='bold', color='#854d0e', zorder=4)

# Path 1: Ya (Benar) -> Y=24
ax.annotate('', xy=(x_f - 6, 24), xytext=(x_f - 2, 18), arrowprops=dict(arrowstyle="-|>", color='#10b981', lw=1.5))
ax.text(x_f - 5, 19.5, "Ya", ha='right', va='bottom', fontsize=7, fontweight='bold', color='#10b981')
draw_act(ax, x_f - 9, 24, "Mainkan 'correct.mp3'\nTampilkan Panel Sukses\nDiagram & Panduan Jari", "#ecfdf5")

# Path 2: Tidak (Salah) -> Y=24
ax.annotate('', xy=(x_f + 6, 24), xytext=(x_f + 2, 18), arrowprops=dict(arrowstyle="-|>", color='#ef4444', lw=1.5))
ax.text(x_f + 5, 19.5, "Tidak", ha='left', va='bottom', fontsize=7, fontweight='bold', color='#ef4444')
draw_act(ax, x_f + 9, 24, "Mainkan 'wrong.mp3'\nTampilkan Koreksi Salah\nDiagram & Panduan Jari", "#fef2f2")

# Merge arrows to User final display
ax.annotate('', xy=(x_u + 3, 14), xytext=(x_f - 9, 21.5), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.2))
ax.annotate('', xy=(x_u + 3, 12), xytext=(x_f + 9, 21.5), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.2))

# Activity 11: User menerima hasil & evaluasi (Y=13)
draw_act(ax, x_u, 13, "Melihat Hasil Prediksi &\nEvaluasi Jari Tangan", "#eff6ff")
ax.annotate('', xy=(x_u, 7.5), xytext=(x_u, 10.8), arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.5))

# --- End Node (User) (Y=6) ---
circle_end_outer = patches.Circle((x_u, 5.5), 1.2, fill=False, edgecolor='#0f172a', lw=1.5, zorder=3)
circle_end_inner = patches.Circle((x_u, 5.5), 0.7, facecolor='#0f172a', edgecolor='none', zorder=4)
ax.add_patch(circle_end_outer)
ax.add_patch(circle_end_inner)

# Save plot
output_dir = r"g:\jammingdulu\backend\models"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "activity_diagram.png")
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
plt.close()

print(f"Activity diagram successfully saved at: {save_path}")
