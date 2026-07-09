import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set up figure size (Portrait layout, long vertically for maximum readability)
fig, ax = plt.subplots(figsize=(10, 16), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Data arsitektur layer kustom
layers = [
    {"name": "Input Layer", "detail": "Mel-Spectrogram Audio\nRescaling (1./255)", "shape": "(128, 128, 3)", "color": "#f1f5f9"}, # Abu-abu terang
    
    # Blok 1
    {"name": "Conv2D (Block 1)", "detail": "32 Filters, Kernel 3x3, Padding: same\nActivation: Relu", "shape": "(128, 128, 32)", "color": "#ffedd5"}, # Orange terang
    {"name": "Batch Normalization + MaxPool 1", "detail": "Batch Normalization\nMaxPooling2D (Pool: 2x2, stride: 2)\nDropout: 20%", "shape": "(64, 64, 32)", "color": "#ffe3e3"}, # Merah muda
    
    # Blok 2
    {"name": "Conv2D (Block 2)", "detail": "64 Filters, Kernel 3x3, Padding: same\nActivation: Relu", "shape": "(64, 64, 64)", "color": "#dbeafe"}, # Biru terang
    {"name": "Batch Normalization + MaxPool 2", "detail": "Batch Normalization\nMaxPooling2D (Pool: 2x2, stride: 2)\nDropout: 20%", "shape": "(32, 32, 64)", "color": "#e0f2fe"}, # Cyan terang
    
    # Blok 3
    {"name": "Conv2D (Block 3)", "detail": "128 Filters, Kernel 3x3, Padding: same\nActivation: Relu", "shape": "(32, 32, 128)", "color": "#dcfce7"}, # Hijau terang
    {"name": "Batch Normalization + MaxPool 3", "detail": "Batch Normalization\nMaxPooling2D (Pool: 2x2, stride: 2)\nDropout: 30%", "shape": "(16, 16, 128)", "color": "#f0fdf4"}, # Hijau toska terang
    
    # Klasifikasi
    {"name": "Flatten Layer", "detail": "Meratakan matriks 3D menjadi 1D Vector\n16 x 16 x 128 = 32.768 units", "shape": "(32768)", "color": "#fef9c3"}, # Kuning terang
    {"name": "Fully Connected (Dense)", "detail": "Dense: 128 units, Activation: Relu\nBatch Normalization\nDropout: 40%", "shape": "(128)", "color": "#fef08a"}, # Kuning pekat
    {"name": "Output Layer (Dense)", "detail": "Dense: 9 units (A, Am, C, D, Dm, E, Em, F, G)\nActivation: Softmax", "shape": "(9)", "color": "#ffe4e6"} # Merah/Rose terang
]

# Drawing settings
start_y = 92
x_center = 42
box_width = 38
box_height = 6.2  # Diperbesar dari 4.8 agar tidak menumpuk
spacing = 3.2     # Spacing disesuaikan

# Draw layers from top to bottom
for i, layer in enumerate(layers):
    y = start_y - i * (box_height + spacing)
    
    # Draw standard Rectangle
    rect = patches.Rectangle(
        (x_center - box_width/2, y - box_height/2), 
        box_width, box_height, 
        linewidth=1.5, 
        edgecolor='#475569', 
        facecolor=layer["color"],
        zorder=3
    )
    ax.add_patch(rect)
    
    # Judul Layer (Name) diletakkan di bagian atas kotak (va='top')
    ax.text(
        x_center - box_width/2 + 1.5, 
        y + 2.3, 
        layer["name"], 
        ha='left', 
        va='top', 
        fontsize=9.5, 
        fontweight='bold', 
        color='#0f172a',
        zorder=4
    )
    
    # Detail Layer diletakkan di bawah judul dengan jarak yang aman (va='top')
    ax.text(
        x_center - box_width/2 + 1.5, 
        y + 0.5, 
        layer["detail"], 
        ha='left', 
        va='top', 
        fontsize=8, 
        color='#334155', 
        fontweight='medium', 
        linespacing=1.3,
        zorder=4
    )
    
    # Output Shape Label (Diletakkan di sebelah kanan box)
    shape_x = x_center + box_width/2 + 4
    ax.text(
        shape_x, 
        y, 
        f"Output Shape:\n{layer['shape']}", 
        ha='left', 
        va='center', 
        fontsize=9, 
        fontweight='bold', 
        color='#0f172a', 
        bbox=dict(boxstyle="square,pad=0.4", edgecolor='#cbd5e1', facecolor='#f8fafc', lw=1),
        zorder=4
    )
    
    # Garis hubung tipis dari box ke label output shape
    ax.plot([x_center + box_width/2, shape_x - 0.5], [y, y], color='#94a3b8', linestyle=':', lw=1.5, zorder=2)
    
    # Draw connection arrow to next box (except the last one)
    if i < len(layers) - 1:
        arrow_top_y = y - box_height/2 - 0.1
        arrow_bottom_y = y - box_height/2 - spacing + 0.1
        ax.annotate('', xy=(x_center, arrow_bottom_y), xytext=(x_center, arrow_top_y),
                    arrowprops=dict(arrowstyle="-|>", color='#475569', lw=2, mutation_scale=12, shrinkA=0, shrinkB=0), zorder=2)

# Add grouping brackets on the left side
# 1. Feature Extraction Group (Layers 1 to 7)
y_feat_top = start_y + box_height/2 + 1
y_feat_bottom = start_y - 6 * (box_height + spacing) - box_height/2 - 1
ax.plot([14, 11, 11, 14], [y_feat_top, y_feat_top, y_feat_bottom, y_feat_bottom], color='#334155', lw=2)
ax.text(8, (y_feat_top + y_feat_bottom)/2, "BLOK EKSTRAKSI FITUR\n(CNN FEATURE EXTRACTION)", ha='center', va='center', rotation=90, fontsize=10.5, fontweight='bold', color='#0f172a')

# 2. Classification Group (Layers 8 to 10)
y_class_top = start_y - 7 * (box_height + spacing) + box_height/2 + 1
y_class_bottom = start_y - 9 * (box_height + spacing) - box_height/2 - 1
ax.plot([14, 11, 11, 14], [y_class_top, y_class_top, y_class_bottom, y_class_bottom], color='#334155', lw=2)
ax.text(8, (y_class_top + y_class_bottom)/2, "BLOK KLASIFIKASI\n(CLASSIFICATION HEAD)", ha='center', va='center', rotation=90, fontsize=10.5, fontweight='bold', color='#0f172a')

# Save plot
output_dir = r"g:\jammingdulu\backend\models"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "cnn_architecture.png")
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
plt.close()

print(f"Vertical CNN architecture diagram successfully saved at: {save_path}")
