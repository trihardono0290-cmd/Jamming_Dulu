import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
import matplotlib.pyplot as plt
import os

# 1. Bangun Model Custom CNN yang sama persis
IMG_HEIGHT, IMG_WIDTH = 128, 128
num_classes = 9

model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Blok Conv 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    # Blok Conv 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    # Blok Conv 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    # Klasifikasi
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

# 2. Tangkap output model.summary() ke dalam string
stream = io.StringIO()
model.summary(print_fn=lambda x: stream.write(x + '\n'))
summary_string = stream.getvalue()
stream.close()

# 3. Render string tersebut ke Matplotlib Gambar
fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
ax.axis('off')

# Tulis teks summary dengan font monospaced (Courier) agar tabelnya lurus rapi
ax.text(
    0.01, 0.99, 
    summary_string, 
    fontfamily='monospace', 
    fontsize=8.5, 
    va='top', 
    ha='left',
    color='#0f172a',
    linespacing=1.25
)

# Tentukan folder penyimpanan
output_dir = r"g:\jammingdulu\backend\models"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "model_summary.png")

# Simpan gambar
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
plt.close()

print(f"Gambar Model Summary berhasil disimpan di: {save_path}")
