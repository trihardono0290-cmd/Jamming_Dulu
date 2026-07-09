import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# KONFIGURASI TRAINING
# ==========================================
DATASET_DIR = r"g:\jammingdulu\dataset\spectrograms"
MODEL_SAVE_DIR = r"g:\jammingdulu\backend\models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "chord_cnn.keras")
PLOT_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "training_history.png")

IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 16 # Lebih kecil agar lebih sering update bobot (dataset kita kecil)
EPOCHS = 50
CHORD_CLASSES = ['A', 'Am', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G'] # Sesuai abjad folder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ==========================================
# KONFIGURASI TRAINING
# ==========================================
DATASET_DIR = r"g:\jammingdulu\dataset\spectrograms"
MODEL_SAVE_DIR = r"g:\jammingdulu\backend\models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "chord_cnn.keras")
PLOT_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "training_history.png")

IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 50
CHORD_CLASSES = ['A', 'Am', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G'] # Sesuai abjad folder

# Memastikan folder penyimpanan model tersedia
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==========================================
# 1. PERSIAPAN DATASET
# ==========================================
def prepare_dataset():
    print("Mempersiapkan dataset...")
    
    # 1. Generator dengan Augmentasi Ringan untuk Data Training (Spektrogram aman)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        width_shift_range=0.1,  # Geser waktu sedikit
        fill_mode='nearest'
    )

    # 2. Generator murni (tanpa augmentasi) untuk Data Validasi
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2 
    )

    # Data Training
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=42, # Gunakan seed yang sama untuk memastikan split konsisten
        shuffle=True
    )

    # Data Validasi
    validation_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=42, # Harus sama dengan training
        shuffle=False
    )
    
    return train_generator, validation_generator

# ==========================================
# 2. DEFINISI ARSITEKTUR CNN
# ==========================================
def build_model(num_classes):
    print("Membangun Custom CNN Khusus Spektrogram Audio...")
    
    model = Sequential([
        # Lapisan Input
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
        
        # Blok Klasifikasi Akhir
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    # Kompilasi Model (Menggunakan learning rate 0.001 karena melatih dari awal)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

# ==========================================
# 3. PLOT HASIL TRAINING
# ==========================================
def plot_training_history(history):
    print("Menyimpan plot grafik loss dan accuracy...")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(PLOT_SAVE_PATH)
    print(f"Grafik disimpan di: {PLOT_SAVE_PATH}")
    plt.close()

# ==========================================
# 4. PLOT CONFUSION MATRIX
# ==========================================
def plot_confusion_matrix(model, validation_generator):
    print("\nMenghitung Prediksi untuk Confusion Matrix...")
    # Reset generator untuk memastikan prediksi berurutan dari awal
    validation_generator.reset()
    
    # Ambil label asli
    y_true = validation_generator.classes
    
    # Lakukan prediksi
    predictions = model.predict(validation_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Dapatkan nama class
    class_names = list(validation_generator.class_indices.keys())
    
    # Buat matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix pada Data Validasi')
    plt.ylabel('Label Asli (Kenyataan)')
    plt.xlabel('Tebakan Model')
    
    cm_path = os.path.join(MODEL_SAVE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix disimpan di: {cm_path}")
    
    # Laporan Teks
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # 1. Siapkan data
    # Pastikan ada file di dalam direktori sebelum mengeksekusi ini
    try:
        train_gen, val_gen = prepare_dataset()
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        print("Pastikan Anda sudah mengisi folder dataset/spectrograms/ dengan gambar!")
        return

    num_classes = len(train_gen.class_indices)
    if num_classes != 9:
        print(f"Warning: Jumlah kelas yang ditemukan ({num_classes}) tidak sama dengan 9!")
        print("Kelas terdeteksi:", train_gen.class_indices)

    # 2. Bangun Model
    model = build_model(num_classes)

    # 3. Definisikan Callbacks
    # Checkpoint untuk menyimpan model terbaik berdasarkan validation loss
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Menurunkan learning rate jika model stuck (plateau)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )

    # 4. Latih Model
    print("\nMemulai training model...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 5. Visualisasikan Hasil
    plot_training_history(history)
    plot_confusion_matrix(model, val_gen)
    
    print(f"\nProses selesai. Model terbaik disimpan di: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    # Menghindari error GPU memory leak jika menggunakan GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    main()
