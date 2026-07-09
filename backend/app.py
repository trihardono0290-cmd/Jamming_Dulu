import io
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import librosa

# Import fungsi pemrosesan audio yang sudah dibuat di Part 1
from core.audio_processing import process_audio_realtime, SR

app = FastAPI(title="Guitar Chord Identification API")

# Setup CORS untuk mengizinkan frontend mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. INIT & LOAD MODEL
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "chord_cnn.keras")
CHORD_CLASSES = ['A', 'Am', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G']

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model CNN berhasil dimuat ke memori.")
except Exception as e:
    print(f"Warning: Model tidak ditemukan di {MODEL_PATH}.")
    print("Sistem akan berjalan tanpa model (kembalikan None) hingga Anda melatihnya di Part 2.")
    model = None

# ==========================================
# 2. FUNGSI PREDIKSI IN-MEMORY
# ==========================================
def predict_audio(audio_data, sr):
    if model is None:
        return "Not Loaded", 0.0
    
    try:
        # 1. Konversi array suara ke Mel-Spectrogram (In-Memory)
        # Hasilnya adalah array numpy (128, 128, 1)
        processed_input = process_audio_realtime(audio_data, sr)
        
        # Model membutuhkan batch dimension: (1, 128, 128, 1)
        batch_input = np.expand_dims(processed_input, axis=0)
        
        # 2. Prediksi probabilitas menggunakan CNN
        predictions = model.predict(batch_input, verbose=0)
        
        # 3. Ekstrak Hasil
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_chord = CHORD_CLASSES[predicted_idx]
        
        return predicted_chord, confidence
        
    except Exception as e:
        print(f"Prediksi error: {e}")
        return "Error", 0.0

# ==========================================
# 3. WEBSOCKET ENDPOINT (LOW-LATENCY REAL-TIME)
# ==========================================
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    Endpoint utama untuk Frontend (Part 4).
    Menerima stream rekaman mikrofon (.wav) berulang-ulang tanpa overhead HTTP HTTP request.
    """
    await websocket.accept()
    print("Client WebSocket terhubung.")
    try:
        while True:
            # Menerima chunk/blob audio binary dari browser
            audio_bytes = await websocket.receive_bytes()
            print(f"Menerima {len(audio_bytes)} bytes dari frontend...")
            
            # Decode in-memory menggunakan library librosa & bytesIO
            try:
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR)
                print(f"Berhasil decode audio. Durasi: {len(audio_data)/sr:.2f} detik")
                chord, conf = predict_audio(audio_data, sr)
                print(f"Prediksi: {chord} ({conf*100:.2f}%)")
                
                # Kirim respons JSON ke frontend untuk di-render (Benar/Salah)
                await websocket.send_json({
                    "predicted_chord": chord,
                    "confidence": round(conf, 4)
                })
            except Exception as e:
                print(f"Gagal memproses audio: {e}")
                await websocket.send_json({"error": "Failed to decode frame"})
                
    except WebSocketDisconnect:
        print("Client WebSocket terputus.")

# ==========================================
# 4. REST API ENDPOINT (FALLBACK / SHORT POLLING)
# ==========================================
@app.post("/predict")
async def predict_chord_endpoint(file: UploadFile = File(...)):
    """
    Endpoint tradisional untuk mengirim file .wav satu-per-satu.
    Berguna untuk integrasi testing via Postman / cURL.
    """
    audio_bytes = await file.read()
    
    try:
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR)
        chord, conf = predict_audio(audio_data, sr)
        return {
            "predicted_chord": chord, 
            "confidence": round(conf, 4)
        }
    except Exception as e:
        return {"error": str(e), "predicted_chord": None, "confidence": 0.0}

if __name__ == "__main__":
    import uvicorn
    # Menjalankan server asinkron
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
