/**
 * Frontend Logic for GitarLive
 * Menerapkan desain Figma yang baru & Raw PCM WebSocket streaming.
 */

// UI Elements
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const chordSelectorGroup = document.getElementById('chordSelectorGroup');
const targetChordSelect = document.getElementById('targetChord'); // Hidden select
const canvas = document.getElementById('visualizerCanvas');
const ctx = canvas.getContext('2d');
const connectionStatus = document.getElementById('connectionStatus');

const resultPanel = document.getElementById('resultPanel');
const idleState = document.getElementById('idleState');
const activeState = document.getElementById('activeState');
const reviewState = document.getElementById('reviewState');

const recordingActionButtons = document.getElementById('recordingActionButtons');
const reviewActionButtons = document.getElementById('reviewActionButtons');
const btnRetake = document.getElementById('btnRetake');
const btnSubmit = document.getElementById('btnSubmit');
const audioPlayback = document.getElementById('audioPlayback');

const soundCorrect = document.getElementById('soundCorrect');
const soundWrong = document.getElementById('soundWrong');

// Result Panel Elements
const resultHeader = document.getElementById('resultHeader');
const resultIcon = document.getElementById('resultIcon');
const resultTitle = document.getElementById('resultTitle');
const confidenceValue = document.getElementById('confidenceValue');
const chordDiagramContainer = document.getElementById('chordDiagramContainer');
const lblTarget = document.getElementById('lblTarget');
const lblPrediksi = document.getElementById('lblPrediksi');
const fingeringGuide = document.getElementById('fingeringGuide');
const wrongWarningBox = document.getElementById('wrongWarningBox');
const wrongWarningText = document.getElementById('wrongWarningText');

// State Variables
let audioContext;
let scriptProcessor;
let analyser;
let ws;
let animationId;
let isRecording = false;
let audioBuffer = [];
let audioStream;

// Database Diagram Chord & Panduan Jari
const CHORD_DB = {
    'C': { 
        frets: [-1, 3, 2, 0, 1, 0], 
        fingers: [-1, 3, 2, -1, 1, -1],
        guide: [
            "Jari Manis (3) di Senar 5, Fret 3",
            "Jari Tengah (2) di Senar 4, Fret 2",
            "Jari Telunjuk (1) di Senar 2, Fret 1",
            "Senar 1, 3, dan 6 dibiarkan terbuka (atau Senar 6 dibisukan/X)"
        ]
    },
    'D': { 
        frets: [-1, -1, 0, 2, 3, 2],
        fingers: [-1, -1, -1, 1, 3, 2],
        guide: [
            "Jari Telunjuk (1) di Senar 3, Fret 2",
            "Jari Manis (3) di Senar 2, Fret 3",
            "Jari Tengah (2) di Senar 1, Fret 2",
            "Senar 4 dibunyikan open, Senar 5 & 6 di-mute (X)"
        ]
    },
    'E': { 
        frets: [0, 2, 2, 1, 0, 0],
        fingers: [-1, 2, 3, 1, -1, -1],
        guide: [
            "Jari Tengah (2) di Senar 5, Fret 2",
            "Jari Manis (3) di Senar 4, Fret 2",
            "Jari Telunjuk (1) di Senar 3, Fret 1",
            "Senar 1, 2, dan 6 dimainkan lepas (O)"
        ]
    },
    'F': { 
        frets: [-1, 3, 3, 2, 1, 1],
        fingers: [-1, 3, 4, 2, 1, 1],
        guide: [
            "Jari Telunjuk (1) menekan rata Senar 1 & 2 di Fret 1",
            "Jari Tengah (2) di Senar 3, Fret 2",
            "Jari Kelingking (4) di Senar 4, Fret 3",
            "Jari Manis (3) di Senar 5, Fret 3"
        ]
    },
    'G': { 
        frets: [3, 2, 0, 0, 0, 3],
        fingers: [2, 1, -1, -1, -1, 3],
        guide: [
            "Jari Tengah (2) di Senar 6, Fret 3",
            "Jari Telunjuk (1) di Senar 5, Fret 2",
            "Jari Manis (3) di Senar 1, Fret 3"
        ]
    },
    'A': { 
        frets: [-1, 0, 2, 2, 2, 0],
        fingers: [-1, -1, 1, 2, 3, -1],
        guide: [
            "Jari Telunjuk (1) di Senar 4, Fret 2",
            "Jari Tengah (2) di Senar 3, Fret 2",
            "Jari Manis (3) di Senar 2, Fret 2",
            "Senar 1 dan 5 dimainkan lepas (O), Senar 6 di-mute"
        ]
    },
    'Am': { 
        frets: [-1, 0, 2, 2, 1, 0],
        fingers: [-1, -1, 2, 3, 1, -1],
        guide: [
            "Jari Tengah (2) di Senar 4, Fret 2",
            "Jari Manis (3) di Senar 3, Fret 2",
            "Jari Telunjuk (1) di Senar 2, Fret 1",
            "Senar 1 dan 5 dimainkan lepas (O)"
        ]
    },
    'Em': { 
        frets: [0, 2, 2, 0, 0, 0],
        fingers: [-1, 2, 3, -1, -1, -1],
        guide: [
            "Jari Tengah (2) di Senar 5, Fret 2",
            "Jari Manis (3) di Senar 4, Fret 2",
            "Semua senar lainnya dimainkan lepas (O)"
        ]
    },
    'Dm': { 
        frets: [-1, -1, 0, 2, 3, 1],
        fingers: [-1, -1, -1, 2, 3, 1],
        guide: [
            "Jari Tengah (2) di Senar 3, Fret 2",
            "Jari Manis (3) di Senar 2, Fret 3",
            "Jari Telunjuk (1) di Senar 1, Fret 1",
            "Senar 4 open (O), Senar 5 & 6 di-mute"
        ]
    }
};
const CHORD_LIST = ['C', 'D', 'E', 'F', 'G', 'A', 'Am', 'Em', 'Dm'];

// --- INIT CHORD SELECTOR ---
function initChordSelector() {
    chordSelectorGroup.innerHTML = '';
    CHORD_LIST.forEach(chord => {
        const btn = document.createElement('button');
        btn.className = `btn-chord px-6 py-3 rounded-xl font-bold text-lg min-w-[70px] ${chord === 'C' ? 'active' : ''}`;
        btn.textContent = chord;
        btn.onclick = () => {
            // Hapus active dari semua
            document.querySelectorAll('.btn-chord').forEach(b => b.classList.remove('active'));
            // Set active ke yang diklik
            btn.classList.add('active');
            targetChordSelect.value = chord;
            
            if (isRecording) {
                // Sembunyikan hasil lama saat target diganti di tengah rekaman
                resultPanel.classList.remove('show');
            }
        };
        chordSelectorGroup.appendChild(btn);
    });
}
initChordSelector();

// --- WEBSOCKET CONNECTION ---
function connectWebSocket() {
    ws = new WebSocket('ws://localhost:8000/ws/predict');
    
    ws.onopen = () => {
        connectionStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-[#00d285] mr-2"></span> Connected`;
        connectionStatus.className = 'flex items-center text-xs text-[#00d285] font-bold tracking-wider uppercase';
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.predicted_chord && data.predicted_chord !== "Unknown" && data.predicted_chord !== "Not Loaded") {
                handlePredictionResult(data.predicted_chord, data.confidence);
            }
        } catch (e) {
            console.error("Gagal parsing respons", e);
        }
    };

    ws.onclose = () => {
        connectionStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500 mr-2"></span> Disconnected`;
        connectionStatus.className = 'flex items-center text-xs text-rose-500 font-bold tracking-wider uppercase';
        if (isRecording) stopRecording();
    };
    
    ws.onerror = (err) => console.error("WebSocket Error:", err);
}

// --- UI RENDER: RESULT & SVG ---
function handlePredictionResult(predicted, confidence) {
    const target = targetChordSelect.value;
    
    // Tampilkan Panel Result
    resultPanel.classList.add('show');
    
    lblTarget.textContent = target;
    lblPrediksi.textContent = predicted;
    confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
    
    if (predicted === target) {
        // STYLE BENAR
        resultHeader.className = "p-6 flex justify-between items-center result-header-correct";
        resultIcon.className = "w-10 h-10 rounded-full bg-[#00d285] bg-opacity-20 border border-[#00d285] flex items-center justify-center text-[#00d285]";
        resultIcon.innerHTML = `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>`;
        resultTitle.textContent = "Chord Benar (Sesuai Target)";
        resultTitle.className = "text-xl font-extrabold text-white";
        confidenceValue.className = "text-3xl font-extrabold text-[#00d285]";
        wrongWarningBox.classList.add('hidden');
        
        renderChordDiagram(target, true);
        renderGuide(target);
        
        // Putar audio dan gulir layar
        soundCorrect.play().catch(e => console.log("Audio correct.mp3 tidak ditemukan."));

    } else {
        // STYLE SALAH
        resultHeader.className = "p-6 flex justify-between items-center result-header-wrong";
        resultIcon.className = "w-10 h-10 rounded-full bg-rose-500 bg-opacity-20 border border-rose-500 flex items-center justify-center text-rose-500";
        resultIcon.innerHTML = `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>`;
        resultTitle.textContent = "Chord Salah (Koreksi Diperlukan)";
        resultTitle.className = "text-xl font-extrabold text-rose-400";
        confidenceValue.className = "text-3xl font-extrabold text-[#00d285]"; // Tetap hijau sesuai figma
        wrongWarningBox.classList.remove('hidden');
        wrongWarningText.innerHTML = `⚠️ Hasil petikan Anda salah, model mengenali bunyi ini sebagai chord <b>${predicted}</b>. Silakan perhatikan penempatan jari pada diagram dan rapihkan fretboard sebelum mencoba kembali.`;
        
        renderChordDiagram(target, false); // Tampilkan target sbg koreksi
        renderGuide(target);
        
        // Putar audio salah dan gulir layar
        soundWrong.play().catch(e => console.log("Audio wrong.mp3 tidak ditemukan."));
    }
    
    // Auto-scroll layar menuju ke hasil panel agar user langsung melihatnya
    setTimeout(() => {
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function renderGuide(chordName) {
    const chord = CHORD_DB[chordName];
    if(!chord) return;
    fingeringGuide.innerHTML = chord.guide.map(text => `<li>${text}</li>`).join('');
}

function renderChordDiagram(chordName, isCorrect) {
    // Memuat gambar dari folder assets/diagrams/ berdasarkan nama chord
    const imgHtml = `
        <div class="w-full h-full flex items-center justify-center text-center">
            <img src="assets/diagrams/${chordName}.png" 
                 alt="Diagram Chord ${chordName}" 
                 class="w-full h-full object-contain"
                 onerror="this.onerror=null; this.parentElement.innerHTML='<p class=\\'text-rose-400 text-xs mt-4\\'>Gambar <b>${chordName}.png</b> belum ada di folder assets/diagrams/</p>';">
        </div>
    `;
    chordDiagramContainer.innerHTML = imgHtml;
}

// --- WAV ENCODER ---
function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return new Blob([view], { type: 'audio/wav' });
}

// --- RECORDING & WEBAUDIO ---
async function startRecording() {
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } 
        });
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        
        const source = audioContext.createMediaStreamSource(audioStream);
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 0;
        
        source.connect(analyser);
        analyser.connect(scriptProcessor);
        scriptProcessor.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        audioBuffer = [];
        const currentSampleRate = audioContext.sampleRate;
        
        scriptProcessor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const channelData = e.inputBuffer.getChannelData(0);
            audioBuffer.push(new Float32Array(channelData));
            // Hanya mengumpulkan buffer tanpa mengirim apa-apa
        };
        
        isRecording = true;
        resultPanel.classList.remove('show'); // Sembunyikan hasil lama
        
        // UI Updates
        idleState.classList.add('hidden');
        activeState.classList.remove('hidden');
        activeState.classList.add('flex');
        
        btnStart.classList.add('hidden');
        btnStop.classList.remove('hidden');
        
        // Lock selectors
        document.querySelectorAll('.btn-chord').forEach(b => {
            b.disabled = true;
            b.classList.add('opacity-50', 'cursor-not-allowed');
        });
        
        // Start Canvas
        resizeCanvas();
        visualize();
        
    } catch (err) {
        alert("Akses mikrofon ditolak.");
        console.error(err);
    }
}

let finalWavBlob = null;

function stopRecording() {
    isRecording = false;
    
    // Gabungkan seluruh data dari audioBuffer
    let currentLength = audioBuffer.length * 4096;
    if (currentLength > 0) {
        let mergedBuffer = new Float32Array(currentLength);
        let offset = 0;
        for (let i = 0; i < audioBuffer.length; i++) {
            mergedBuffer.set(audioBuffer[i], offset);
            offset += audioBuffer[i].length;
        }
        
        // Buat file WAV akhir
        finalWavBlob = encodeWAV(mergedBuffer, audioContext.sampleRate);
        const audioUrl = URL.createObjectURL(finalWavBlob);
        audioPlayback.src = audioUrl;
    }
    
    if (scriptProcessor) scriptProcessor.disconnect();
    if (audioStream) audioStream.getTracks().forEach(track => track.stop());
    if (audioContext) audioContext.close();
    
    cancelAnimationFrame(animationId);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // UI Updates: Switch to Review Mode
    activeState.classList.add('hidden');
    activeState.classList.remove('flex');
    reviewState.classList.remove('hidden');
    reviewState.classList.add('flex');
    
    recordingActionButtons.classList.add('hidden');
    reviewActionButtons.classList.remove('hidden');
}

// Tombol Rekam Ulang
btnRetake.addEventListener('click', () => {
    // Reset UI back to idle
    reviewState.classList.add('hidden');
    reviewState.classList.remove('flex');
    idleState.classList.remove('hidden');
    
    reviewActionButtons.classList.add('hidden');
    recordingActionButtons.classList.remove('hidden');
    btnStart.classList.remove('hidden');
    btnStop.classList.add('hidden');
    
    // Unlock selectors
    document.querySelectorAll('.btn-chord').forEach(b => {
        b.disabled = false;
        b.classList.remove('opacity-50', 'cursor-not-allowed');
    });
});

// Tombol Koreksi Rekaman
btnSubmit.addEventListener('click', () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        setTimeout(() => {
            if (ws && ws.readyState === WebSocket.OPEN && finalWavBlob) {
                ws.send(finalWavBlob);
            }
        }, 500);
    } else {
        if (finalWavBlob) {
            ws.send(finalWavBlob);
        }
    }
    
    // Ubah text tombol sbg indikator loading
    btnSubmit.innerHTML = `<span class="animate-pulse">Mengoreksi...</span>`;
    setTimeout(() => {
        btnSubmit.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"></path></svg> Koreksi Rekaman`;
    }, 1500);
});

function resizeCanvas() {
    const parent = canvas.parentElement;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;
}

function visualize() {
    if (!analyser) return;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
        if (!isRecording) return;
        animationId = requestAnimationFrame(draw);
        
        analyser.getByteTimeDomainData(dataArray);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#00d285';
        ctx.beginPath();
        
        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }
        
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    };
    draw();
}

// Event Listeners
window.addEventListener('resize', () => {
    if (isRecording) resizeCanvas();
});

btnStart.addEventListener('click', () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        setTimeout(startRecording, 500);
    } else {
        startRecording();
    }
});

btnStop.addEventListener('click', stopRecording);

connectWebSocket();
