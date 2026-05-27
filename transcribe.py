
import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# =========================
# 1. CONFIG
# =========================

MODEL_PATH = r"D:\SLT\Wav2vec\wav2vec2-sinhala-final-model (1)\content\wav2vec2-sinhala-final-model"

AUDIO_PATH = r"C:\Users\Royal PC\Downloads\Email 2\GB_Si_20260312-112524_112860026-all.wav"

SR = 16000
CHUNK_SEC = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# =========================
# 2. LOAD MODEL
# =========================

print("Loading model...")

processor = Wav2Vec2Processor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(device)
model.eval()

print("Model loaded successfully")

# =========================
# 3. AUDIO CHUNKING
# =========================

def chunk_audio(audio, sr=SR, chunk_sec=CHUNK_SEC):
    chunk_size = sr * chunk_sec

    return [
        audio[i:i + chunk_size]
        for i in range(0, len(audio), chunk_size)
        if len(audio[i:i + chunk_size]) > sr * 1  # ignore too small chunks
    ]

# =========================
# 4. CLEAN TRANSCRIBE FUNCTION
# =========================

def transcribe(audio_path):

    audio, sr = librosa.load(audio_path, sr=SR)

    # normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-7)

    chunks = chunk_audio(audio)

    results = []

    for chunk in chunks:

        inputs = processor(
            chunk,
            sampling_rate=SR,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        # smoothing (important)
        logits = logits / 1.3

        pred_ids = torch.argmax(logits, dim=-1)

        text = processor.batch_decode(pred_ids)[0]

        # cleanup
        #text = text.replace("[UNK]", "")
        #text = " ".join(text.split())

        results.append(text)

    return " ".join(results)

# =========================
# 5. RUN TEST
# =========================

if __name__ == "__main__":

    print("\nTranscribing...\n")

    result = transcribe(AUDIO_PATH)

    print("\n===== FINAL TRANSCRIPTION =====\n")
    print(result)