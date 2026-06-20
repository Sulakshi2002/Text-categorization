import torch
import librosa
import re
import os

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)

MODEL_PATH = r"D:\SLT\ASR\Wav2vec2 682 data\sinhala_asr_v1_2026-20260616T141519Z-3-001\sinhala_asr_v1_2026"

print("Loading model...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully")


def transcribe(audio_path):
    # Load the audio
    speech, sr = librosa.load(
        audio_path,
        sr=16000
    )

    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(
        logits,
        dim=-1
    )

    # Decode while skipping special tokens
    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    # Clean up [UNK] tags and fix duplicate spaces
    text = text.replace("[UNK]", "")
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --- Loop to process multiple audios ---
while True:
    audio_file = input("\nEnter WAV path (or type 'q' to quit): ").strip().strip('"')

    # Exit condition
    if audio_file.lower() == 'q':
        print("Exiting pipeline...")
        break

    # Check if the file actually exists before passing to model
    if not os.path.exists(audio_file):
        print(f"[Error]: File not found at '{audio_file}'. Please check the path.")
        continue

    try:
        print("Transcribing...")
        result = transcribe(audio_file)

        print("\n====================")
        print("TRANSCRIPTION")
        print("====================\n")
        print(result)
        
    except Exception as e:
        print(f"\n[ERROR processing this file]: {e}")