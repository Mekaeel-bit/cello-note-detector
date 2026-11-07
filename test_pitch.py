
import librosa
import numpy as np

# ----------------------------------
# Load cello recording
# ----------------------------------

audio_path = "data/recordings/test_cello.wav"

audio, sr = librosa.load(
    audio_path,
    sr=24000,
    mono=True
)

# ----------------------------------
# Detect fundamental frequency
# ----------------------------------

f0, voiced_flag, voiced_probs = librosa.pyin(
    audio,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("A5"),
    sr=sr,
    frame_length=4096,
    hop_length=512
)

# ----------------------------------
# Convert frequency into notes
# ----------------------------------

times = librosa.times_like(
    f0,
    sr=sr,
    hop_length=512
)

print("\n--- CELLO PITCH DETECTION ---")

for i in range(len(f0)):

    if voiced_flag[i] and np.isfinite(f0[i]):

        note = librosa.hz_to_note(f0[i])

        print(
            f"Time: {times[i]:.2f}s | "
            f"Frequency: {f0[i]:.2f} Hz | "
            f"Note: {note} | "
            f"Confidence: {voiced_probs[i]:.2f}"
        )
