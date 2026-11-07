import torch
import librosa
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor

model_id = "m-a-p/MERT-v1-95M"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

# 1. Load an audio file (using librosa's built-in sample track for testing)
audio_path = librosa.example('nutcracker')
audio, sr = librosa.load(audio_path, sr=24000, duration=10) # process first 10 seconds

inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 2. Get frame representations (Shape: [1, Num_Frames, 768])
# We detach to numpy for easy mathematical processing
frames = outputs.last_hidden_state.squeeze(0).numpy()

# 3. Create a Chroma Matrix to map 768 dimensions -> 12 Pitch Classes
chroma_filters = librosa.filters.chroma(sr=24000, n_fft=768, n_chroma=12)

# 4. Project MERT features onto the note space
# We pad/truncate arrays to align dimensions if necessary
if frames.shape[1] != chroma_filters.shape[1]:
    # Dynamic alignment to match hidden dimension map
    chroma_filters = librosa.filters.chroma(sr=24000, n_fft=(frames.shape[1]-1)*2, n_chroma=12)[:, :frames.shape[1]]

note_activations = np.dot(frames, chroma_filters.T)

# 5. Output information to PyCharm Console
pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
print("\n--- PITCH / NOTE PROFILE OUTPUT ---")
for frame_idx in range(min(5, len(note_activations))): # Print first 5 time blocks
    strongest_note_idx = np.argmax(note_activations[frame_idx])
    print(f"Time Window {frame_idx}: Most prominent musical pitch detected is {pitch_names[strongest_note_idx]}")
