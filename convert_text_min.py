import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS (HU model)
tts = TTS("tts_models/hu/css10/vits").to(device)

# Run TTS
tts.tts_to_file(text="Teszt sztring", file_path="output_python.wav")
