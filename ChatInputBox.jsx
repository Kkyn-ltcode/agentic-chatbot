from faster_whisper import WhisperModel

# Define the path to the folder you just downloaded
model_path = "./my-local-whisper-model"

# Load the model from the local path
# Note: "model_size_or_path" accepts either a size ("tiny", "large-v3") OR a local directory path.
model = WhisperModel(
    model_size_or_path=model_path, 
    device="cuda", # or "cpu"
    compute_type="float16" # or "int8"
)

# Run a transcription just like normal
segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
