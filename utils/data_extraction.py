import os
import subprocess
import whisper
import json
from pathlib import Path

# ----------------------------------
# CONFIG
# ----------------------------------
VIDEO_PATH = "data/video.mp4"
AUDIO_PATH = "data/audios/audio.wav"
TRANSCRIPT_PATH = "data/transcripts/full_transcript.json"
CHUNKS_DIR = "data/transcripts/chunks"
KEYFRAMES_DIR = "data/keyframes"
MAPPING_PATH = "data/mappings/frame_text_mapping.json"

FPS_INTERVAL = 2  # seconds between frames
CHUNK_DURATION = 10  # seconds per text chunk

# ----------------------------------
# Step 1: Extract Audio from Video
# ----------------------------------
def extract_audio(video_path, audio_path):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    command = [
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",  # resample to 16kHz for better STT
        "-ac", "1",      # mono audio
        "-vn",           # no video
        audio_path
    ]
    subprocess.run(command, check=True)
    print("[INFO] Audio extracted successfully.")

# ----------------------------------
# Step 2: Transcribe Audio
# ----------------------------------
def transcribe_audio(audio_path, transcript_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)

    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    
    print(f"[INFO] Transcript saved with {len(result['segments'])} segments.")

# ----------------------------------
# Step 3: Chunk Transcript
# ----------------------------------
def chunk_transcript(transcript_path, chunks_dir, chunk_duration=10):
    os.makedirs(chunks_dir, exist_ok=True)
    with open(transcript_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    segments = result["segments"]
    chunks = []
    current_chunk = {"start": None, "end": None, "text": ""}
    
    for seg in segments:
        if current_chunk["start"] is None:
            current_chunk["start"] = seg["start"]
        
        current_chunk["end"] = seg["end"]
        current_chunk["text"] += " " + seg["text"]

        if (current_chunk["end"] - current_chunk["start"]) >= chunk_duration:
            chunks.append(current_chunk)
            current_chunk = {"start": None, "end": None, "text": ""}
    
    if current_chunk["text"].strip():
        chunks.append(current_chunk)

    # Save each chunk
    for idx, chunk in enumerate(chunks):
        chunk_path = Path(chunks_dir) / f"chunk_{idx:04d}.json"
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=4)

    print(f"[INFO] {len(chunks)} text chunks saved.")

# ----------------------------------
# Step 4: Extract Keyframes
# ----------------------------------
def extract_keyframes(video_path, keyframes_dir, fps_interval=2):
    os.makedirs(keyframes_dir, exist_ok=True)
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{fps_interval}",
        f"{keyframes_dir}/frame_%04d.jpg"
    ]
    subprocess.run(command, check=True)
    print("[INFO] Keyframes extracted successfully.")

# ----------------------------------
# Step 5: Match Frames to Text
# ----------------------------------
def match_frames_to_text(keyframes_dir, chunks_dir, mapping_path, fps_interval=2):
    frame_files = sorted(Path(keyframes_dir).glob("frame_*.jpg"))
    chunk_files = sorted(Path(chunks_dir).glob("chunk_*.json"))

    # Load all chunks
    chunks = []
    for file in chunk_files:
        with open(file, "r", encoding="utf-8") as f:
            chunks.append(json.load(f))

    # Build mapping
    mapping = []
    for idx, frame_path in enumerate(frame_files):
        frame_time = idx * fps_interval  # estimate time based on extraction interval
        
        matched_text = ""
        for chunk in chunks:
            if chunk["start"] <= frame_time <= chunk["end"]:
                matched_text = chunk["text"]
                break
        
        mapping.append({
            "frame": str(frame_path.name),
            "timestamp": frame_time,
            "text": matched_text
        })

    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
    
    print(f"[INFO] Frame-text mapping saved with {len(mapping)} entries.")

# ----------------------------------
# MAIN
# ----------------------------------
def main():
    extract_audio(VIDEO_PATH, AUDIO_PATH)
    transcribe_audio(AUDIO_PATH, TRANSCRIPT_PATH)
    chunk_transcript(TRANSCRIPT_PATH, CHUNKS_DIR, chunk_duration=CHUNK_DURATION)
    extract_keyframes(VIDEO_PATH, KEYFRAMES_DIR, fps_interval=FPS_INTERVAL)
    match_frames_to_text(KEYFRAMES_DIR, CHUNKS_DIR, MAPPING_PATH, fps_interval=FPS_INTERVAL)

if __name__ == "__main__":
    main()
