#!/usr/bin/env python3
"""
Owl Sound Detector for Sokoke Scops Owl.
Uses example recordings and a known detection to find owl calls.
"""

import numpy as np
import librosa
from pathlib import Path
from scipy import signal


def get_mel_spec(audio: np.ndarray, sr: int) -> np.ndarray:
    """Get mel spectrogram focused on owl frequencies."""
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=64, 
        fmin=300, fmax=3000,
        hop_length=512, n_fft=2048
    )
    return librosa.power_to_db(S, ref=np.max)


def spec_similarity(spec1: np.ndarray, spec2: np.ndarray) -> float:
    """Compute normalized similarity between spectrograms."""
    min_w = min(spec1.shape[1], spec2.shape[1])
    s1 = spec1[:, :min_w].flatten()
    s2 = spec2[:, :min_w].flatten()
    
    s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-8)
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-8)
    
    return np.dot(s1, s2) / len(s1)


def find_owl(target_file: Path, template_spec: np.ndarray, template_audio_len: int, 
             sr: int = 22050, threshold: float = 0.5):
    """Find owl calls in a file using template matching."""
    audio, _ = librosa.load(target_file, sr=sr)
    
    hop_samples = sr // 4  # 0.25s hop for speed
    scores = []
    times = []
    
    for start in range(0, len(audio) - template_audio_len, hop_samples):
        window = audio[start:start + template_audio_len]
        window_spec = get_mel_spec(window, sr)
        sim = spec_similarity(window_spec, template_spec)
        scores.append(sim)
        times.append(start / sr)
    
    scores = np.array(scores)
    times = np.array(times)
    
    # Find peaks above threshold
    peaks, props = signal.find_peaks(scores, height=threshold, distance=8)
    
    detections = []
    if len(peaks) > 0:
        for peak_idx, height in zip(peaks, props['peak_heights']):
            detections.append((times[peak_idx], height))
    
    return detections, scores, times


def main():
    examples_dir = Path("/home/jonathan/owl/examples")
    files_dir = Path("/home/jonathan/owl/files")
    sr = 22050
    
    print("🦉 Sokoke Scops Owl Detector")
    print("=" * 50)
    
    # Create template from known owl location
    known_file = files_dir / "20250823_062400.WAV"
    known_time = 34.0
    
    print(f"\nBuilding template from known call at {known_file.name} @ 0:34...")
    audio, _ = librosa.load(known_file, sr=sr)
    
    # Extract 2-second window around known call
    start = int((known_time - 1) * sr)
    end = int((known_time + 1) * sr)
    template_audio = audio[start:end]
    template_spec = get_mel_spec(template_audio, sr)
    
    # Validate against example recordings
    print("\nValidating against example recordings:")
    example_files = list(examples_dir.glob("*.mp3"))
    for f in example_files[:3]:
        ex_audio, _ = librosa.load(f, sr=sr, duration=2.0)
        ex_audio, _ = librosa.effects.trim(ex_audio, top_db=20)
        if len(ex_audio) > sr * 0.5:
            ex_spec = get_mel_spec(ex_audio, sr)
            sim = spec_similarity(template_spec, ex_spec)
            print(f"  {f.name[:20]}... : {sim:.3f}")
    
    # Search all files
    print("\n" + "=" * 50)
    print("Searching recordings (threshold: 0.5)...")
    print("=" * 50)
    
    target_files = sorted(files_dir.glob("*.WAV"))
    all_detections = []
    
    for file_path in target_files:
        print(f"\n📁 {file_path.name}")
        
        detections, scores, times = find_owl(
            file_path, template_spec, len(template_audio), sr, threshold=0.5
        )
        
        if detections:
            for time_sec, score in detections:
                mins, secs = divmod(int(time_sec), 60)
                print(f"   ✅ OWL DETECTED at {mins:02d}:{secs:02d} (confidence: {score:.2f})")
                all_detections.append((file_path.name, time_sec, score))
        else:
            # Show best match for reference
            best_idx = np.argmax(scores)
            mins, secs = divmod(int(times[best_idx]), 60)
            print(f"   No owl (best match: {mins:02d}:{secs:02d} = {scores[best_idx]:.2f})")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if all_detections:
        print(f"\n🎯 Found {len(all_detections)} owl call(s):\n")
        for name, t, score in sorted(all_detections, key=lambda x: (x[0], x[1])):
            mins, secs = divmod(int(t), 60)
            print(f"   {name} at {mins:02d}:{secs:02d} (confidence: {score:.2f})")
    else:
        print("\nNo owl calls detected above threshold.")


if __name__ == "__main__":
    main()
