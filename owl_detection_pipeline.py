#!/usr/bin/env python3
"""
Complete owl detection pipeline:
1. Create filtered_examples from examples (500-1500 Hz)
2. Create filtered_files from files (500-1500 Hz)
3. Analyze filtered_examples to find frequency bands
4. Find files with peaks in those bands (±50 Hz)
5. For each matched file, check interval patterns (>=5 intervals between 0.2-1.5s, close together)
6. Return original sound files that meet all conditions
"""

import librosa
import soundfile as sf
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import mode
from pathlib import Path
import json
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm


def apply_bandpass_filter(audio, sample_rate, low_freq=500, high_freq=1500):
    """Apply a Butterworth bandpass filter."""
    nyquist = sample_rate / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    low = max(0.01, min(low, 0.99))
    high = max(0.01, min(high, 0.99))
    if low >= high:
        raise ValueError(f"Invalid frequency range: {low_freq}-{high_freq} Hz")
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio


def filter_directory(input_dir, output_dir, low_freq=500, high_freq=1500):
    """Filter all audio files in a directory and subdirectories."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    audio_extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC']
    audio_files = []
    for ext in audio_extensions:
        # Use rglob to recursively search subdirectories
        audio_files.extend(input_path.rglob(f"*{ext}"))
    
    if not audio_files:
        print(f"  No audio files found in {input_dir}")
        return 0
    
    print(f"  Found {len(audio_files)} file(s)")
    
    for audio_file in tqdm(sorted(audio_files), desc="  Filtering files", unit="file"):
        try:
            audio, sr = librosa.load(audio_file, sr=None)
            filtered_audio = apply_bandpass_filter(audio, sr, low_freq, high_freq)
            max_val = np.max(np.abs(filtered_audio))
            if max_val > 0:
                filtered_audio = filtered_audio / max_val * 0.95
            # Preserve relative path structure in output
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_file, filtered_audio, sr)
        except Exception as e:
            tqdm.write(f"    Error processing {audio_file}: {e}")
    
    return len(audio_files)


def find_dominant_frequencies(audio, sample_rate, n_fft=2048, hop_length=512):
    """Find dominant frequency peaks in audio."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    avg_magnitude = np.mean(magnitude, axis=1)
    
    freq_mask = (frequencies >= 200) & (frequencies <= 3000)
    freq_range = frequencies[freq_mask]
    magnitude_range = avg_magnitude[freq_mask]
    
    if len(magnitude_range) == 0:
        return []
    
    magnitude_norm = (magnitude_range - np.min(magnitude_range)) / (
        np.max(magnitude_range) - np.min(magnitude_range) + 1e-10
    )
    
    min_height = np.percentile(magnitude_norm, 70)
    peaks, _ = find_peaks(
        magnitude_norm,
        height=min_height,
        prominence=0.10,
        distance=len(freq_range) // 50
    )
    
    peak_freqs = []
    for peak_idx in peaks:
        peak_freq = freq_range[peak_idx]
        peak_magnitude = magnitude_range[peak_idx]
        peak_freqs.append({'freq': peak_freq, 'magnitude': peak_magnitude})
    
    peak_freqs.sort(key=lambda x: x['magnitude'], reverse=True)
    return peak_freqs


def analyze_example_frequencies(filtered_examples_dir):
    """Analyze filtered examples to find frequency bands."""
    example_files = sorted(list(Path(filtered_examples_dir).glob("*.mp3")))
    
    if not example_files:
        return []
    
    all_peak_freqs = []
    
    for example_file in tqdm(example_files, desc="  Analyzing examples", unit="file"):
        audio, sr = librosa.load(example_file, sr=None)
        peaks = find_dominant_frequencies(audio, sr)
        if peaks:
            all_peak_freqs.extend([p['freq'] for p in peaks[:3]])
    
    if not all_peak_freqs:
        return []
    
    # Find unique frequency bands (±50 Hz)
    sorted_freqs = np.sort(all_peak_freqs)
    bands = []
    current_band = [sorted_freqs[0]]
    
    for freq in sorted_freqs[1:]:
        if freq - current_band[-1] <= 100:  # Within ±50 Hz of any freq in band
            current_band.append(freq)
        else:
            bands.append({
                'center': np.mean(current_band),
                'min': np.min(current_band),
                'max': np.max(current_band)
            })
            current_band = [freq]
    
    if current_band:
        bands.append({
            'center': np.mean(current_band),
            'min': np.min(current_band),
            'max': np.max(current_band)
        })
    
    # Merge overlapping bands
    merged_bands = []
    for band in bands:
        band_min = band['min'] - 50  # ±50 Hz tolerance
        band_max = band['max'] + 50  # ±50 Hz tolerance
        
        merged = False
        for mb in merged_bands:
            if not (band_max < mb['min'] or band_min > mb['max']):
                mb['min'] = min(mb['min'], band_min)
                mb['max'] = max(mb['max'], band_max)
                mb['center'] = (mb['min'] + mb['max']) / 2
                merged = True
                break
        
        if not merged:
            merged_bands.append({
                'center': band['center'],
                'min': band_min,
                'max': band_max
            })
    
    return merged_bands


def find_spikes_above_threshold(audio_file, peak_freq, threshold_db=-10, 
                                hop_length=512, n_fft=2048, bandwidth=30):
    """Find spikes above threshold in a filtered track around peak frequency."""
    audio, sr = librosa.load(audio_file, sr=None)
    
    # Create narrow bandpass filter around peak
    low_freq = max(20, peak_freq - bandwidth / 2)
    high_freq = min(sr / 2 - 100, peak_freq + bandwidth / 2)
    filtered_audio = apply_bandpass_filter(audio, sr, low_freq, high_freq)
    
    # Compute STFT
    stft = librosa.stft(filtered_audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    rms_per_frame = np.sqrt(np.mean(magnitude ** 2, axis=0))
    rms_db = librosa.amplitude_to_db(rms_per_frame, ref=np.max(rms_per_frame))
    
    # Find spikes above threshold
    above_threshold = rms_db >= threshold_db
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)
    
    spike_times = []
    in_spike = False
    spike_start = None
    
    for i, is_above in enumerate(above_threshold):
        if is_above and not in_spike:
            spike_start = times[i]
            in_spike = True
        elif not is_above and in_spike:
            spike_end = times[i]
            spike_center = (spike_start + spike_end) / 2
            spike_times.append(spike_center)
            in_spike = False
    
    if in_spike:
        spike_times.append((spike_start + times[-1]) / 2)
    
    # Calculate intervals
    intervals = []
    if len(spike_times) > 1:
        for i in range(1, len(spike_times)):
            interval = spike_times[i] - spike_times[i-1]
            intervals.append(interval)
    
    return intervals


def check_interval_pattern(intervals, min_intervals=5, min_interval=0.2, max_interval=1.5, max_cv=0.3):
    """
    Check if there are enough intervals in the specified range that are close together.
    
    Args:
        intervals: List of interval values
        min_intervals: Minimum number of intervals required (default 5)
        min_interval: Minimum interval value (default 0.2s)
        max_interval: Maximum interval value (default 1.5s)
        max_cv: Maximum coefficient of variation for intervals to be considered "close together" (default 0.3)
    
    Returns:
        Tuple: (True/False, list of intervals in range, CV value)
    """
    intervals_in_range = [i for i in intervals if min_interval <= i <= max_interval]
    
    if len(intervals_in_range) < min_intervals:
        return False, intervals_in_range, None
    
    # Check if intervals are close together (low CV)
    intervals_array = np.array(intervals_in_range)
    mean_interval = np.mean(intervals_array)
    
    if mean_interval == 0:
        return False, intervals_in_range, None
    
    std_interval = np.std(intervals_array)
    cv = std_interval / mean_interval
    
    # Intervals are "close together" if CV is below threshold
    is_close_together = cv <= max_cv
    
    return is_close_together, intervals_in_range, cv


def find_spike_times(audio_file, peak_freq, threshold_db=-10, 
                    hop_length=512, n_fft=2048, bandwidth=30):
    """Find spike times above threshold in a filtered track."""
    audio, sr = librosa.load(audio_file, sr=None)
    
    # Create narrow bandpass filter around peak
    low_freq = max(20, peak_freq - bandwidth / 2)
    high_freq = min(sr / 2 - 100, peak_freq + bandwidth / 2)
    filtered_audio = apply_bandpass_filter(audio, sr, low_freq, high_freq)
    
    # Compute STFT
    stft = librosa.stft(filtered_audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    rms_per_frame = np.sqrt(np.mean(magnitude ** 2, axis=0))
    rms_db = librosa.amplitude_to_db(rms_per_frame, ref=np.max(rms_per_frame))
    
    # Find spikes
    above_threshold = rms_db >= threshold_db
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)
    
    spike_times = []
    in_spike = False
    spike_start = None
    
    for i, is_above in enumerate(above_threshold):
        if is_above and not in_spike:
            spike_start = times[i]
            in_spike = True
        elif not is_above and in_spike:
            spike_end = times[i]
            spike_center = (spike_start + spike_end) / 2
            spike_duration = spike_end - spike_start
            spike_times.append({
                'start': spike_start,
                'end': spike_end,
                'center': spike_center,
                'duration': spike_duration
            })
            in_spike = False
    
    if in_spike:
        spike_end = times[-1]
        spike_center = (spike_start + spike_end) / 2
        spike_duration = spike_end - spike_start
        spike_times.append({
            'start': spike_start,
            'end': spike_end,
            'center': spike_center,
            'duration': spike_duration
        })
    
    return spike_times, times, rms_db


def detect_regular_pattern(spike_times, intervals, tolerance=0.3):
    """Detect if spikes occur at regular intervals."""
    if len(intervals) < 2:
        return {
            'is_regular': False,
            'mean_interval': None,
            'median_interval': None,
            'mode_interval': None,
            'std_interval': None,
            'cv': None,
            'regularity_score': 0.0,
            'num_spikes': len(spike_times)
        }
    
    intervals = np.array(intervals)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    median_interval = np.median(intervals)
    cv = std_interval / mean_interval if mean_interval > 0 else np.inf
    is_regular = cv < tolerance
    regularity_score = 1.0 / (1.0 + cv) if cv < np.inf else 0.0
    
    rounded_intervals = np.round(intervals / 0.1) * 0.1
    if len(rounded_intervals) > 0:
        mode_interval = mode(rounded_intervals, keepdims=True)[0][0]
    else:
        mode_interval = mean_interval
    
    return {
        'is_regular': bool(is_regular),
        'mean_interval': float(mean_interval),
        'median_interval': float(median_interval),
        'mode_interval': float(mode_interval),
        'std_interval': float(std_interval),
        'cv': float(cv) if cv < np.inf else None,
        'regularity_score': float(regularity_score),
        'num_spikes': len(spike_times),
        'intervals': [float(i) for i in intervals]
    }


def plot_pattern_detection(audio_file, peak_freq, spike_times, intervals, times, rms_db, 
                          pattern_info, threshold_db=-10, detection_intervals=None, output_file=None):
    """
    Plot spike pattern detection results.
    
    Args:
        detection_intervals: List of intervals that count as detections (in 0.2-1.5s range, close together)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    # Identify which spikes are detections
    detection_spike_indices = set()
    if detection_intervals and len(spike_times) > 1:
        spike_centers = [s['center'] for s in spike_times]
        for i in range(len(spike_centers) - 1):
            interval = spike_centers[i+1] - spike_centers[i]
            if interval in detection_intervals or any(abs(interval - di) < 0.01 for di in detection_intervals):
                detection_spike_indices.add(i)
                detection_spike_indices.add(i+1)
    
    # Plot 1: RMS energy with spikes marked
    ax1.plot(times, rms_db, 'b-', linewidth=1, alpha=0.7, label='RMS Energy (dB)')
    ax1.axhline(y=threshold_db, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold_db} dB')
    
    # Mark spike centers - highlight detections
    for i, spike in enumerate(spike_times):
        if i in detection_spike_indices:
            # Detection spikes - highlight in bright green/yellow
            ax1.axvline(x=spike['center'], color='lime', linestyle='-', linewidth=3, alpha=0.8, label='Detection' if i == min(detection_spike_indices) else '')
            ax1.axvspan(spike['start'], spike['end'], alpha=0.4, color='lime')
        else:
            # Other spikes - lighter green
            ax1.axvline(x=spike['center'], color='green', linestyle='-', linewidth=1, alpha=0.3)
            ax1.axvspan(spike['start'], spike['end'], alpha=0.1, color='green')
    
    ax1.set_ylabel('RMS Energy (dB)', fontsize=10)
    filename = Path(audio_file).name
    ax1.set_title(f'Spike Detection: {filename}\nPeak Frequency: {peak_freq:.1f} Hz', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Intervals between spikes - highlight detections
    if len(intervals) > 0:
        spike_centers = [s['center'] for s in spike_times]
        interval_times = [(spike_centers[i] + spike_centers[i+1]) / 2 
                          for i in range(len(spike_centers) - 1)]
        
        # Separate detection intervals from others
        detection_interval_times = []
        detection_interval_values = []
        other_interval_times = []
        other_interval_values = []
        
        for i, (it, iv) in enumerate(zip(interval_times, intervals)):
            if detection_intervals and any(abs(iv - di) < 0.01 for di in detection_intervals):
                detection_interval_times.append(it)
                detection_interval_values.append(iv)
            else:
                other_interval_times.append(it)
                other_interval_values.append(iv)
        
        # Plot other intervals
        if other_interval_times:
            ax2.plot(other_interval_times, other_interval_values, 'go-', linewidth=1, 
                    markersize=6, alpha=0.5, label='Other intervals')
        
        # Plot detection intervals - highlight
        if detection_interval_times:
            ax2.plot(detection_interval_times, detection_interval_values, 'mo-', 
                    linewidth=3, markersize=10, label='Detection intervals', zorder=5)
        
        # Add range lines
        ax2.axhspan(0.2, 1.5, alpha=0.1, color='blue', label='Detection range (0.2-1.5s)')
        
        if pattern_info['mean_interval']:
            ax2.axhline(y=pattern_info['mean_interval'], color='r', linestyle='--', 
                       linewidth=2, label=f"Mean: {pattern_info['mean_interval']:.2f}s")
        
        cv_str = f"{pattern_info['cv']:.3f}" if pattern_info['cv'] else "N/A"
        ax2.set_ylabel('Interval (seconds)', fontsize=10)
        ax2.set_title(f'Intervals Between Spikes (CV: {cv_str}, Regular: {pattern_info["is_regular"]})', 
                     fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No intervals (less than 2 spikes)', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_ylabel('Interval (seconds)', fontsize=10)
    
    # Plot 3: Histogram of intervals
    if len(intervals) > 0:
        ax3.hist(intervals, bins=min(20, len(intervals)), edgecolor='black', alpha=0.7)
        if pattern_info['mean_interval']:
            ax3.axvline(x=pattern_info['mean_interval'], color='r', linestyle='--', 
                       linewidth=2, label=f"Mean: {pattern_info['mean_interval']:.2f}s")
            if pattern_info['mode_interval']:
                ax3.axvline(x=pattern_info['mode_interval'], color='g', linestyle='--', 
                           linewidth=2, label=f"Mode: {pattern_info['mode_interval']:.2f}s")
        ax3.set_xlabel('Interval (seconds)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Distribution of Intervals', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No intervals to histogram', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        ax3.set_xlabel('Interval (seconds)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
    
    # Add pattern info text
    info_text = f"Regular Pattern: {'YES' if pattern_info['is_regular'] else 'NO'}\n"
    info_text += f"Spikes: {pattern_info['num_spikes']}\n"
    if pattern_info['mean_interval']:
        info_text += f"Mean Interval: {pattern_info['mean_interval']:.2f}s\n"
        info_text += f"Std Interval: {pattern_info['std_interval']:.2f}s\n"
        if pattern_info['cv']:
            info_text += f"CV: {pattern_info['cv']:.3f}\n"
        info_text += f"Regularity Score: {pattern_info['regularity_score']:.3f}"
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    Saved pattern plot: {output_file}")
    
    return fig


def plot_db_threshold_for_track(audio_file, peak_freq, threshold_db=-10, 
                                bandwidth=30, detection_spike_times=None, output_file=None, 
                                hop_length=512, n_fft=2048):
    """
    Plot dB threshold for a filtered track around peak frequency.
    Similar to plot_db_threshold but creates a narrow bandpass filter first.
    """
    # Load audio
    audio, sr = librosa.load(audio_file, sr=None)
    
    # Create narrow bandpass filter around peak
    low_freq = max(20, peak_freq - bandwidth / 2)
    high_freq = min(sr / 2 - 100, peak_freq + bandwidth / 2)
    filtered_audio = apply_bandpass_filter(audio, sr, low_freq, high_freq)
    
    # Compute STFT
    stft = librosa.stft(filtered_audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Get time array
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)
    
    # Calculate RMS energy per frame
    rms_per_frame = np.sqrt(np.mean(magnitude ** 2, axis=0))
    
    # Convert RMS to dB
    rms_db = librosa.amplitude_to_db(rms_per_frame, ref=np.max(rms_per_frame))
    
    # Create binary mask: 1 if above threshold, 0 if below
    above_threshold = (rms_db >= threshold_db).astype(float)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Identify detection spikes
    detection_mask = np.zeros(len(times), dtype=bool)
    if detection_spike_times:
        for det_spike in detection_spike_times:
            # Find time indices within spike range
            mask = (times >= det_spike['start']) & (times <= det_spike['end'])
            detection_mask = detection_mask | mask
    
    # Plot 1: Binary plot (above/below threshold) - highlight detections
    # Plot non-detection regions first
    above_threshold_bool = above_threshold.astype(bool)
    non_detection_above = (above_threshold_bool & ~detection_mask).astype(float)
    non_detection_below = (~above_threshold_bool & ~detection_mask).astype(float)
    ax1.fill_between(times, 0, non_detection_above, step='post', alpha=0.5, color='green', 
                     label=f'Above {threshold_db} dB')
    ax1.fill_between(times, non_detection_above, 1, step='post', alpha=0.5, color='red', 
                     label=f'Below {threshold_db} dB')
    
    # Highlight detection regions
    if detection_mask.any():
        detection_above = (above_threshold_bool & detection_mask).astype(float)
        ax1.fill_between(times, 0, detection_above, step='post', alpha=0.9, color='lime', 
                         label='Detection spikes')
        # Mark detection spike centers
        for det_spike in detection_spike_times:
            ax1.axvline(x=det_spike['center'], color='yellow', linestyle='-', 
                       linewidth=2, alpha=0.8)
    ax1.set_ylim([-0.1, 1.1])
    ax1.set_ylabel('Threshold Status', fontsize=10)
    filename = Path(audio_file).name
    ax1.set_title(f'Signal Above/Below {threshold_db} dB Threshold: {filename}\n'
                  f'Peak Frequency: {peak_freq:.1f} Hz (Bandwidth: {bandwidth} Hz)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    if len(times) > 1:
        ax1.set_xticks(np.arange(0, times[-1] + 10, 10))
    
    # Plot 2: Actual dB level over time
    ax2.plot(times, rms_db, 'b-', linewidth=1, alpha=0.7, label='RMS Energy (dB)')
    ax2.axhline(y=threshold_db, color='r', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold_db} dB')
    ax2.fill_between(times, threshold_db, rms_db, where=(rms_db >= threshold_db), 
                     alpha=0.3, color='green', label='Above threshold')
    ax2.fill_between(times, threshold_db, rms_db, where=(rms_db < threshold_db), 
                     alpha=0.3, color='red', label='Below threshold')
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.set_ylabel('RMS Energy (dB)', fontsize=10)
    ax2.set_title('RMS Energy Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Calculate statistics
    if len(times) > 1:
        time_above = np.sum(above_threshold) * (times[1] - times[0])
    else:
        time_above = 0
    total_time = times[-1] if len(times) > 0 else 0
    percent_above = (time_above / total_time * 100) if total_time > 0 else 0
    
    # Count intervals in 0-1s range
    intervals = []
    spike_times = []
    in_spike = False
    spike_start = None
    
    for i, is_above in enumerate(above_threshold):
        if is_above and not in_spike:
            spike_start = times[i]
            in_spike = True
        elif not is_above and in_spike:
            spike_end = times[i]
            spike_center = (spike_start + spike_end) / 2
            spike_times.append(spike_center)
            in_spike = False
    
    if in_spike:
        spike_times.append((spike_start + times[-1]) / 2)
    
    if len(spike_times) > 1:
        for i in range(1, len(spike_times)):
            interval = spike_times[i] - spike_times[i-1]
            intervals.append(interval)
    
    intervals_0_1s = [i for i in intervals if 0.0 <= i <= 1.0]
    
    # Add text box with statistics
    stats_text = f'Time above threshold: {time_above:.1f}s ({percent_above:.1f}%)\n'
    stats_text += f'Time below threshold: {total_time - time_above:.1f}s ({100 - percent_above:.1f}%)\n'
    stats_text += f'Intervals in 0-1s: {len(intervals_0_1s)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    Saved plot: {output_file}")
    
    return fig


def main():
    """Main pipeline function."""
    print("=" * 70)
    print("OWL DETECTION PIPELINE")
    print("=" * 70)
    
    # Step 1: Create filtered_examples
    print("\n[1/5] Creating filtered_examples from examples...")
    examples_dir = Path("examples")
    filtered_examples_dir = Path("filtered_examples")
    if examples_dir.exists():
        num_filtered = filter_directory(examples_dir, filtered_examples_dir, 
                                       low_freq=500, high_freq=1500)
        print(f"  Created {num_filtered} filtered example file(s)")
    else:
        print(f"  Examples directory not found, using existing filtered_examples")
    
    # Step 2: Create filtered_files
    print("\n[2/5] Creating filtered_files from files...")
    files_dir = Path("files")
    filtered_files_dir = Path("filtered_files")
    if files_dir.exists():
        num_filtered = filter_directory(files_dir, filtered_files_dir,
                                        low_freq=500, high_freq=1500)
        print(f"  Created {num_filtered} filtered file(s)")
    else:
        print(f"  Files directory not found, using existing filtered_files")
    
    # Step 3: Analyze example frequencies
    print("\n[3/5] Analyzing filtered_examples to find frequency bands...")
    frequency_bands = analyze_example_frequencies(filtered_examples_dir)
    if not frequency_bands:
        print("  ERROR: No frequency bands found in examples!")
        return
    
    print(f"  Found {len(frequency_bands)} frequency band(s):")
    for i, band in enumerate(frequency_bands, 1):
        print(f"    Band {i}: {band['min']:.1f} - {band['max']:.1f} Hz")
    
    # Step 4: Find files with peaks in frequency bands
    print("\n[4/5] Finding files with peaks in frequency bands...")
    # Recursively find all WAV files in filtered_files and subdirectories
    target_files = sorted(list(filtered_files_dir.rglob("*.WAV")))
    matching_files = []
    
    for target_file in tqdm(target_files, desc="  Checking files for peaks", unit="file"):
        audio, sr = librosa.load(target_file, sr=None)
        peaks = find_dominant_frequencies(audio, sr)
        
        for peak in peaks:
            peak_freq = peak['freq']
            for band in frequency_bands:
                if band['min'] <= peak_freq <= band['max']:
                    # Store relative path from filtered_files_dir
                    relative_path = target_file.relative_to(filtered_files_dir)
                    matching_files.append({
                        'file': str(relative_path),
                        'peak_freq': peak_freq,
                        'band': f"{band['min']:.1f}-{band['max']:.1f}"
                    })
                    break
            if any(b['min'] <= peak_freq <= b['max'] for b in frequency_bands):
                break
    
    print(f"  Found {len(matching_files)} file(s) with matching peaks")
    
    # Step 5: Check interval patterns and create plots for ALL matching files
    print("\n[5/5] Checking interval patterns (>=5 intervals between 0.2-1.5s, close together)...")
    final_matches = []
    all_analyzed = []
    plots_dir = Path("pipeline_plots")
    plots_dir.mkdir(exist_ok=True)
    
    for match in tqdm(matching_files, desc="  Processing matched files", unit="file"):
        filename = match['file']  # This is now a relative path
        peak_freq = match['peak_freq']
        target_file = filtered_files_dir / filename
        
        try:
            intervals = find_spikes_above_threshold(target_file, peak_freq, 
                                                   threshold_db=-10, bandwidth=30)
            
            # Check interval pattern: >=5 intervals between 0.2-1.5s that are close together
            meets_criteria, intervals_in_range, cv = check_interval_pattern(
                intervals, min_intervals=5, min_interval=0.2, max_interval=1.5, max_cv=0.3
            )
            num_intervals_in_range = len(intervals_in_range)
            
            # Get spike times for pattern detection
            spike_times, times, rms_db = find_spike_times(target_file, peak_freq, 
                                                         threshold_db=-10, bandwidth=30)
            spike_intervals = []
            if len(spike_times) > 1:
                for i in range(1, len(spike_times)):
                    interval = spike_times[i]['center'] - spike_times[i-1]['center']
                    spike_intervals.append(interval)
            
            pattern_info = detect_regular_pattern(spike_times, spike_intervals, tolerance=0.3)
            
            # Identify detection spikes (spikes whose intervals are in the detection set)
            detection_spike_times = []
            if meets_criteria and intervals_in_range and len(spike_times) > 1:
                spike_centers = [s['center'] for s in spike_times]
                detection_spike_indices = set()
                
                for i in range(len(spike_centers) - 1):
                    interval = spike_centers[i+1] - spike_centers[i]
                    # Check if this interval matches any detection interval (with small tolerance)
                    if any(abs(interval - di) < 0.01 for di in intervals_in_range):
                        detection_spike_indices.add(i)
                        detection_spike_indices.add(i+1)
                
                # Get detection spike times
                detection_spike_times = [spike_times[i] for i in detection_spike_indices]
            
            # Only create plots for files that meet criteria
            if meets_criteria:
                # Create pattern detection plot with detection markers
                # Preserve subdirectory structure in plots directory
                filename_path = Path(filename)
                pattern_filename = f"{filename_path.stem}_peak_{peak_freq:.1f}Hz_pattern.png"
                # Create subdirectory structure matching the original file structure
                pattern_path = plots_dir / filename_path.parent / pattern_filename
                pattern_path.parent.mkdir(parents=True, exist_ok=True)
                plot_pattern_detection(target_file, peak_freq, spike_times, spike_intervals, 
                                     times, rms_db, pattern_info, threshold_db=-10,
                                     detection_intervals=intervals_in_range,
                                     output_file=pattern_path)
                plt.close('all')
                
                # Ensure pattern_info is JSON serializable
                serializable_pattern_info = {
                    'is_regular': bool(pattern_info.get('is_regular', False)),
                    'mean_interval': float(pattern_info['mean_interval']) if pattern_info.get('mean_interval') is not None else None,
                    'median_interval': float(pattern_info['median_interval']) if pattern_info.get('median_interval') is not None else None,
                    'mode_interval': float(pattern_info['mode_interval']) if pattern_info.get('mode_interval') is not None else None,
                    'std_interval': float(pattern_info['std_interval']) if pattern_info.get('std_interval') is not None else None,
                    'cv': float(pattern_info['cv']) if pattern_info.get('cv') is not None else None,
                    'regularity_score': float(pattern_info.get('regularity_score', 0.0)),
                    'num_spikes': int(pattern_info.get('num_spikes', 0)),
                    'intervals': [float(i) for i in pattern_info.get('intervals', [])]
                }
                
                # Create file_info with all data
                # Store relative path from plots_dir for the plot file
                plot_relative_path = pattern_path.relative_to(plots_dir)
                file_info = {
                    'file': filename,
                    'peak_freq': float(peak_freq),
                    'band': match['band'],
                    'num_intervals_0_2_1_5s': int(num_intervals_in_range),
                    'total_intervals': int(len(intervals)),
                    'intervals_0_2_1_5s': [float(i) for i in intervals_in_range],
                    'cv': float(cv) if cv is not None else None,
                    'meets_criteria': bool(meets_criteria),
                    'pattern_info': serializable_pattern_info,
                    'pattern_plot_file': str(plot_relative_path)
                }
                
                cv_str = f"{cv:.3f}" if cv is not None else "N/A"
                tqdm.write(f"  ✓ {filename}: {num_intervals_in_range} intervals in 0.2-1.5s range, CV={cv_str} (MEETS CRITERIA)")
                tqdm.write(f"    Created plots with {len(detection_spike_times)} detection spike(s) marked")
                final_matches.append(file_info)
            else:
                reason = f"{num_intervals_in_range} intervals"
                if num_intervals_in_range < 5:
                    reason += " (<5 intervals)"
                elif cv is not None and cv > 0.3:
                    reason += f", CV={cv:.3f} (>0.3, not close together)"
                tqdm.write(f"  - {filename}: {reason} (does not meet criteria, skipping plots)")
                
        except Exception as e:
            tqdm.write(f"  ✗ {filename}: Error checking intervals - {e}")
            import traceback
            traceback.print_exc()
    
    # Final results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTotal files meeting all conditions: {len(final_matches)}")
    
    if final_matches:
        print("\nOriginal sound files with owl calls:")
        for match in sorted(final_matches, key=lambda x: x['num_intervals_0_2_1_5s'], reverse=True):
            print(f"  {match['file']}")
            print(f"    Peak frequency: {match['peak_freq']:.1f} Hz")
            print(f"    Intervals in 0.2-1.5s: {match['num_intervals_0_2_1_5s']}")
            if match.get('cv'):
                print(f"    CV: {match['cv']:.3f}")
            print(f"    Total intervals: {match['total_intervals']}")
        
        # Copy original audio files to results folder
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        print(f"\nCopying original audio files to {results_dir}...")
        
        files_dir = Path("files")
        copied_files = []
        
        for match in tqdm(final_matches, desc="  Copying files", unit="file"):
            filename = match['file']  # This is a relative path from filtered_files_dir
            # Find the corresponding file in the original files directory
            source_file = files_dir / filename
            
            if source_file.exists():
                # Preserve directory structure in results
                dest_file = results_dir / filename
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, dest_file)
                copied_files.append(filename)
            else:
                # Try to find the file by name in subdirectories if direct path doesn't work
                found = False
                for possible_file in files_dir.rglob(Path(filename).name):
                    if possible_file.name == Path(filename).name:
                        dest_file = results_dir / filename
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(possible_file, dest_file)
                        copied_files.append(filename)
                        found = True
                        break
                if not found:
                    tqdm.write(f"  Warning: {source_file} not found, skipping")
        
        # Save results
        results_file = Path("owl_detection_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'frequency_bands': [
                    {'min': float(b['min']), 'max': float(b['max']), 
                     'center': float(b['center'])} for b in frequency_bands
                ],
                'matching_files': final_matches,
                'all_analyzed_files': all_analyzed,
                'summary': {
                    'total_files_analyzed': len(target_files),
                    'files_with_matching_peaks': len(matching_files),
                    'files_meeting_all_conditions': len(final_matches),
                    'total_plots_created': len(final_matches),
                    'original_files_copied': len(copied_files)
                }
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Plots saved to: {plots_dir} (only matched files)")
        print(f"Original audio files saved to: {results_dir}")
        print(f"Total matched files: {len(final_matches)}")
        print(f"Total plots created: {len(final_matches)} (pattern plots only)")
        print(f"Total audio files copied: {len(copied_files)}")
    else:
        print("\nNo files met all conditions.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

