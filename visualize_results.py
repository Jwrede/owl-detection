#!/usr/bin/env python3
"""
Streamlit app to visualize owl detection results with audio playback.
"""

import streamlit as st
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import functions from pipeline
sys.path.insert(0, str(Path(__file__).parent))
from owl_detection_pipeline import find_spike_times

# Page config
st.set_page_config(
    page_title="Owl Detection Results Viewer",
    page_icon="🦉",
    layout="wide"
)

# Load results
@st.cache_data
def load_results():
    """Load detection results from JSON file."""
    results_file = Path("owl_detection_results.json")
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

# Load audio data
@st.cache_data
def load_audio(audio_path):
    """Load audio file and return audio data and sample rate."""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None

# Create detection visualization
def create_detection_plot(audio_path, detection_info, audio_data, sr):
    """Create an interactive plot showing detections over time."""
    peak_freq = detection_info['peak_freq']
    intervals_in_range = detection_info['intervals_0_2_1_5s']
    
    try:
        # Use the filtered file path instead of results path
        filtered_files_dir = Path("filtered_files")
        filtered_audio_path = filtered_files_dir / detection_info['file']
        
        if not filtered_audio_path.exists():
            st.warning(f"Filtered audio file not found: {filtered_audio_path}")
            return None
        
        spike_times, times, rms_db = find_spike_times(
            filtered_audio_path, peak_freq, threshold_db=-10, bandwidth=30
        )
        
        # Calculate spike centers
        spike_centers = [s['center'] for s in spike_times]
        
        # Identify detection spikes (those whose intervals are in the detection set)
        detection_spike_indices = set()
        if len(spike_centers) > 1:
            for i in range(len(spike_centers) - 1):
                interval = spike_centers[i+1] - spike_centers[i]
                if any(abs(interval - di) < 0.01 for di in intervals_in_range):
                    detection_spike_indices.add(i)
                    detection_spike_indices.add(i+1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: RMS energy with spikes
        ax1 = axes[0]
        ax1.plot(times, rms_db, 'b-', linewidth=0.5, alpha=0.7, label='RMS Energy')
        ax1.axhline(y=-10, color='r', linestyle='--', linewidth=1, label='Threshold (-10 dB)')
        
        # Mark all spikes
        for i, spike in enumerate(spike_times):
            if i in detection_spike_indices:
                ax1.axvline(x=spike['center'], color='lime', linewidth=2, alpha=0.7, label='Detection Spike' if i == min(detection_spike_indices) else '')
            else:
                ax1.axvline(x=spike['center'], color='green', linewidth=1, alpha=0.3, label='Other Spike' if i == 0 else '')
        
        ax1.set_xlabel('Time (seconds)', fontsize=11)
        ax1.set_ylabel('RMS Energy (dB)', fontsize=11)
        ax1.set_title(f'Spike Detection - Peak Frequency: {peak_freq:.1f} Hz', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_xlim(0, times[-1] if len(times) > 0 else 1)
        
        # Plot 2: Intervals between spikes
        ax2 = axes[1]
        spike_intervals = []
        interval_times = []
        detection_intervals = []
        detection_interval_times = []
        
        if len(spike_centers) > 1:
            for i in range(1, len(spike_centers)):
                interval = spike_centers[i] - spike_centers[i-1]
                interval_time = (spike_centers[i] + spike_centers[i-1]) / 2
                
                if interval in intervals_in_range or any(abs(interval - di) < 0.01 for di in intervals_in_range):
                    detection_intervals.append(interval)
                    detection_interval_times.append(interval_time)
                else:
                    spike_intervals.append(interval)
                    interval_times.append(interval_time)
        
        if spike_intervals:
            ax2.scatter(interval_times, spike_intervals, color='green', alpha=0.5, s=30, label='Other Intervals')
        if detection_intervals:
            ax2.scatter(detection_interval_times, detection_intervals, color='magenta', s=50, 
                       marker='o', edgecolors='darkmagenta', linewidths=1.5, label='Detection Intervals', zorder=5)
        
        ax2.axhspan(0.2, 1.5, alpha=0.2, color='blue', label='Detection Range (0.2-1.5s)')
        
        if intervals_in_range:
            mean_interval = np.mean(intervals_in_range)
            ax2.axhline(y=mean_interval, color='r', linestyle='--', linewidth=1.5, 
                       label=f'Mean Interval: {mean_interval:.3f}s')
        
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Interval (seconds)', fontsize=11)
        ax2.set_title('Intervals Between Spikes', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_xlim(0, times[-1] if len(times) > 0 else 1)
        ax2.set_ylim(0, max(2.0, max(spike_intervals + detection_intervals) * 1.1) if (spike_intervals or detection_intervals) else 2.0)
        
        # Plot 3: Interval histogram
        ax3 = axes[2]
        all_intervals = spike_intervals + detection_intervals
        if all_intervals:
            ax3.hist(all_intervals, bins=30, color='skyblue', alpha=0.7, edgecolor='black', label='All Intervals')
            if detection_intervals:
                ax3.hist(detection_intervals, bins=30, color='magenta', alpha=0.5, edgecolor='darkmagenta', label='Detection Intervals')
            
            if intervals_in_range:
                mean_interval = np.mean(intervals_in_range)
                ax3.axvline(x=mean_interval, color='r', linestyle='--', linewidth=1.5, 
                           label=f'Mean: {mean_interval:.3f}s')
                
                # Calculate mode if available
                from scipy.stats import mode as scipy_mode
                try:
                    mode_result = scipy_mode(np.round(intervals_in_range, 3))
                    if mode_result.count[0] > 0:
                        mode_interval = mode_result.mode[0]
                        ax3.axvline(x=mode_interval, color='g', linestyle='--', linewidth=1.5, 
                                   label=f'Mode: {mode_interval:.3f}s')
                except:
                    pass
        
        ax3.axvspan(0.2, 1.5, alpha=0.2, color='blue', label='Detection Range')
        ax3.set_xlabel('Interval (seconds)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Interval Distribution', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating plot: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def main():
    st.title("SSO Owl Detection Results Viewer")
    
    # Load results
    results = load_results()
    if results is None:
        st.error("❌ Results file not found. Please run the pipeline first.")
        st.info("Run: `python owl_detection_pipeline.py`")
        return
    
    matching_files = results.get('matching_files', [])
    if not matching_files:
        st.warning("No matching files found in results.")
        return
    
    # Create file selector
    file_options = [f"{f['file']} ({f['num_intervals_0_2_1_5s']} detections)" for f in matching_files]
    selected_index = st.selectbox(
        "Select a detected file:",
        range(len(file_options)),
        format_func=lambda x: file_options[x]
    )
    
    selected_file = matching_files[selected_index]
    
    # Find and load audio file
    results_dir = Path("results")
    audio_path = results_dir / selected_file['file']
    
    if not audio_path.exists():
        st.error(f"Audio file not found: {audio_path}")
        return
    
    # Load audio
    audio_data, sr = load_audio(audio_path)
    if audio_data is not None:
        # Display audio player
        st.audio(str(audio_path), format='audio/wav')
        
        # Audio info
        duration = len(audio_data) / sr
        st.caption(f"Duration: {duration:.2f} seconds | Sample Rate: {sr} Hz")
    
    if audio_data is not None:
        # Create plot (use filtered file for spike detection)
        filtered_files_dir = Path("filtered_files")
        filtered_audio_path = filtered_files_dir / selected_file['file']
        fig = create_detection_plot(filtered_audio_path, selected_file, audio_data, sr)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()

