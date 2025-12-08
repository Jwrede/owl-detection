#!/usr/bin/env python3
"""
Streamlit app to visualize owl detection results with audio playback.
"""

import streamlit as st
import json
import librosa
from pathlib import Path

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
    
    # Display saved plot
    st.markdown("---")
    st.header("📊 Detection Visualization")
    
    plot_path = Path("pipeline_plots") / selected_file.get('pattern_plot_file', '')
    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)
    else:
        st.warning(f"Plot not found at: {plot_path}")
        st.info("The plot may not have been generated during the pipeline run.")

if __name__ == "__main__":
    main()

