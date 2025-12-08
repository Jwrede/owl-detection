# Owl Call Detection Pipeline

Automated detection system for identifying owl calls in audio recordings using frequency analysis and pattern recognition.

## Overview

This pipeline analyzes audio files to detect owl calls by:
1. Filtering audio to focus on owl call frequency ranges (500-1500 Hz)
2. Identifying dominant frequency peaks matching example owl calls
3. Detecting regular spike patterns that indicate owl calls
4. Validating detections based on interval consistency

## Quick Start: Viewing Results

**If you're new to this project or just want to review detection results**, you can use the interactive visualization app to listen to detected audio files and see the detection patterns:

```bash
python -m venv venv
source venv activate
pip install -r requirements.txt
streamlit run visualize_results.py
```

This will open a web interface where you can:
- **Select detected files** from a dropdown menu
- **Play audio** directly in your browser
- **View interactive visualizations** showing spike detections, intervals, and patterns

The visualization app loads results from `owl_detection_results.json` (created by running the pipeline). If it's not in the project, see the [Usage](#usage) section below to run the detection pipeline first.

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
owl/
├── examples/              # Example owl call recordings (MP3)
├── files/                # Target audio files to analyze (WAV) - **YOU MUST FILL THIS FOLDER**
├── filtered_examples/    # Filtered example files (500-1500 Hz)
├── filtered_files/       # Filtered target files (500-1500 Hz)
├── pipeline_plots/       # Detection visualization plots
├── results/              # Original audio files with detected owl calls
├── owl_detection_pipeline.py  # Main pipeline script
└── visualize_results.py  # Interactive Streamlit app for viewing results
```

**Important**: Before running the pipeline, you must populate the `files/` folder with audio files (WAV format) that you want to analyze for owl calls.

## Usage

**Prerequisites**: 
- Place your example owl call recordings in the `examples/` folder (MP3 format)
- Place the audio files you want to analyze in the `files/` folder (WAV format)

Run the complete pipeline:

```bash
python owl_detection_pipeline.py
```

The pipeline will:
1. **Filter examples**: Create `filtered_examples/` from `examples/` (500-1500 Hz bandpass)
2. **Filter files**: Create `filtered_files/` from `files/` (500-1500 Hz bandpass)
3. **Analyze frequency bands**: Extract dominant frequencies from filtered examples
4. **Find matching peaks**: Identify files with peaks in example frequency bands (±50 Hz tolerance)
5. **Detect patterns**: Check for regular spike patterns (>=5 intervals between 0.2-1.5s, CV <= 0.3)
6. **Save results**: Copy matching audio files to `results/` and create visualization plots

## Detection Criteria

A file is considered to contain owl calls if it meets **all** of the following criteria:

1. **Frequency Match**: Has a dominant peak frequency within ±50 Hz of example frequencies
2. **Regular Pattern**: Has >=5 intervals between spikes in the 0.2-1.5 second range
3. **Consistent Intervals**: The intervals are "close together" (coefficient of variation CV <= 0.3)

### Example Frequency Bands

The pipeline analyzes example owl calls to determine frequency bands. Typical bands found:
- **Band 1**: ~817-1105 Hz (merged from multiple example frequencies)

### Interval Pattern Detection

The system detects spikes above -10 dB threshold and analyzes intervals between spikes:
- **Detection range**: 0.2-1.5 seconds
- **Minimum intervals**: >=5 intervals in this range
- **Consistency requirement**: CV (coefficient of variation) <= 0.3

The CV measures how consistent the intervals are:
- **Low CV (< 0.3)**: Intervals are regular and consistent → likely owl calls
- **High CV (> 0.3)**: Intervals are irregular → likely noise or other sounds

## Output Files

### Results JSON (`owl_detection_results.json`)

Contains detection results with:
- Frequency bands used for matching
- Files that meet all criteria
- Detection statistics (intervals, CV, peak frequencies)

### Visualization Plots (`pipeline_plots/`)

For each detected file, a pattern plot is created showing:

1. **Spike Detection Plot**: RMS energy over time with detected spikes marked
   - **Lime/yellow markers**: Detection spikes (count toward >=5 requirement)
   - **Green markers**: Other spikes

2. **Interval Analysis Plot**: Intervals between spikes over time
   - **Magenta markers**: Detection intervals (0.2-1.5s range, close together)
   - **Green markers**: Other intervals
   - **Blue shaded area**: Detection range (0.2-1.5s)

3. **Interval Distribution Histogram**: Distribution of all intervals
   - Shows how intervals cluster around detection range

### Detected Audio Files (`results/`)

Original WAV files that meet all detection criteria are copied here for easy access.

## Interactive Results Viewer

After running the pipeline, you can use the interactive visualization app to review detections:

```bash
streamlit run visualize_results.py
```

The app provides:
- **File Selection**: Browse and select from all detected files
- **Audio Playback**: Listen to detected audio files directly in your browser
- **Detection Visualization**: Interactive plots showing:
  - Spike detection over time with detection spikes highlighted
  - Interval analysis showing regular patterns
  - Interval distribution histogram

This is especially useful for:
- **Reviewing detections**: Verify that detected files actually contain owl calls
- **Understanding patterns**: See how the algorithm identifies regular spike patterns
- **Quality control**: Quickly listen to and visualize multiple detections

The visualization app automatically loads results from `owl_detection_results.json` and displays all files that met the detection criteria.

## Example Results

Based on the current analysis:

**Files with detected owl calls:**
- `20250821_000000.WAV`: 5 intervals in 0.2-1.5s range, CV=0.217
- `20250821_061800.WAV`: 8 intervals in 0.2-1.5s range, CV=0.057
- `20250821_194200.WAV`: 11 intervals in 0.2-1.5s range, CV=0.211
- `20250822_053600.WAV`: 18 intervals in 0.2-1.5s range, CV=0.076

All detected files are saved in `results/` and have corresponding pattern plots in `pipeline_plots/`.

## Visualization Examples

### Pattern Detection Plot

Each detected file generates a pattern plot (`*_pattern.png`) with three views:

1. **Top Plot - Spike Detection**:
   - RMS energy over time (blue line)
   - **Lime/yellow vertical lines**: Detection spikes (count toward >=5 requirement)
   - **Green shaded areas**: Other spikes
   - Red dashed line: -10 dB threshold
   - Statistics box: Regularity score, CV, mean interval

2. **Middle Plot - Interval Analysis**:
   - Intervals between spikes over time
   - **Magenta markers**: Detection intervals (0.2-1.5s range, CV <= 0.3)
   - **Green markers**: Other intervals
   - **Blue shaded area**: Detection range (0.2-1.5 seconds)
   - Red dashed line: Mean interval
   - Shows CV (coefficient of variation) and regularity status

3. **Bottom Plot - Interval Distribution**:
   - Histogram showing distribution of all intervals
   - Red dashed line: Mean interval
   - Green dashed line: Mode interval (if regular)
   - Helps visualize clustering around detection range

**Example plots available in `pipeline_plots/`:**
- `20250821_000000_peak_890.6Hz_pattern.png` - 5 detection intervals, CV=0.217
- `20250821_061800_peak_1078.1Hz_pattern.png` - 8 detection intervals, CV=0.057 (very regular)
- `20250821_194200_peak_1031.2Hz_pattern.png` - 11 detection intervals, CV=0.211
- `20250822_053600_peak_1007.8Hz_pattern.png` - 18 detection intervals, CV=0.076 (very regular)

You can view these plots to see how the detection algorithm identifies owl calls through regular spike patterns.

### Understanding the Plots

**Detection spikes** (lime/yellow) are the spikes whose intervals between them fall in the 0.2-1.5s range and are consistent (low CV). These represent the actual owl calls detected.

**Other spikes** (green) may be noise, other sounds, or irregular patterns that don't meet the consistency requirement.

## Dependencies

See `requirements.txt` for complete list. Main dependencies:
- `librosa`: Audio analysis and processing
- `soundfile`: Audio file I/O
- `numpy`: Numerical computations
- `scipy`: Signal processing and statistics
- `matplotlib`: Plotting and visualization
