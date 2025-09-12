from swift_f0 import *

# Initialize the detector
# For speech analysis, consider setting fmin=65 and fmax=400
detector = SwiftF0(fmin=46.875, fmax=2093.75, confidence_threshold=0.9)

# Run pitch detection from an audio file
result = detector.detect_from_file("recorded_samples.wav")

# For raw audio arrays (e.g., loaded via librosa or scipy)
# result = detector.detect_from_array(audio_data, sample_rate)

# Visualize and export results
plot_pitch(result, show=False, output_path="pitch.jpg")
export_to_csv(result, "pitch_data.csv")

# Segment pitch contour into musical notes
notes = segment_notes(
    result,
    split_semitone_threshold=0.8,
    min_note_duration=0.05
)
plot_notes(notes, output_path="note_segments.jpg")
plot_pitch_and_notes(result, notes, output_path="combined_analysis.jpg")
export_to_midi(notes, "notes.mid")