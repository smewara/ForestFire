import os
import pyloudnorm
import soundfile as sf
import librosa
import numpy as np

# Input and output folders
input_folder = "No Fire"
output_folder = "No_Fire_Normalized"

# Normalises to EBU standard, splits audio into 3 second chunks, converts to 16kHz
def normalize_audio(input_folder, output_folder, target_loudness=-23):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over audio files in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav") or filename.endswith(".mp3"): 
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)

            # Load audio file
            audio, sample_rate = sf.read(input_file)
            
            # Measure loudness
            meter = pyloudnorm.Meter(sample_rate)
            
            loudness = meter.integrated_loudness(audio)

            # Normalize audio
            normalized_audio = pyloudnorm.normalize.loudness(audio, loudness, target_loudness)

            # Split audio
            normalized_audio = split_audio_into_segments(normalized_audio, 16000)

            # Save segmented audio chunks
            for i, segment in enumerate(normalized_audio):
                segment_audio, start_time = segment
                segment_filename = f"{os.path.splitext(filename)[0]}_{i}.wav"
                segment_output_file = os.path.join(output_folder, segment_filename)
                sf.write(segment_output_file, segment_audio, 16000, subtype='PCM_16')

            print(f"Normalized, converted, and segmented {filename} successfully.")

def split_audio_into_segments(audio, sample_rate, duration=3, overlap=0.5):
    segment_samples = int(duration * sample_rate)
    hop_length = int(segment_samples * (1 - overlap))
    # Split audio into segments
    segments = []
    start_sample = 0
    while start_sample + segment_samples < len(audio):
        segment = audio[start_sample:start_sample + segment_samples]
        segments.append((segment, start_sample / sample_rate))
        start_sample += hop_length

    return segments


# Call function
normalize_audio(input_folder, output_folder)
