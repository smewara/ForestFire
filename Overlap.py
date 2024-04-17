import os
import numpy as np
import soundfile as sf

# Folders
fire_folder = "Fire_Normalized"
nofire_folder = "No_Fire_Normalized"
output_folder = "fire_nofire_mix"

def overlap_fire_and_nofire(fire_folder, nofire_folder, output_folder, fire_volume_mean=0.5, 
                            fire_volume_std=0.1, fire_gain_factor=2.0):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of fire and no-fire sound files
    fire_files = [os.path.join(fire_folder, f) for f in os.listdir(fire_folder) if f.endswith(".wav")]
    nofire_files = [os.path.join(nofire_folder, f) for f in os.listdir(nofire_folder) if f.endswith(".wav")]

    # Iterate over each pair of fire and no-fire sound files
    for fire_file, nofire_file in zip(fire_files, nofire_files):
        # Load fire and no-fire sounds
        fire_sound, fire_sr = sf.read(fire_file)
        nofire_sound, nofire_sr = sf.read(nofire_file)

        # Ensure both sounds have the same sample rate
        assert fire_sr == nofire_sr, "Sample rates of fire and no-fire sounds do not match"

        # Determine length of the overlapped sound (use the shorter length)
        overlapped_length = min(len(fire_sound), len(nofire_sound))

        # Convert fire sound to mono by taking the mean of stereo channels
        fire_sound_mono = np.mean(fire_sound, axis=1)

        # Generate random volume level for the fire sound from a normal distribution
        fire_volume = np.random.normal(fire_volume_mean, fire_volume_std)

        # Normalize fire volume to [0, 1] range
        fire_volume = np.clip(fire_volume, 0, 1)

        # Apply fire gain factor
        fire_sound_mono = fire_sound_mono[:overlapped_length] * fire_gain_factor
        
        # Resize no-fire sound to match length of fire sound
        nofire_sound_resized = nofire_sound[:overlapped_length]

        # Mix fire and no-fire sounds
        overlapped_sound = fire_sound_mono + nofire_sound_resized

        # Normalize overlapped sound
        max_amplitude = max(np.max(np.abs(overlapped_sound)), 1.0)
        overlapped_sound /= max_amplitude

        # Save overlapped sound
        output_filename = os.path.basename(fire_file)[:-4] + "_Fire+NoFire.wav"
        output_path = os.path.join(output_folder, output_filename)
        sf.write(output_path, overlapped_sound, fire_sr)

        print(f"Created Fire+NoFire sound: {output_filename}")

# Call function
overlap_fire_and_nofire(fire_folder, nofire_folder, output_folder)
