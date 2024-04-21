def normalize_audio(audio, sample_rate, target_loudness=-23, augment=False, augmentation_probabilities=[0.4, 0.2, 0.2, 0.2]):
    # Measure loudness
    meter = pyloudnorm.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)

    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio
    normalized_audio = pyloudnorm.normalize.loudness(audio, loudness, target_loudness)

    # Augment audio if requested
    if augment:
        augmentation_type = np.random.choice(['none', 'pitch_shift', 'time_stretch', 'both'], p=augmentation_probabilities)
        if augmentation_type == 'none':
            augmented_audio = normalized_audio
        elif augmentation_type == 'pitch_shift':
            pitch_shift_factor = np.random.uniform(-4, 4)  
            augmented_audio = librosa.effects.pitch_shift(normalized_audio, sr=sample_rate, n_steps=pitch_shift_factor)
        elif augmentation_type == 'time_stretch':
            stretch_rate = np.random.uniform(0.8, 1.2)  
            augmented_audio = librosa.effects.time_stretch(normalized_audio, rate=stretch_rate)
        else: #augmentation_type == both
            pitch_shift_factor = np.random.uniform(-4, 4)  
            stretch_rate = np.random.uniform(0.8, 1.2)  
            pitch_shifted_audio = librosa.effects.pitch_shift(normalized_audio, sr=sample_rate, n_steps=pitch_shift_factor)
            augmented_audio = librosa.effects.time_stretch(pitch_shifted_audio, rate=stretch_rate)
            return augmented_audio
    return augmented_audio if augment else normalized_audio
