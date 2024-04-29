import pywt
import numpy as np
from scipy.signal import resample
from DataProcessors.SpectrogramProcessor import SpectrogramProcessor

class CWTProcessor(SpectrogramProcessor):
    def __init__(self, n_fft=2048, hop_length=512, wavelet='morl', num_scales=64, target_size=(64, 512)):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.target_size = target_size

    def compute_segmented_spectrograms(self, audio_path, kind_of_augmentation = None, duration_in_sec=2.5):
        # Load audio file
        y, sr = super().normalize_audio(audio_path=audio_path, kind_of_augmentation=kind_of_augmentation)
        
        # Split audio into segments
            segments = super().split_audio_into_segments(audio, sample_rate, duration=2.5, overlap=0.5):

            for segment, start_time in segments:

                # Compute CWT scalograms from segments
                segment_scalogram, _ = compute_cwt_scalogram_single(segment, sr, wavelet, num_scales, target_size)
                
                unique_id = str(uuid.uuid4())[:8]
            
                # Construct the filename with a unique code
                segment_filename = f"{os.path.splitext(filename)[0]}_{idx}_{start_time:.2f}_{unique_id}"
                
                os.makedirs(output_dir, exist_ok=True)
                np.save(os.path.join(output_dir, f"{segment_filename}_scalogram.npy"), segment_scalogram)

        return None

    def compute_cwt_scalogram_single(audio, sample_rate, wavelet='morl', num_scales=64, target_size=(64, 512)):
        scales = np.linspace(1, 64, num_scales, dtype=int)
        coeffs, freqs = pywt.cwt(audio, scales, wavelet)
        scalogram = np.abs(coeffs) ** 2
        
        # Zero-pad the scalogram to a fixed size
        padded_scalogram = np.zeros((num_scales, coeffs.shape[1]))
        padded_scalogram[:scalogram.shape[0], :scalogram.shape[1]] = scalogram
        
        # Normalize the scalogram
        padded_scalogram = (padded_scalogram - padded_scalogram.min()) / (padded_scalogram.max() - padded_scalogram.min())
        
        # Resample the scalogram to the target size
        resampled_scalogram = resample(padded_scalogram, target_size[0], axis=0)
        resampled_scalogram = resample(resampled_scalogram, target_size[1], axis=1)
        
        return resampled_scalogram, freqs
