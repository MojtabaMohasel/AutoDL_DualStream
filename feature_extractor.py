import numpy as np
from scipy.signal import stft

class FeatureExtractor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def compute_stft_features(self, X_time_data, fs, nperseg):
        """
        Computes STFT for time-series data using a directly provided nperseg.
        X_time_data: (num_windows, window_length_samples, num_features)
        fs: Sampling frequency
        nperseg: The segment length for STFT (this is now used directly).
        """


        if X_time_data.ndim != 3 or X_time_data.shape[0] == 0:
            self.logger.warning(f"STFT input X_time_data is empty or has incorrect dimensions: {X_time_data.shape}. Returning empty array.")



            channels = X_time_data.shape[2] if X_time_data.ndim == 3 and X_time_data.shape[2] > 0 else 1
            return np.array([]).reshape(0, 1, 1, channels)


        if nperseg <= 0:
            self.logger.warning(f"Received nperseg ({nperseg}) for STFT is not positive. Setting to 1. This may affect STFT quality.")
            nperseg = 1

        self.logger.info(f"Computing STFT with fs={fs}, nperseg={nperseg} (received directly)")



        num_total_windows, _, num_features = X_time_data.shape
        all_stft_results = []

        for i in range(num_total_windows):
            window_spectrograms = []
            for feature_idx in range(num_features):

                signal_slice = X_time_data[i, :, feature_idx]
                if len(signal_slice) < nperseg:
                    self.logger.warning(f"Signal slice length {len(signal_slice)} < nperseg {nperseg} for window {i}, feature {feature_idx}. Padding with zeros.")
                    padding_length = nperseg - len(signal_slice)
                    signal_slice = np.pad(signal_slice, (0, padding_length), 'constant')

                _, _, Zxx = stft(signal_slice, fs=fs, nperseg=nperseg)
                window_spectrograms.append(np.abs(Zxx))


            stacked_spectrograms = np.stack(window_spectrograms, axis=0)
            all_stft_results.append(stacked_spectrograms)

        if not all_stft_results:
             self.logger.warning("STFT computation resulted in no data.")
             num_features = X_time_data.shape[2] if X_time_data.ndim == 3 else 1
             return np.array([]).reshape(0, 1, 1, num_features)


        final_stft_array = np.array(all_stft_results)
        final_stft_array = final_stft_array.transpose(0, 2, 3, 1)

        final_stft_array = np.nan_to_num(final_stft_array.astype(float), nan=0.0)

        self.logger.info(f"STFT computation complete. Output shape: {final_stft_array.shape}")
        return final_stft_array
