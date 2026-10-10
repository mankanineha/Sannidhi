import numpy as np

class CSIProcessor:
    """
    ESP-Fi-HAR standard preprocessing pipeline matching:
    https://github.com/AutoSmartGroup/ESP-Fi-HAR

    Contract:
    - 52 OFDM subcarriers (amplitude only: 26 negative + 26 positive)
    - 950 frames per model window
    - Global per-sample Z-score standardization: (x - mean) / (std + 1e-8)
    - Output tensor shape: (1, 950, 52) float32
    """
    def __init__(self, window_size=950, feature_count=52):
        self.window_size = window_size
        self.feature_count = feature_count
        self.buffer = []

        # 52 active OFDM data subcarrier indices in 64-bin FFT (802.11n HT20):
        # Negative subcarriers (-26 to -1): bins 38 to 63 (26 bins)
        # Positive subcarriers (+1 to +26): bins 1 to 26 (26 bins)
        self.valid_subcarriers = list(range(38, 64)) + list(range(1, 27))

    def extract_amplitude_from_iq(self, raw_iq_data, first_word_invalid=False):
        """
        Extracts 52 calibrated amplitude subcarriers from 128 raw ESP32 I/Q bytes.
        """
        if len(raw_iq_data) != 128:
            return None

        iq_array = np.asarray(raw_iq_data, dtype=np.float32).reshape(64, 2)
        # Amplitude = sqrt(I^2 + Q^2)
        amplitude = np.sqrt(iq_array[:, 0]**2 + iq_array[:, 1]**2)

        # Handle ESP32 first_word_invalid hardware quirk (corrupts bytes 0-3 = bins 0 & 1)
        if first_word_invalid:
            # Interpolate bin 1 (+1) from neighboring active subcarrier bin 2 (+2)
            amplitude[1] = amplitude[2]

        return amplitude[self.valid_subcarriers]

    def add_frame(self, features_52):
        """
        Add a 52-feature amplitude frame to the rolling buffer.
        """
        feat = np.asarray(features_52, dtype=np.float32)
        if feat.size != self.feature_count:
            return None

        self.buffer.append(feat)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        if len(self.buffer) < self.window_size:
            return None

        # Format full 950x52 window
        raw_matrix = np.asarray(self.buffer, dtype=np.float32)
        # Global sample Z-score normalization matching ESP-Fi-HAR dataset.py
        mean = float(np.mean(raw_matrix))
        std = float(np.std(raw_matrix)) + 1e-8
        normalized = (raw_matrix - mean) / std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        input_tensor = np.expand_dims(normalized, axis=0)  # (1, 950, 52)
        return raw_matrix, input_tensor

    def clear(self):
        self.buffer.clear()

