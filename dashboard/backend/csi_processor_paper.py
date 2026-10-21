import numpy as np

class CSIProcessorCSIBench:
    """
    Implements the preprocessing to map ESP32 1x1 data (64 subcarriers)
    to the CSI-Bench model's expected 2x2 NXP format (116 amp + 116 phase = 232 features).
    """
    def __init__(self, sampling_rate=100, window_seconds=5):
        self.window_size = sampling_rate * window_seconds # 500 packets
        self.buffer = [] 

    def process_raw_esp32_csi(self, raw_iq_data):
        if len(raw_iq_data) != 128: # 64 I + 64 Q
            return None
            
        iq_array = np.array(raw_iq_data).reshape(64, 2)
        I = iq_array[:, 0]
        Q = iq_array[:, 1]
        
        # 1. Extract Amplitude and Phase
        amplitude = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)
        
        # 2. Map to 116 features (simulating NXP 2x2 MIMO with 58 subcarriers per stream)
        # We take 58 valid subcarriers from the ESP32 (e.g., indices 3 to 60)
        valid_amp = amplitude[3:61] # 58 features
        valid_phase = phase[3:61]   # 58 features
        
        # Duplicate to simulate the 2 spatial streams the model expects (58 * 2 = 116)
        mapped_amp = np.tile(valid_amp, 2)     # 116 features
        mapped_phase = np.tile(valid_phase, 2) # 116 features
        
        # Combine into the flat 232-feature vector the model expects
        combined_features = np.concatenate([mapped_amp, mapped_phase])
        
        # 3. Buffer the data
        self.buffer.append(combined_features)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            
        if len(self.buffer) < self.window_size:
            return None # Wait until we have 500 packets (5 seconds)
            
        # 4. Matrix Normalization (Z-score over the temporal window)
        csi_matrix = np.array(self.buffer) # Shape: (500, 232)
        
        mu_H = np.mean(csi_matrix, axis=0)
        sigma_H = np.std(csi_matrix, axis=0) + 1e-6
        
        normalized_matrix = (csi_matrix - mu_H) / sigma_H
        
        # Return the fully prepared (500, 232) matrix ready for ONNX inference
        return normalized_matrix
