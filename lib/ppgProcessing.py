import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import bisect

class ppgSignalProcessing:
    def __init__(self, ppg_signal, fs = 125):
        self.ppg_signal = ppg_signal
        self.fs = fs
        self.normalized_signal = None
        self.pre_processed_signal = None
        self.downsampled_signal = None
        
        self.respiratory_signal_auto = None
        self.vasomotor_signal_auto = None
        self.cardiac_signal_auto = None

        self.T_delays_samples = [] # Will store [T1, T2, ..., T8]
        self.cardiac_signal_delay_samples = 0
        
        self.last_downsample_factor = 1
        
        self.j_signals = {}
        self.q_coeffs = {}
        self._init_filter_coefficients() 

    def _init_filter_coefficients(self):
        self.q_coeffs = {
            'q1' : np.array([-2, 2], dtype = float),
            'q2' : np.array([-1, -3, -2, 2, 3, 1], dtype = float),
            'q3' : np.array([-1, -3, -6, -10, -11, -9, -4, 4, 9, 11, 10, 6, 3, 1], dtype = float),
            'q4' : np.array([-1, -3, -6, -10, -15, -21, -28, -36, -41, -43, -42, -38, -31, -21, -8, 8, 
                             21, 31, 38, 42, 43, 41, 36, 28, 21, 15, 10, 6,  3, 1], dtype = float),
            'q5' : np.array([-1, -3, -6, -10, -15, -21, -28, -36, -45, -55, -66, -78, -91, -105, -120, -136, -149, -159, -166, -170, -171, -169, -164, -156,
                            -145, -131, -114,  -94,  -71, -45, -16, 16, 45, 71, 94, 114, 131, 145, 156,
                            164, 169, 171, 170, 166, 159, 149, 136, 120, 105, 91, 78, 66, 55, 45, 36, 28,
                            21, 15, 10, 6, 3, 1], dtype = float),
            'q6' : np.array([-1,   -3,   -6,  -10,  -15,  -21,  -28,  -36,  -45,  -55,  -66,
                            -78,  -91, -105, -120, -136, -153, -171, -190, -210, -231, -253,
                            -276, -300, -325, -351, -378, -406, -435, -465, -496, -528, -557,
                            -583, -606, -626, -643, -657, -668, -676, -681, -683, -682, -678,
                            -671, -661, -648, -632, -613, -591, -566, -538, -507, -473, -436,
                            -396, -353, -307, -258, -206, -151,  -93,  -32,   32,   93,  151,
                            206,  258,  307,  353,  396,  436,  473,  507,  538,  566,  591,
                            613,  632,  648,  661,  671,  678,  682,  683,  681,  676,  668,
                            657,  643,  626,  606,  583,  557,  528,  496,  465,  435,  406,
                            378,  351,  325,  300,  276,  253,  231,  210,  190,  171,  153,
                            136,  120,  105,   91,   78,   66,   55,   45,   36,   28,   21,
                            15,   10,    6,    3,    1], dtype = float),
            'q7' : np.array([-1,    -3,    -6,   -10,   -15,   -21,   -28,   -36,   -45,
                            -55,   -66,   -78,   -91,  -105,  -120,  -136,  -153,  -171,
                            -190,  -210,  -231,  -253,  -276,  -300,  -325,  -351,  -378,
                            -406,  -435,  -465,  -496,  -528,  -561,  -595,  -630,  -666,
                            -703,  -741,  -780,  -820,  -861,  -903,  -946,  -990, -1035,
                            -1081, -1128, -1176, -1225, -1275, -1326, -1378, -1431, -1485,
                            -1540, -1596, -1653, -1711, -1770, -1830, -1891, -1953, -2016,
                            -2080, -2141, -2199, -2254, -2306, -2355, -2401, -2444, -2484,
                            -2521, -2555, -2586, -2614, -2639, -2661, -2680, -2696, -2709,
                            -2719, -2726, -2730, -2731, -2729, -2724, -2716, -2705, -2691,
                            -2674, -2654, -2631, -2605, -2576, -2544, -2509, -2471, -2430,
                            -2386, -2339, -2289, -2236, -2180, -2121, -2059, -1994, -1926,
                            -1855, -1781, -1704, -1624, -1541, -1455, -1366, -1274, -1179,
                            -1081,  -980,  -876,  -769,  -659,  -546,  -430,  -311,  -189,
                            -64,    64,   189,   311,   430,   546,   659,   769,   876,
                            980,  1081,  1179,  1274,  1366,  1455,  1541,  1624,  1704,
                            1781,  1855,  1926,  1994,  2059,  2121,  2180,  2236,  2289,
                            2339,  2386,  2430,  2471,  2509,  2544,  2576,  2605,  2631,
                            2654,  2674,  2691,  2705,  2716,  2724,  2729,  2731,  2730,
                            2726,  2719,  2709,  2696,  2680,  2661,  2639,  2614,  2586,
                            2555,  2521,  2484,  2444,  2401,  2355,  2306,  2254,  2199,
                            2141,  2080,  2016,  1953,  1891,  1830,  1770,  1711,  1653,
                            1596,  1540,  1485,  1431,  1378,  1326,  1275,  1225,  1176,
                            1128,  1081,  1035,   990,   946,   903,   861,   820,   780,
                            741,   703,   666,   630,   595,   561,   528,   496,   465,
                            435,   406,   378,   351,   325,   300,   276,   253,   231,
                            210,   190,   171,   153,   136,   120,   105,    91,    78,
                            66,    55,    45,    36,    28,    21,    15,    10,     6,
                            3,     1], dtype = float),
            'q8' : np.array([-1,     -3,     -6,    -10,    -15,    -21,    -28,    -36,
                            -45,    -55,    -66,    -78,    -91,   -105,   -120,   -136,
                            -153,   -171,   -190,   -210,   -231,   -253,   -276,   -300,
                            -325,   -351,   -378,   -406,   -435,   -465,   -496,   -528,
                            -561,   -595,   -630,   -666,   -703,   -741,   -780,   -820,
                            -861,   -903,   -946,   -990,  -1035,  -1081,  -1128,  -1176,
                            -1225,  -1275,  -1326,  -1378,  -1431,  -1485,  -1540,  -1596,
                            -1653,  -1711,  -1770,  -1830,  -1891,  -1953,  -2016,  -2080,
                            -2145,  -2211,  -2278,  -2346,  -2415,  -2485,  -2556,  -2628,
                            -2701,  -2775,  -2850,  -2926,  -3003,  -3081,  -3160,  -3240,
                            -3321,  -3403,  -3486,  -3570,  -3655,  -3741,  -3828,  -3916,
                            -4005,  -4095,  -4186,  -4278,  -4371,  -4465,  -4560,  -4656,
                            -4753,  -4851,  -4950,  -5050,  -5151,  -5253,  -5356,  -5460,
                            -5565,  -5671,  -5778,  -5886,  -5995,  -6105,  -6216,  -6328,
                            -6441,  -6555,  -6670,  -6786,  -6903,  -7021,  -7140,  -7260,
                            -7381,  -7503,  -7626,  -7750,  -7875,  -8001,  -8128,  -8256,
                            -8381,  -8503,  -8622,  -8738,  -8851,  -8961,  -9068,  -9172,
                            -9273,  -9371,  -9466,  -9558,  -9647,  -9733,  -9816,  -9896,
                            -9973, -10047, -10118, -10186, -10251, -10313, -10372, -10428,
                            -10481, -10531, -10578, -10622, -10663, -10701, -10736, -10768,
                            -10797, -10823, -10846, -10866, -10883, -10897, -10908, -10916,
                            -10921, -10923, -10922, -10918, -10911, -10901, -10888, -10872,
                            -10853, -10831, -10806, -10778, -10747, -10713, -10676, -10636,
                            -10593, -10547, -10498, -10446, -10391, -10333, -10272, -10208,
                            -10141, -10071,  -9998,  -9922,  -9843,  -9761,  -9676,  -9588,
                            -9497,  -9403,  -9306,  -9206,  -9103,  -8997,  -8888,  -8776,
                            -8661,  -8543,  -8422,  -8298,  -8171,  -8041,  -7908,  -7772,
                            -7633,  -7491,  -7346,  -7198,  -7047,  -6893,  -6736,  -6576,
                            -6413,  -6247,  -6078,  -5906,  -5731,  -5553,  -5372,  -5188,
                            -5001,  -4811,  -4618,  -4422,  -4223,  -4021,  -3816,  -3608,
                            -3397,  -3183,  -2966,  -2746,  -2523,  -2297,  -2068,  -1836,
                            -1601,  -1363,  -1122,   -878,   -631,   -381,   -128,    128,
                            381,    631,    878,   1122,   1363,   1601,   1836,   2068,
                            2297,   2523,   2746,   2966,   3183,   3397,   3608,   3816,
                            4021,   4223,   4422,   4618,   4811,   5001,   5188,   5372,
                            5553,   5731,   5906,   6078,   6247,   6413,   6576,   6736,
                            6893,   7047,   7198,   7346,   7491,   7633,   7772,   7908,
                            8041,   8171,   8298,   8422,   8543,   8661,   8776,   8888,
                            8997,   9103,   9206,   9306,   9403,   9497,   9588,   9676,
                            9761,   9843,   9922,   9998,  10071,  10141,  10208,  10272,
                            10333,  10391,  10446,  10498,  10547,  10593,  10636,  10676,
                            10713,  10747,  10778,  10806,  10831,  10853,  10872,  10888,
                            10901,  10911,  10918,  10922,  10923,  10921,  10916,  10908,
                            10897,  10883,  10866,  10846,  10823,  10797,  10768,  10736,
                            10701,  10663,  10622,  10578,  10531,  10481,  10428,  10372,
                            10313,  10251,  10186,  10118,  10047,   9973,   9896,   9816,
                            9733,   9647,   9558,   9466,   9371,   9273,   9172,   9068,
                            8961,   8851,   8738,   8622,   8503,   8381,   8256,   8128,
                            8001,   7875,   7750,   7626,   7503,   7381,   7260,   7140,
                            7021,   6903,   6786,   6670,   6555,   6441,   6328,   6216,
                            6105,   5995,   5886,   5778,   5671,   5565,   5460,   5356,
                            5253,   5151,   5050,   4950,   4851,   4753,   4656,   4560,
                            4465,   4371,   4278,   4186,   4095,   4005,   3916,   3828,
                            3741,   3655,   3570,   3486,   3403,   3321,   3240,   3160,
                            3081,   3003,   2926,   2850,   2775,   2701,   2628,   2556,
                            2485,   2415,   2346,   2278,   2211,   2145,   2080,   2016,
                            1953,   1891,   1830,   1770,   1711,   1653,   1596,   1540,
                            1485,   1431,   1378,  1326,   1275,   1225,   1176,   1128,
                            1081,   1035,    990,    946,    903,    861,    820,    780,
                            741,    703,    666,    630,    595,    561,    528,    496,
                            465,    435,    406,    378,    351,    325,    300,    276,
                            253,    231,    210,    190,    171,    153,    136,    120,
                            105,     91,     78,     66,     55,     45,     36,     28,
                            21,     15,     10,      6,      3,      1], dtype = float)
        }

        self.q_coeffs['q2'] /= 4.0
        self.q_coeffs['q3'] /= 32.0
        self.q_coeffs['q4'] /= 256.0
        self.q_coeffs['q5'] /= 2048.0
        self.q_coeffs['q6'] /= 16384.0
        self.q_coeffs['q7'] /= 131072.0
        self.q_coeffs['q8'] /= 1048576.0

    def normalize_signal(self):
        if self.ppg_signal is None:
            print("Error: Signal not loaded.")
            return

        mean = np.mean(self.ppg_signal)
        std = np.std(self.ppg_signal)

        if std == 0:
            std = 1.0

        self.normalized_signal = (self.ppg_signal - mean) / std
        return self.normalized_signal

    def preProcessing(self):
        sig_len = len(self.normalized_signal)
        normalized_signal_arr = self.normalized_signal
        
        hpf_signal = [0.0] * sig_len
        b = [0.99598683, -2.98796050, 2.98796050, -0.99598683]
        a = [0, -2.99195753, 2.98394736, -0.99198977]

        if sig_len > 0:
            hpf_signal[0] = b[0] * normalized_signal_arr[0]
        if sig_len > 1:
            hpf_signal[1] = (b[0] * normalized_signal_arr[1] + 
                             b[1] * normalized_signal_arr[0] - 
                             a[1] * hpf_signal[0])
        if sig_len > 2:
            hpf_signal[2] = (b[0] * normalized_signal_arr[2] + 
                             b[1] * normalized_signal_arr[1] + 
                             b[2] * normalized_signal_arr[0] - 
                             a[1] * hpf_signal[1] - 
                             a[2] * hpf_signal[0])
        
        for n in range(3, sig_len):
            hpf_signal[n] = (b[0] * normalized_signal_arr[n] +
                             b[1] * normalized_signal_arr[n-1] +
                             b[2] * normalized_signal_arr[n-2] +
                             b[3] * normalized_signal_arr[n-3] -
                             a[1] * hpf_signal[n-1] -
                             a[2] * hpf_signal[n-2] -
                             a[3] * hpf_signal[n-3])

        self.pre_processed_signal = np.array(hpf_signal)
        return self.pre_processed_signal
    
    def _apply_iir_filter(self, signal_in, b, a):
        sig_len = len(signal_in)
        signal_out = np.zeros(sig_len)
        
        b0, b1, b2 = b
        a1, a2 = a
        
        if sig_len > 0:
            signal_out[0] = b0 * signal_in[0]
        if sig_len > 1:
            signal_out[1] = (b0 * signal_in[1] + 
                             b1 * signal_in[0] - 
                             a1 * signal_out[0])
        
        for n in range(2, sig_len):
            signal_out[n] = (b0 * signal_in[n] + 
                             b1 * signal_in[n-1] + 
                             b2 * signal_in[n-2] -
                             a1 * signal_out[n-1] - 
                             a2 * signal_out[n-2])
            
        return signal_out
    
    def dwtSignal(self, downsample_factor = 16):
        if not isinstance(downsample_factor, int) or downsample_factor <= 0:
            print("Warning: Downsample factor must be a positive integer. Defaulting to 1.")
            downsample_factor = 1

        self.last_downsample_factor = downsample_factor
        pre_processed_arr = self.pre_processed_signal
        
        signal_to_downsample = None

        if downsample_factor > 1:
            print(f"Applying adaptive LPF for downsample factor {downsample_factor}...")
            effective_fs = self.fs / downsample_factor
            fc = effective_fs / 2.0 
            fs_orig = self.fs
            
            print(f"  Original Fs: {fs_orig} Hz. New Fs: {effective_fs} Hz. LPF Cutoff (fc): {fc} Hz.")

            K = np.tan(np.pi * fc / fs_orig)
            sqrt2 = np.sqrt(2)
            
            norm = K**2 + sqrt2*K + 1
            
            b0 = (K**2) / norm
            b1 = (2 * K**2) / norm
            b2 = (K**2) / norm
            
            a1_raw = 2 * (K**2 - 1)
            a2_raw = K**2 - sqrt2*K + 1
            
            b_coeffs = [b0, b1, b2]
            a_coeffs = [a1_raw / norm, a2_raw / norm] 

            print("  Applying forward-backward (zero-phase) IIR filter...")
            filtered_fwd = self._apply_iir_filter(pre_processed_arr, b_coeffs, a_coeffs)
            filtered_rev = filtered_fwd[::-1]
            filtered_bwd = self._apply_iir_filter(filtered_rev, b_coeffs, a_coeffs)
            
            signal_to_downsample = filtered_bwd[::-1]
            print("  Anti-aliasing filter applied.")

        else:
            signal_to_downsample = pre_processed_arr

        downsampled_signal_list = []
        if downsample_factor == 1:
            downsampled_signal_list = list(signal_to_downsample)
        else:
            sig_len_new = len(signal_to_downsample) // downsample_factor
            source_index = 0
            for _ in range(sig_len_new):
                current_sum = 0.0
                for k in range(downsample_factor):
                    if (source_index + k) < len(signal_to_downsample):
                        current_sum += signal_to_downsample[source_index + k]
                
                downsampled_signal_list.append(current_sum / downsample_factor)
                source_index += downsample_factor
        
        self.downsampled_signal = np.array(downsampled_signal_list)
        print(f"Signal manually downsampled by factor {downsample_factor}. New length: {len(self.downsampled_signal)}")

        y_downsampled = self.downsampled_signal
        jumlahdata = len(y_downsampled)
        
        T1 = round(2**(1-1)) - 1 # 0
        T2 = round(2**(2-1)) - 1 # 1
        T3 = round(2**(3-1)) - 1 # 3
        T4 = round(2**(4-1)) - 1 # 7
        T5 = round(2**(5-1)) - 1 # 15
        T6 = round(2**(6-1)) - 1 # 31
        T7 = round(2**(7-1)) - 1 # 63
        T8 = round(2**(8-1)) - 1 # 127
        self.T_delays_samples = [T1, T2, T3, T4, T5, T6, T7, T8]
        T_max = self.T_delays_samples[-1] # Get max delay (T8)

        w2fb = np.zeros((9, jumlahdata + T_max)) 
        
        self.j_signals.clear()

        for n in range(jumlahdata):
            for j in range(1, 9): 
                delay = self.T_delays_samples[j-1]
                coeffs_key = f'q{j}'
                if coeffs_key not in self.q_coeffs:
                    continue 
                
                coeffs = self.q_coeffs[coeffs_key]

                for k_idx in range(len(coeffs)):
                    idx = n - k_idx 
                    if 0 <= idx < jumlahdata:
                        w2fb[j][n + delay] += coeffs[k_idx] * y_downsampled[idx]
        
        for j in range(1, 9):
            self.j_signals[f'j{j}'] = w2fb[j]

        print(f"Signal manually decomposed into {len(self.j_signals)} levels (using user's delay logic).")
        return self.j_signals
    
    def _find_best_signal_for_band(self, fs, min_freq, max_freq):
        best_signal_key = None
        max_power = -1

        for key, signal in self.j_signals.items():
            if len(signal) > 1: 
                n = len(signal)
                fft_vals = fft(signal)
                freqs = fftfreq(n, d=1/fs)[:n//2]
                power_spectrum = (np.abs(fft_vals)[:n//2])**2
                freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
                power_in_band = np.sum(power_spectrum[freq_mask])
                
                print(f"Scale {key}: Power in {min_freq}-{max_freq} Hz band = {power_in_band:.4f}")

                if power_in_band > max_power:
                    max_power = power_in_band
                    best_signal_key = key
        
        return best_signal_key

    def find_respiratory_signal(self, effective_fs):
        print("\n--- Analyzing for best RESPIRATORY signal (0.15-0.40 Hz) ---")
        resp_key = self._find_best_signal_for_band(effective_fs, min_freq=0.15, max_freq=0.35)
        
        if resp_key:
            self.respiratory_signal_auto = np.copy(self.j_signals[resp_key])
            
            try:
                window_size = 5 
                if len(self.respiratory_signal_auto) > window_size:
                    print(f"Applying {window_size}-point MAV filter to respiratory signal...")
                    
                    weights = np.ones(window_size) / window_size
                    self.respiratory_signal_auto = np.convolve(
                        self.respiratory_signal_auto, 
                        weights, 
                        mode='same'
                    )
                else:
                    print("Respiratory signal too short for MAV, skipping.")
                    
            except Exception as e:
                print(f"Error applying MAV filter: {e}. Using raw signal.")

        return resp_key

    def find_vasomotor_signal(self, effective_fs):

        print("\n--- Analyzing for best VASOMOTOR signal (0.04-0.15 Hz) ---")
        vaso_key = self._find_best_signal_for_band(effective_fs, min_freq=0.04, max_freq=0.15)
        if vaso_key:
            self.vasomotor_signal_auto = np.copy(self.j_signals[vaso_key])
        return vaso_key
    
    @staticmethod
    def calculate_peak_frequency_fft(signal, fs, min_freq_hz=0.15, max_freq_hz=0.40):
        """
        Calculates the peak frequency of a signal within a specified band using FFT.
        """
        n = len(signal)
        if n == 0:
            return 0.0, np.array([]), np.array([])

        window = np.hanning(n)
        signal_windowed = signal * window

        fft_vals = fft(signal_windowed)[:n//2]
        fft_freqs = fftfreq(n, d=1/fs)[:n//2]
        
        power_spectrum = np.abs(fft_vals)**2
        
        freq_mask = (fft_freqs >= min_freq_hz) & (fft_freqs <= max_freq_hz)
        
        power_in_band = power_spectrum[freq_mask]
        
        peak_freq_hz = 0.0
        
        if len(power_in_band) > 0:
            # Find the index of the peak power *within the masked band*
            peak_power_index_in_mask = np.argmax(power_in_band)
            
            # Find the corresponding frequency from the *masked frequencies*
            peak_freq_hz = fft_freqs[freq_mask][peak_power_index_in_mask]
        
        # Return the peak frequency in Hz, all frequencies, and the full power spectrum
        return peak_freq_hz, fft_freqs, power_spectrum
    
    def find_cardiac_signal(self, effective_fs):
        print(f"\n--- Analyzing for best CARDIAC signal (0.8-3.0 Hz) at Fs={effective_fs} Hz ---")
        cardiac_key = self._find_best_signal_for_band(effective_fs, min_freq=0.8, max_freq=3.0)

        self.cardiac_signal_delay_samples = 0

        if cardiac_key:
            self.cardiac_signal_auto = np.copy(self.j_signals[cardiac_key])
            try:
                # Get the 'j' number (e.g., 5 from 'j5')
                j_index = int(cardiac_key.replace('j', ''))
                # Look up the corresponding delay from our list
                if 1 <= j_index <= len(self.T_delays_samples):
                    self.cardiac_signal_delay_samples = self.T_delays_samples[j_index - 1]
                    print(f"Stored delay for {cardiac_key}: {self.cardiac_signal_delay_samples} samples.")
            except Exception as e:
                print(f"Warning: Could not parse delay for {cardiac_key}: {e}")
        return cardiac_key

class peakDetection:
    @staticmethod
    def _find_extreme_points(signal, sampling_rate):
        PEAK_DURATION_MS = 300.0
        signal_len = len(signal)
        search_radius = round((PEAK_DURATION_MS / 1000.0 * sampling_rate) / 2.0)
        if search_radius < 1:
            search_radius = 1

        maxima_indices = []
        minima_indices = []

        i = 1
        while i <= signal_len - search_radius - 1:
            is_peak = True
            is_valley = True

            for j in range(1, search_radius + 1):
                if (signal[i] <= signal[i + j]) or (signal[i] <= signal[i - j]):
                    is_peak = False
                    break
            
            if not is_peak:
                for j in range(1, search_radius + 1):
                    if (signal[i] >= signal[i + j]) or (signal[i] >= signal[i - j]):
                        is_valley = False
                        break
            
            if is_peak:
                maxima_indices.append(i)
                i += search_radius  
                continue 

            if is_valley:
                minima_indices.append(i)
                i += search_radius 
            else:
                i += 1 
        return maxima_indices, minima_indices

    @staticmethod
    def _process_window_with_pd(extreme_points_in_window, signal, window_start, window_length, thresholds, M):
        a, b, TH_min = 0.5, 0.1, 0.003
        count = 0
        max_s = -float('inf')
        max_pos = -1
        for ex_pos in extreme_points_in_window:
            if window_length > 0:
                interpolated_th = thresholds['th2'] - (thresholds['th2'] - thresholds['th3']) / window_length * (ex_pos - window_start)
            else:
                interpolated_th = thresholds['th2']
            if signal[ex_pos] > M * interpolated_th:
                count += 1
                if signal[ex_pos] > max_s:
                    max_s = signal[ex_pos]
                    max_pos = ex_pos
        if count >= 1:
            found_peak = True
            r_peak_position = int(round(max_pos))
            temp_th2 = max_s
            temp_th3 = temp_th2 - a * (temp_th2 - TH_min) - b * (temp_th2 - thresholds['th1'])
            thresholds['th1'] = thresholds['th2']
            thresholds['th2'] = temp_th2
            thresholds['th3'] = temp_th3
        else:
            found_peak = False
            r_peak_position = -1
            temp_th3 = thresholds['th3'] - a * (thresholds['th3'] - TH_min) - b * (thresholds['th3'] - thresholds['th2'])
            thresholds['th1'] = thresholds['th2']
            thresholds['th2'] = thresholds['th3']
            thresholds['th3'] = temp_th3
        return found_peak, r_peak_position

    def detect_peaks(signal, fs, effective_fs, min_peak_dist_ms=1500.0, initial_th_multiplier=0.15, threshold_m_factor=1.3):
        WINDOW_DURATION_MS = 250.0
        PEAK_SEARCH_WINDOW_MS = 200.0
        MIN_PEAK_DISTANCE_MS = min_peak_dist_ms

        sampling_rate = effective_fs
        window_length = int(round(WINDOW_DURATION_MS / 1000.0 * sampling_rate))
        min_peak_distance_samples = int(round(MIN_PEAK_DISTANCE_MS / 1000.0 * sampling_rate))
        last_accepted_peak_index = -min_peak_distance_samples * 2

        peak_candidates = []
        final_peaks = []

        maxima_indices, minima_indices = peakDetection._find_extreme_points(signal, sampling_rate)
        two_seconds_samples = min(len(signal), int(round(2 * sampling_rate)))
        
        max_s_initial = np.max(signal[:two_seconds_samples]) if two_seconds_samples > 0 else 0

        initial_th = initial_th_multiplier * max_s_initial
        thresholds = {'th1': initial_th, 'th2': initial_th, 'th3': initial_th}
        current_pos = 0

        while current_pos < len(signal):
            window_points = [p for p in maxima_indices if current_pos <= p < current_pos + window_length]
            found_peak, found_peak_pos = peakDetection._process_window_with_pd(
                window_points, signal, current_pos, window_length, thresholds, threshold_m_factor
            )
            if found_peak:
                if (found_peak_pos - last_accepted_peak_index) > min_peak_distance_samples:
                    peak_candidates.append(found_peak_pos)
                    last_accepted_peak_index = found_peak_pos
            current_pos += window_length
        search_window_samples = int(round(PEAK_SEARCH_WINDOW_MS / 1000.0 * sampling_rate))

        for found_peak_pos in peak_candidates:
            start = max(0, found_peak_pos - search_window_samples)
            end = min(len(signal), found_peak_pos + search_window_samples)
            if start < end:
                window_slice = signal[start:end]
                actual_peak_index = np.argmax(window_slice) + start
                if not final_peaks or actual_peak_index != final_peaks[-1]:
                    final_peaks.append(actual_peak_index)

        return final_peaks, maxima_indices, minima_indices

class Spline:
    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)
        h = np.diff(x)

        self.a = [iy for iy in y]
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                 (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        return result

    def __search_index(self, x):
        i = bisect.bisect_right(self.x, x) - 1
        
        if i >= self.nx - 1:
            i = self.nx - 2
        return i

    def __calc_A(self, h):
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h):
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                       h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B