from lib.ppgProcessing import ppgSignalProcessing, peakDetection, Spline 
import numpy as np
import matplotlib.pyplot as plt 
import math 
from scipy.integrate import trapezoid 

def rrInterval(signal, fs, effective_fs, **peak_params):
    print(f"\nDetecting Peaks for RR Interval Calculation")
    
    final_peaks, maxima, minima = peakDetection.detect_peaks(
        signal, fs, effective_fs, **peak_params
    )
    
    print(f"Found {len(final_peaks)} systolic peaks")
    
    if len(final_peaks) < 2:
        print("Not enough peaks found to calculate RR intervals")
        return final_peaks, maxima, minima, np.array([])

    rr_intervals_samples = np.diff(final_peaks)
    rr_intervals_ms = (rr_intervals_samples / effective_fs) * 1000.0
    
    if rr_intervals_ms.size > 0:
        mean_rr_ms = np.mean(rr_intervals_ms)
        mean_hr_bpm = 60000.0 / mean_rr_ms if mean_rr_ms > 0 else 0
        print(f"Mean Heart Rate: {mean_hr_bpm:.2f} bpm")

    return final_peaks, maxima, minima, rr_intervals_ms


class hrvCalculation:
    def _mean(self, data):
        n = len(data)
        if n == 0:
            return 0.0
        sum_val = 0
        for x in data:
            sum_val += x
        return sum_val / n

    def _std(self, data, mean_val=None):
        n = len(data)
        if n == 0:
            return 0.0
        if mean_val is None:
            mean_val = self._mean(data)
        variance_sum = 0
        for x in data:
            variance_sum += (x - mean_val) ** 2
        variance = variance_sum / n
        return variance ** 0.5
    
    def _median(self, data):
        n = len(data)
        if n == 0:
            return 0.0
        
        sorted_data = sorted(data)
        midpoint = n // 2
        
        if n % 2 == 1:
            return sorted_data[midpoint]
        else:
            return (sorted_data[midpoint - 1] + sorted_data[midpoint]) / 2.0
    
    def calculate_time_domain(self, rr_intervals_ms, segment_width_min=5):
        rr_intervals = list(rr_intervals_ms)
        rr_count = len(rr_intervals)

        if rr_count < 2:
            print("HRV Warning: Not enough RR intervals to calculate metrics")
            return {
                'RR (count)': 0, 'Mean RR (ms)': 0, 'Mean HR (bpm)': 0,
                'SDNN (ms)': 0, 'RMSSD (ms)': 0, 'SDSD (ms)': 0,
                'NN50 (count)': 0, 'pNN50 (%)': 0, 'SDANN (ms)': 0,
                'SDNN Index (ms)': 0, 'HRV Triangular Index': 0, 'TINN (ms)': 0,
                'Skewness': 0,
                'CVNN (%)': 0, 'CVSD (%)': 0 
            }

        mean_rr = self._mean(rr_intervals)
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0
        sdnn = self._std(rr_intervals, mean_rr) 

        rr_diffs = []
        for i in range(rr_count - 1):
            diff = rr_intervals[i+1] - rr_intervals[i]
            rr_diffs.append(diff)
        rr_diff_count = len(rr_diffs)

        rmssd_sum_sq = 0
        nn50_count = 0

        for diff in rr_diffs:
            rmssd_sum_sq += diff ** 2
            abs_diff = diff if diff >= 0 else -diff 
            if abs_diff > 50:
                nn50_count += 1

        rmssd = (rmssd_sum_sq / rr_diff_count) ** 0.5 if rr_diff_count > 0 else 0
        nn50 = nn50_count
        pNN50 = (nn50 / rr_diff_count) * 100.0 if rr_diff_count > 0 else 0
        mean_diff = self._mean(rr_diffs)
        sdsd = self._std(rr_diffs, mean_diff)
        
        segment_ms = segment_width_min * 60 * 1000
        segment_means, segment_sds = [], []
        current_segment_rrs, current_segment_duration = [], 0

        for rr in rr_intervals:
            if (current_segment_duration + rr) > segment_ms and len(current_segment_rrs) > 1:
                seg_mean = self._mean(current_segment_rrs)
                seg_sd = self._std(current_segment_rrs, seg_mean)
                segment_means.append(seg_mean)
                segment_sds.append(seg_sd)
                current_segment_rrs, current_segment_duration = [rr], rr
            else:
                current_segment_rrs.append(rr)
                current_segment_duration += rr

        if len(current_segment_rrs) > 1:
            seg_mean = self._mean(current_segment_rrs)
            seg_sd = self._std(current_segment_rrs, seg_mean)
            segment_means.append(seg_mean)
            segment_sds.append(seg_sd)
        
        sdann = self._std(segment_means) if len(segment_means) > 1 else 0.0
        sdnn_index = self._mean(segment_sds) if len(segment_sds) > 0 else 0.0

        cvnn = (sdnn / mean_rr)  if mean_rr > 0 else 0.0
        cvsd = (rmssd / mean_rr)  if mean_rr > 0 else 0.0

        bin_width_ms = 8
        bins = {} 
        max_bin_count = 0
        min_rr = rr_intervals[0]
        max_rr = rr_intervals[0]
        
        for rr in rr_intervals:
            bin_index = int(rr // bin_width_ms)
            bins[bin_index] = bins.get(bin_index, 0) + 1
            if bins[bin_index] > max_bin_count:
                max_bin_count = bins[bin_index]
            if rr < min_rr: min_rr = rr
            if rr > max_rr: max_rr = rr
        
        hrv_ti = (rr_count / max_bin_count) if max_bin_count > 0 else 0.0
        tinn = max_rr - min_rr

        skew_sum = 0
        for rr in rr_intervals:
            skew_sum += (rr - mean_rr) ** 3
        m3 = skew_sum / rr_count 
        skewness = (m3 / (sdnn ** 3)) if sdnn > 0 else 0.0

        metrics = {
            'RR (count)': rr_count, 'Mean RR (ms)': mean_rr, 'Mean HR (bpm)': mean_hr,
            'SDNN (ms)': sdnn, 'RMSSD (ms)': rmssd, 'SDSD (ms)': sdsd,
            'NN50 (count)': nn50, 'pNN50 (%)': pNN50, 'SDANN (ms)': sdann,
            'SDNN Index (ms)': sdnn_index, 'HRV Triangular Index': hrv_ti, 
            'TINN (ms)': tinn,
            'Skewness': skewness,
            'CVNN': cvnn, 
            'CVSD': cvsd  
        }
        return metrics

    def plot_tachogram(self, rr_intervals_ms):
        beat_numbers = np.arange(len(rr_intervals_ms))
        fig = plt.figure(figsize=(12, 6))
        
        plt.plot(beat_numbers, rr_intervals_ms, marker='o', linestyle='-', markersize=4, label='RR Interval')

        mean_rr = self._mean(list(rr_intervals_ms)) 
        median_rr = self._median(list(rr_intervals_ms))

        plt.axhline(mean_rr, color='red', linestyle='--', label=f'Mean RR: {mean_rr:.2f} ms')
        plt.axhline(median_rr, color='orange', linestyle='--', label=f'Median RR: {median_rr:.2f} ms')

        plt.title("RR Tachogram (Beat-to-Beat Intervals)")
        plt.xlabel("Beat Number")
        plt.ylabel("RR Interval (ms)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return fig
    
    def plot_hr_tachogram(self, rr_intervals_ms):
        hr_bpm = 60000.0 / rr_intervals_ms
        beat_numbers = np.arange(len(hr_bpm))

        hr_list = list(hr_bpm)
        mean_hr = self._mean(hr_list) 
        median_hr = self._median(hr_list)
        std_hr = self._std(hr_list, mean_val=mean_hr)

        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        ax.set_facecolor('#E0E0E0')
        plt.grid(color='white', linestyle='-', linewidth=0.5, zorder=0)

        plt.plot(beat_numbers, hr_bpm, marker='o', linestyle='-', markersize=2, 
                 color='black', markeredgecolor='black', markeredgewidth=0.5, 
                 label='Heart Rate (BPM)', zorder=2)
        
        plt.axhline(mean_hr, color='red', linestyle='-', 
                    label=f'Mean HR: {mean_hr:.2f} BPM', zorder=3)
        plt.axhline(median_hr, color='green', linestyle=':', 
                    label=f'Median HR: {median_hr:.2f} BPM', zorder=3)
        
        plt.axhline(mean_hr + std_hr, color='red', linestyle='--', 
                    label=f'+1 SD ({mean_hr + std_hr:.2f} BPM)', zorder=3)
        plt.axhline(mean_hr - std_hr, color='red', linestyle='--', 
                    label=f'-1 SD ({mean_hr - std_hr:.2f} BPM)', zorder=3)

        plt.title("HRV Tachogram")
        plt.xlabel("Beat Number")
        plt.ylabel("BPM")
        plt.ylim(40, 140) 
        plt.legend(facecolor='white')
        plt.tight_layout()

        return fig

    def plot_hti_histogram(self, rr_intervals_ms, calculated_hti_value=None):
        rr_list = list(rr_intervals_ms)
        bin_width_ms = 8.0

        min_rr = min(rr_list)
        max_rr = max(rr_list)

        start_bin = math.floor(min_rr / bin_width_ms) * bin_width_ms
        end_bin = math.ceil(max_rr / bin_width_ms) * bin_width_ms
        if start_bin >= end_bin:
             end_bin = start_bin + bin_width_ms

        bin_edges = np.arange(start_bin, end_bin + bin_width_ms, bin_width_ms)

        fig = plt.figure(figsize=(8, 5)) 
        
        counts, bins, patches = plt.hist(rr_list, bins=bin_edges, 
                                         color='#F0C86E', 
                                         edgecolor='black', 
                                         alpha=0.9) 
        
        title_str = "Histogram RR Intervals"
        if calculated_hti_value is not None:
             title_str += f" (HTI ≈ {calculated_hti_value:.2f})" 
        plt.title(title_str)
        plt.xlabel(f'RR Interval (ms) [Bin Width = {bin_width_ms:.1f} ms]')
        plt.ylabel('Frekuensi') 
        plt.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.6)
        plt.tight_layout()
        
        return fig

    def plot_tinn(self, rr_intervals_ms, calculated_tinn=None):
        rr_list = list(rr_intervals_ms)
        bin_width_ms = 8.0 

        min_rr = min(rr_list)
        max_rr = max(rr_list)
        
        N_plot = math.floor(min_rr / bin_width_ms) * bin_width_ms
        M_plot = math.ceil(max_rr / bin_width_ms) * bin_width_ms
        tinn_plot_val = M_plot - N_plot 

        if N_plot >= M_plot:
            M_plot = N_plot + bin_width_ms
            
        bin_edges = np.arange(N_plot, M_plot + bin_width_ms, bin_width_ms)

        fig = plt.figure(figsize=(10, 6))
        
        counts, bins, patches = plt.hist(rr_list, bins=bin_edges, alpha=0.7, 
                                         label='RR Interval Counts', color='lightblue', 
                                         edgecolor='black')
        
        modal_bin_index = np.argmax(counts)
        Y = counts[modal_bin_index]
        X = (bins[modal_bin_index] + bins[modal_bin_index+1]) / 2 

        plt.plot([N_plot, X, M_plot, N_plot], [0, Y, 0, 0], 'r--', linewidth=2, label=f'TINN Triangle (Base (X) = {tinn_plot_val:.1f} ms)')
        
        plt.text(N_plot, -Y*0.05, f'N={N_plot:.1f}', horizontalalignment='center', verticalalignment='top', color='black')
        plt.text(M_plot, -Y*0.05, f'M={M_plot:.1f}', horizontalalignment='center', verticalalignment='top', color='black')

        plt.title('TINN Visualization (Triangular Interpolation)')
        plt.xlabel(f'RR Interval (ms) [Bin Width ≈ {bin_width_ms:.2f} ms]')
        plt.ylabel('Number of RR Intervals (Count)')
        
        display_tinn = calculated_tinn if calculated_tinn is not None else (max_rr - min_rr)
        
        plt.text(0.95, 0.95, f'TINN (M-N):\n{display_tinn:.1f} ms', 
                 horizontalalignment='right', verticalalignment='top', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
        
        plt.text(0.95, 0.85, f'Peak (Y) = {int(Y)}', 
                 horizontalalignment='right', verticalalignment='top', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
        
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.ylim(bottom=-Y*0.1)
        plt.tight_layout()
        
        return fig
    
    def plot_rr_histogram(self, rr_intervals_ms, calculated_skewness=None):
        if rr_intervals_ms is None or len(rr_intervals_ms) < 5: 
            print("Cannot plot histogram: Not enough RR intervals.")
            return
        
        rr_list = list(rr_intervals_ms)
        mean_rr = self._mean(rr_list)
        max_rr = 0

        if rr_list: 
            max_rr = rr_list[0]
            for rr in rr_list[1:]:
                if rr > max_rr:
                    max_rr = rr

        x_axis_max = max_rr * 1.05 
        fig = plt.figure(figsize=(10, 6))

        num_bins = int(math.sqrt(len(rr_list))) 
        n, bins, patches = plt.hist(rr_list, bins=num_bins, density=True, alpha=0.7, label='RR Interval Distribution', color='skyblue', edgecolor='black')

        plt.axvline(mean_rr, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_rr:.2f} ms')
        plt.title('RR Interval Histogram')
        plt.xlabel('RR Interval (ms)')
        plt.ylabel('Frequency Density')

        if calculated_skewness is not None:
             plt.text(0.95, 0.95, f'Skewness: {calculated_skewness:.3f}', 
                      horizontalalignment='left', verticalalignment='top', 
                      transform=plt.gca().transAxes, 
                      bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
             
        plt.xlim(left=0, right=x_axis_max)
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        
        return fig

class frequencyDomainHRV:
    def __init__(self, rr_intervals_ms, interp_fs=4):
        self.rr_intervals_ms = rr_intervals_ms
        self.rr_intervals_s = rr_intervals_ms / 1000.0  
        self.interp_fs = interp_fs
        
        self.f = None
        self.Pxx = None
        self.parameters = {}

    def _interpolate_rr(self):
        time_cumulative_s = np.cumsum(self.rr_intervals_s)
        time_cumulative_s = time_cumulative_s - time_cumulative_s[0] 

        time_even = np.arange(0, time_cumulative_s[-1], 1.0 / self.interp_fs)
        
        spline_interpolator = Spline(time_cumulative_s, self.rr_intervals_s)
    
        rr_interpolated_list = []
        for t in time_even:
            val = spline_interpolator.calc(t)
            if val is not None:
                rr_interpolated_list.append(val)
            elif len(rr_interpolated_list) > 0:
                rr_interpolated_list.append(rr_interpolated_list[-1])
            else:
                rr_interpolated_list.append(0)

        rr_interpolated = np.array(rr_interpolated_list) 
        rr_detrended = rr_interpolated - np.polyval(np.polyfit(time_even, rr_interpolated, 1), time_even)
        
        return rr_detrended

    def _manual_welch_psd(self, signal, fs, nperseg, noverlap):
        n_signal = len(signal)
        nstep = nperseg - noverlap
        n_segments = (n_signal - noverlap) // nstep
        if n_segments <= 0:
            print("Warning: Signal too short for manual Welch segmentation. Using full signal.")
            nperseg = n_signal
            n_segments = 1
            noverlap = 0

        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))
        
        psd_segments = []
        
        for i in range(n_segments):
            start = i * nstep
            end = start + nperseg
            if end > n_signal:
                continue
                
            segment = signal[start:end]
            windowed_segment = segment * window
            fft_result = np.fft.fft(windowed_segment)
            fft_onesided = fft_result[:nperseg // 2 + 1]
            power_spectrum = np.abs(fft_onesided)**2
            s2 = np.sum(window**2)
            scale = 1.0 / (fs * s2)
            
            psd = power_spectrum * scale
            
            if nperseg % 2 == 0: 
                psd[1:-1] *= 2
            else: # Odd
                psd[1:] *= 2
                
            psd_segments.append(psd)
        
        if not psd_segments:
            print("Error: No PSD segments were generated.")
            return np.array([]), np.array([])
            
        mean_psd = np.mean(psd_segments, axis=0)
        frequencies = np.fft.fftfreq(nperseg, d=1/fs)[:nperseg // 2 + 1]
        
        return frequencies, mean_psd

    def calculate_welch_psd(self, nperseg=256, noverlap=128):     
        print("Interpolating RR intervals for PSD...")
        rr_interpolated = self._interpolate_rr() 
        n_signal = len(rr_interpolated)

        print(f"Calculating Welch's PSD method (Fs={self.interp_fs} Hz)...")
        
        if n_signal < nperseg:
            print(f"Warning: Signal (len={n_signal}) is shorter than nperseg ({nperseg}). Adjusting nperseg.")
            nperseg = n_signal
            noverlap = 0

        self.f, self.Pxx = self._manual_welch_psd(
            rr_interpolated, 
            fs=self.interp_fs, 
            nperseg=nperseg, 
            noverlap=noverlap
        )
        
        return self.f, self.Pxx
    
    def calculate_frequency_parameters(self):
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)

        df = self.f[1] - self.f[0]
        vlf_idx = np.where((self.f >= vlf_band[0]) & (self.f < vlf_band[1]))[0]
        lf_idx = np.where((self.f >= lf_band[0]) & (self.f < lf_band[1]))[0]
        hf_idx = np.where((self.f >= hf_band[0]) & (self.f < hf_band[1]))[0]

        vlf_power = trapezoid(self.Pxx[vlf_idx], self.f[vlf_idx]) if len(vlf_idx) > 0 else 0
        lf_power = trapezoid(self.Pxx[lf_idx], self.f[lf_idx]) if len(lf_idx) > 0 else 0
        hf_power = trapezoid(self.Pxx[hf_idx], self.f[hf_idx]) if len(hf_idx) > 0 else 0
        total_power = vlf_power + lf_power + hf_power

        vlf_power_ms2 = vlf_power * 1e6
        lf_power_ms2 = lf_power * 1e6
        hf_power_ms2 = hf_power * 1e6
        total_power_ms2 = total_power * 1e6

        lf_nu = (lf_power / (total_power - vlf_power)) * 100 if total_power != vlf_power else 0
        hf_nu = (hf_power / (total_power - vlf_power)) * 100 if total_power != vlf_power else 0

        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf

        lf_peak = self.f[lf_idx][np.argmax(self.Pxx[lf_idx])] if len(lf_idx) > 0 else None
        hf_peak = self.f[hf_idx][np.argmax(self.Pxx[hf_idx])] if len(hf_idx) > 0 else None

        self.parameters = {
        "Total Power (ms^2)": total_power,
        "VLF Power (ms^2)": vlf_power,
        "LF Power (ms^2)": lf_power,
        "HF Power (ms^2)": hf_power,
        "LF/HF Ratio": lf_hf_ratio,
        "LF (nu)": lf_nu,
        "HF (nu)": hf_nu,
        "LF Peak (Hz)": lf_peak,
        "HF Peak (Hz)": hf_peak
        }

        return self.parameters

class nonLinearHRV:
    def __init__(self):
        pass

    def calculate_poincare_metrics(self, time_domain_metrics):
        rmssd = time_domain_metrics.get('RMSSD (ms)')
        sdnn = time_domain_metrics.get('SDNN (ms)')
        
        if rmssd is None or sdnn is None:
            print("Error: RMSSD or SDNN not found in time_domain_metrics. Cannot calculate non-linear metrics.")
            return {}
        
        try:
            sd1 = math.sqrt(0.5) * rmssd 
            sd2_squared = (2 * (sdnn**2)) - (0.5 * (rmssd**2))
            sd2 = math.sqrt(sd2_squared)
            sd1_sd2_ratio = sd1 / sd2 if sd2 != 0 else 0.0

            return {
                'SD1 (ms)': sd1,
                'SD2 (ms)': sd2,
                'SD1/SD2 Ratio': sd1_sd2_ratio
            }
        except Exception as e:
            print(f"Error during non-linear calculation: {e}")
            return {}