import streamlit as st
import numpy as np
import io
import pandas as pd
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from lib.ppgProcessing import peakDetection, Spline, ppgSignalProcessing

st.set_page_config(layout="wide")
st.title("Empirical Mode Decomposition (EMD)")

def _emd_comperror(h, mean, pks, trs):
    if len(pks) == 0 or len(trs) == 0:
        return 1.0 
    samp_start = np.max((np.min(pks),np.min(trs)))
    samp_end = np.min((np.max(pks),np.max(trs))) + 1
    
    if samp_start >= samp_end:
        return 1.0 
        
    h_segment = h[samp_start:samp_end]
    mean_segment = mean[samp_start:samp_end]
    
    sum_mean_sq = np.sum(np.abs(mean_segment**2))
    sum_h_sq = np.sum(np.abs(h_segment**2))
    
    if sum_h_sq == 0:
        return 0.0 
        
    return sum_mean_sq / sum_h_sq

def _emd_complim(mean_t, pks, trs):
    if len(pks) == 0 or len(trs) == 0:
        return mean_t 
    
    samp_start = np.max((np.min(pks),np.min(trs)))
    samp_end = np.min((np.max(pks),np.max(trs))) + 1
    
    if samp_start >= samp_end or samp_end >= len(mean_t):
       return mean_t
       
    mean_t[:samp_start] = mean_t[samp_start]
    mean_t[samp_end:] = mean_t[samp_end]
    return mean_t

def interpolate_envelope(t, extrema_indices, extrema_values):
    spline = Spline(extrema_indices, extrema_values)
    
    interpolated_values = []
    for i in t:
        val = spline.calc(i)
        if val is None:
            if i < spline.x[0]:
                val = spline.y[0]  
            elif i > spline.x[-1]:
                val = spline.y[-1] 
            else:
                val = 0 
        interpolated_values.append(val)
        
    return np.array(interpolated_values)

def emd(x, nIMF = 3, stoplim = .001):
    r = x
    t = np.arange(len(r))
    imfs = np.zeros(nIMF,dtype=object)
    
    first_iteration_data = {} 
    
    for i in range(nIMF):
        r_t = r
        is_imf = False
        
        is_first_sift = True
        
        while is_imf == False:
            pks_list, trs_list = peakDetection._find_extreme_points(r_t, sampling_rate=1)
            pks = np.array(pks_list)
            trs = np.array(trs_list)
            
            if len(pks) < 2 or len(trs) < 2:
                if i == 0: 
                     st.warning(f"Stopping sifting for IMF 1: Not enough extrema found.")
                     first_iteration_data = {
                         'pks': pks, 'trs': trs, 'pks_t': np.zeros_like(t), 
                         'trs_t': np.zeros_like(t), 'mean_t': np.zeros_like(t)
                     }
                is_imf = True
                r_t = np.zeros_like(r_t) 
                continue
                
            pks_r = r_t[pks]
            pks_t = interpolate_envelope(t, pks, pks_r)
            
            trs_r = r_t[trs]
            trs_t = interpolate_envelope(t, trs, trs_r)
            
            mean_t = (pks_t + trs_t) / 2
            mean_t = _emd_complim(mean_t, pks, trs)
            
            if i == 0 and is_first_sift:
                first_iteration_data = {
                    'pks': pks,
                    'trs': trs,
                    'pks_t': pks_t,
                    'trs_t': trs_t,
                    'mean_t': mean_t
                }
                is_first_sift = False 
            
            sdk = _emd_comperror(r_t, mean_t, pks, trs)
            
            if sdk < stoplim:
                is_imf = True
            else:
                r_t = r_t - mean_t 
        
        imfs[i] = r_t
        r = r - imfs[i] 

        pks_res, trs_res = peakDetection._find_extreme_points(r, sampling_rate=1)
        if len(pks_res) < 2 or len(trs_res) < 2:
            st.info("Residue has no more extrema. Stopping decomposition.")
            break
            
    final_imfs = [imf for imf in imfs if isinstance(imf, np.ndarray)]
    
    return final_imfs, r, first_iteration_data

@st.cache_data
def load_signal_from_file(uploaded_file):
    signal_data = []
    try:
        uploaded_file.seek(0)
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        for line in stringio:
            try:
                signal_data.append(float(line.strip()))
            except ValueError:
                pass
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return np.array([])
    st.success(f"Successfully loaded {len(signal_data)} samples from {uploaded_file.name}.")
    return np.array(signal_data)

def find_best_imf_for_respiration(imfs_dict, fs, min_freq, max_freq):
    best_signal_key = None
    max_dominance_ratio = -1.0
    log_output = []

    log_output.append(f"Analyzing IMFs for Dominant Respiratory Signal ({min_freq}-{max_freq} Hz) ---")

    for key, signal in imfs_dict.items():
        if len(signal) > 1:
            n = len(signal)
            fft_vals = fft(signal)
            freqs = fftfreq(n, d=1/fs)[:n//2]
            power_spectrum = (np.abs(fft_vals)[:n//2])**2
            
            total_power = np.sum(power_spectrum)
            if total_power == 0:
                log_output.append(f"  {key}: Empty signal. Skipping.")
                continue

            freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
            power_in_band = 0
            if np.any(freq_mask):
                power_in_band = np.sum(power_spectrum[freq_mask])
            
            dominance_ratio = power_in_band / total_power
            
            log_output.append(f"  {key}: P_resp={power_in_band:.2f}, P_total={total_power:.2f} -> Dominance={dominance_ratio*100:.1f}%")

            if dominance_ratio > max_dominance_ratio:
                max_dominance_ratio = dominance_ratio
                best_signal_key = key
    
    return best_signal_key, log_output

def calculate_br_from_fft(signal, fs, min_freq_hz=0.15, max_freq_hz=0.40):
    n = len(signal)
    if n == 0:
        return 0, np.array([]), np.array([])

    window = np.hanning(n)
    signal_windowed = signal * window

    fft_vals = fft(signal_windowed)[:n//2]
    fft_freqs = fftfreq(n, d=1/fs)[:n//2]
    
    power_spectrum = np.abs(fft_vals)**2
    
    freq_mask = (fft_freqs >= min_freq_hz) & (fft_freqs <= max_freq_hz)
    
    if np.any(freq_mask) and np.sum(power_spectrum[freq_mask]) > 0:
        peak_power_index_in_mask = np.argmax(power_spectrum[freq_mask])
        peak_freq_hz = fft_freqs[freq_mask][peak_power_index_in_mask]
        peak_br_bpm = peak_freq_hz * 60.0
    else:
        peak_br_bpm = 0
    
    return peak_br_bpm, fft_freqs, power_spectrum

def plot_full_decomposition(imfs, residue, t, xaxis_title="Time / Sample Index"):
    n_imfs_found = len(imfs)
    num_plots = n_imfs_found + 1
    
    subplot_titles = [f"IMF {i+1}" for i in range(n_imfs_found)]
    subplot_titles.append("Residue (Trend)")

    fig = make_subplots(
        rows=num_plots, 
        cols=1, 
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.03
    )
    
    for i, imf in enumerate(imfs):
        fig.add_trace(go.Scatter(
            x=t, y=imf, name=f"IMF {i+1}", line=dict(color='green')
        ), row=i + 1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=residue, name="Residue", line=dict(color='blue')
    ), row=num_plots, col=1) 

    fig.update_layout(
        height=num_plots * 220,
        title_text="Full Decomposition Results",
        showlegend=False
    )
    fig.update_xaxes(title_text=xaxis_title, row=num_plots, col=1)
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload your data file (.txt)", 
    type="txt",
)

fs_input = None
if uploaded_file is not None:
    fs_input = st.sidebar.number_input(
        "Enter Sampling Rate (Fs) in Hz", 
        min_value=0.1, 
        value=100.0, 
        step=1.0,
    )

st.sidebar.header("Preprocessing (Savgol Filter)")
st.sidebar.info("Applied before EMD.")
savgol_order = 3
savgol_window = st.sidebar.number_input(
    "Filter Window Length", 
    min_value=5, 
    value=11, 
    step=2, 
)

if savgol_window % 2 == 0:
    savgol_window += 1
    st.sidebar.warning(f"Window length must be odd. Using {savgol_window}.")
if savgol_window <= savgol_order:
    savgol_window = savgol_order + 2
    st.sidebar.warning(f"Window length must be > order (3). Using {savgol_window}.")

st.sidebar.header("EMD Parameters")
n_imf_slider = st.sidebar.slider("Max Number of IMFs to Extract", 1, 10, 5)
stop_lim_slider = st.sidebar.number_input("Sifting Stop Limit (SD)", 0.0001, 0.1, 0.001, format="%.4f")

signal = None
t = None
fs = None 
xaxis_title = "Time / Sample Index"

if uploaded_file is not None:
    signal = load_signal_from_file(uploaded_file)
    if signal.size > 0:
        fs = fs_input  
        t = np.arange(len(signal)) / fs
        xaxis_title = "Time (s)"
        st.info(f"Using uploaded signal ({len(signal)} samples, Fs: {fs} Hz).")

if signal is not None and t is not None:
    filtered_signal = savgol_filter(signal, savgol_window, savgol_order)
    
    with st.spinner("Performing Manual EMD..."):
        imfs, residue, first_iter_data = emd(
            filtered_signal, n_imf_slider, stop_lim_slider
        )
    st.success("EMD processing complete.")
    
    imfs_dict = {f"imf_{i+1}": imf for i, imf in enumerate(imfs)}

    tab1, tab2, tab3 = st.tabs([
        "Sifting Process (First Iteration)", 
        "Full Decomposition",
        "Respiratory Analysis (from EMD)" 
    ])

    with tab1:
        st.header("Sifting Process: Finding the First IMF")
        
        st.subheader("1. Signal Preprocessing")
        fig_input = go.Figure()
        fig_input.add_trace(go.Scatter(
            x=t, y=signal, name="Original Signal", line=dict(color='grey', width=1.5)
        ))
        fig_input.add_trace(go.Scatter(
            x=t, y=filtered_signal, name=f"Filtered (Savgol, W={savgol_window}, O=3)", line=dict(color='red', width=1.5)
        ))
        fig_input.update_layout(
            title="Input Signal vs. Filtered Signal",
            xaxis_title=xaxis_title, yaxis_title="Amplitude", height=350,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_input, use_container_width=True)

        st.subheader("2. Extrema Detection")
        pks_indices = first_iter_data.get('pks', [])
        trs_indices = first_iter_data.get('trs', [])
        
        fig_extrema = go.Figure()
        fig_extrema.add_trace(go.Scatter(
            x=t, y=filtered_signal, name="Filtered Signal", line=dict(color='red')
        ))
        
        if len(pks_indices) > 0:
            fig_extrema.add_trace(go.Scatter(
                x=t[pks_indices], y=filtered_signal[pks_indices], 
                mode='markers', name="Maxima", marker=dict(color='green', size=8)
            ))
        
        if len(trs_indices) > 0:
                fig_extrema.add_trace(go.Scatter(
                x=t[trs_indices], y=filtered_signal[trs_indices], 
                mode='markers', name="Minima", marker=dict(color='purple', size=8)
            ))
        
        fig_extrema.update_layout(
            title="Detected Extrema on Filtered Signal",
            xaxis_title=xaxis_title, yaxis_title="Amplitude", height=350,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_extrema, use_container_width=True)

        st.subheader("3. Envelope Calculation")
        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(
            x=t, y=first_iter_data.get('pks_t'), 
            name="Maxima Envelope", line=dict(color='green', width=2)
        ))
        fig_env.add_trace(go.Scatter(
            x=t, y=first_iter_data.get('trs_t'), 
            name="Minima Envelope", line=dict(color='purple', width=2)
        ))
        fig_env.add_trace(go.Scatter(
            x=t, y=first_iter_data.get('mean_t'), 
            name="Mean Envelope", line=dict(color='blue', dash='dash', width=2.5)
        ))
        fig_env.add_trace(go.Scatter(
            x=t, y=filtered_signal, name="Filtered Signal", 
            line=dict(color='red', width=1.5, dash='dot')
        ))
        fig_env.update_layout(
            title="Envelopes and Mean (First Sift)",
            xaxis_title=xaxis_title, yaxis_title="Amplitude", height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_env, use_container_width=True)

    with tab2:
        st.header("Full Decomposition Results")
        if not imfs:
            st.warning("No IMFs were successfully extracted.")
        else:
            plot_full_decomposition(imfs, residue, t, xaxis_title=xaxis_title)
    
    with tab3:
        st.header("Respiratory Analysis from EMD IMFs")
        
        if fs is None:
            st.warning("Cannot perform respiratory analysis. Please enter a valid Sampling Rate (Fs) in the sidebar.")
        elif not imfs:
            st.warning("No IMFs were extracted. Cannot perform respiratory analysis.")
        else:
            respiratory_key, log = find_best_imf_for_respiration(
                imfs_dict, fs, 0.15, 0.40
            )
            
            if respiratory_key:
                st.success(f"---> {respiratory_key.upper().replace('_', ' ')} selected as the best RESPIRATORY signal (highest dominance).")
                respiratory_signal = imfs_dict[respiratory_key]
                
                average_br, f_fft, mag_fft = calculate_br_from_fft(
                    respiratory_signal, fs, min_freq_hz=0.15, max_freq_hz=0.40
                )
                
                if average_br > 0:
                    st.metric("Average Breathing Rate (FFT)", f"{average_br:.2f} breaths/min")
                else:
                    st.warning("Could not calculate breathing rate from FFT.")
        
                st.subheader(f"EMD Respiratory Signal ({respiratory_key.upper().replace('_', ' ')}) - Time Domain")
                df_resp = pd.DataFrame({'Time': t, 'Amplitude': respiratory_signal})
                fig_resp = px.line(df_resp, x='Time', y='Amplitude', title=f"EMD Respiratory Signal ({respiratory_key.upper().replace('_', ' ')})")
                fig_resp.update_xaxes(title_text=xaxis_title)
                st.plotly_chart(fig_resp, use_container_width=True)
                
                st.subheader("Respiratory FFT")
                df_fft = pd.DataFrame({"Frequency (Hz)": f_fft, "Magnitude": mag_fft})
                
                df_fft['Mag_Inside'] = df_fft['Magnitude'].where((df_fft["Frequency (Hz)"] >= 0.15) & (df_fft["Frequency (Hz)"] <= 0.40))
                df_fft['Mag_Outside'] = df_fft['Magnitude'].where((df_fft["Frequency (Hz)"] < 0.15) | (df_fft["Frequency (Hz)"] > 0.40))
                
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(x=df_fft["Frequency (Hz)"], y=df_fft['Mag_Outside'], mode='lines', line=dict(color='grey'), name='Outside Range'))
                fig_fft.add_trace(go.Scatter(x=df_fft["Frequency (Hz)"], y=df_fft['Mag_Inside'], mode='lines', line=dict(color='blue'), name='Search Range (0.15-0.40 Hz)'))
                
                if average_br > 0:
                    detected_freq_hz = average_br / 60
                    fig_fft.add_vline(x=detected_freq_hz, line_dash="dash", line_color="red", annotation_text=f"Detected: {average_br:.2f} bpm")
                
                fig_fft.add_vrect(x0=0.15, x1=0.40, fillcolor="green", opacity=0.1, line_width=0)
                fig_fft.update_layout(title="FFT of EMD Respiratory Signal", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", showlegend=True)
                st.plotly_chart(fig_fft, use_container_width=True)