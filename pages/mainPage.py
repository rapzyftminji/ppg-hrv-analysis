import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from lib.ppgProcessing import ppgSignalProcessing

@st.cache_data
def load_ppg_from_file(uploaded_file, fs, max_duration_sec=300):
    max_samples = int(max_duration_sec * fs)
    ppg_data = []
    try:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        for line in stringio:
            if len(ppg_data) >= max_samples:
                st.info(f"Stopped loading at {max_duration_sec} seconds ({max_samples} samples).")
                break
            ppg_data.append(float(line.strip()))
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return np.array([])
    
    st.success(f"Successfully loaded {len(ppg_data)} samples from {uploaded_file.name}.")
    return np.array(ppg_data)

st.set_page_config(layout="wide") 
st.title("DWT Decomposition Process and Breath Analysis")

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your PPG data file (.txt)", type="txt")

st.sidebar.header("Processing Parameters")
SAMPLING_RATE = st.sidebar.number_input("Sampling Rate (Hz)", value=50, min_value=1)
DOWNSAMPLE_FACTOR_RESP = st.sidebar.number_input("Downsample Factor (Respiratory)", value=2, min_value=1)

if uploaded_file is not None:
    my_ppg_data = load_ppg_from_file(uploaded_file, fs=SAMPLING_RATE)

    if my_ppg_data.size > 0:
        st.session_state['my_ppg_data'] = my_ppg_data
        st.session_state['sampling_rate'] = SAMPLING_RATE

        processor = ppgSignalProcessing(my_ppg_data, fs=SAMPLING_RATE)
        st.session_state['processor'] = processor
        
        with st.spinner("Normalizing and Pre-Processing signal..."):
            processor.normalize_signal()
            processor.preProcessing()
        
        tab_raw, tab_resp = st.tabs([
            "1. Raw & Pre-Processed", 
            "2. Respiratory Analysis"
        ])
        
        with tab_raw:
            st.header("Raw PPG Signal")
            time_axis_raw = np.arange(len(processor.ppg_signal)) / SAMPLING_RATE
            df_raw = pd.DataFrame({'Time (s)': time_axis_raw, 'Amplitude': processor.ppg_signal})
            fig_raw = px.line(df_raw, x='Time (s)', y='Amplitude', title="Raw PPG Signal (from file)")
            st.plotly_chart(fig_raw, use_container_width=True)

            st.header("Pre-Processing")
            time_axis_pre = np.arange(len(processor.pre_processed_signal)) / SAMPLING_RATE
            df_pre = pd.DataFrame({'Time (s)': time_axis_pre, 'Amplitude': processor.pre_processed_signal})
            fig_pre = px.line(df_pre, x='Time (s)', y='Amplitude', title="Pre-Processed Signal")
            st.plotly_chart(fig_pre, use_container_width=True)

        with tab_resp:
            st.header("Respiratory Signal Decomposition")
            with st.spinner(f"Decomposing for Respiratory (Downsample x{DOWNSAMPLE_FACTOR_RESP})..."):
                processor.dwtSignal(downsample_factor=DOWNSAMPLE_FACTOR_RESP)
            
            effective_fs_resp = SAMPLING_RATE / DOWNSAMPLE_FACTOR_RESP
            st.write(f"Effective Fs for Respiration: {effective_fs_resp:.2f} Hz.")

            # --- MODIFIED: Display all 8 Delay Constants ---
            st.subheader("Delay Constants")
            T1 = round(2**(1-1)) - 1
            T2 = round(2**(2-1)) - 1
            T3 = round(2**(3-1)) - 1
            T4 = round(2**(4-1)) - 1
            T5 = round(2**(5-1)) - 1
            T6 = round(2**(6-1)) - 1
            T7 = round(2**(7-1)) - 1
            T8 = round(2**(8-1)) - 1
            st.write(f"T1 = {T1}, T2 = {T2}, T3 = {T3}, T4 = {T4}, T5 = {T5}, T6 = {T6}, T7 = {T7}, T8 = {T8}")
            # --- End Modification ---

            st.subheader("Decomposed Signals (Respiratory Stage - Orde 1-8)")
            sorted_keys = sorted(processor.j_signals.keys(), key=lambda x: int(x.lstrip('j')))
            num_keys = len(sorted_keys)
            
            # --- MODIFIED: Changed rows back to 4 for 8 plots ---
            rows = 4 
            # --- End Modification ---

            if num_keys >= 8:
                left_col_keys = sorted_keys[0:rows]
                right_col_keys = sorted_keys[rows:rows*2]
                titles_in_plotly_order = [item for pair in zip(left_col_keys, right_col_keys) for item in pair]
            else:
                titles_in_plotly_order = sorted_keys

            fig_decomp = make_subplots(rows=rows, cols=2, subplot_titles=titles_in_plotly_order, shared_xaxes=True, vertical_spacing=0.05)
            
            for i, key in enumerate(sorted_keys):
                signal = processor.j_signals[key]
                # This logic correctly fills column 1, then column 2
                row, col = (i % rows) + 1, (i // rows) + 1 
                if signal is not None:
                    time_axis = np.arange(len(signal)) / effective_fs_resp
                    fig_decomp.add_trace(go.Scatter(x=time_axis, y=signal, name=key), row=row, col=col)
            
            # --- MODIFIED: Updated height for 4 rows ---
            fig_decomp.update_layout(height=800, title_text="Decomposed Signals (Respiratory Stage)", showlegend=False)
            # --- End Modification ---
            fig_decomp.update_xaxes(title_text="Time (s)", row=rows, col=1)
            fig_decomp.update_xaxes(title_text="Time (s)", row=rows, col=2)
            st.plotly_chart(fig_decomp, use_container_width=True)

            # --- Find Signals (this logic is unchanged) ---
            respiratory_key = processor.find_respiratory_signal(effective_fs_resp)
            vasomotor_key = processor.find_vasomotor_signal(effective_fs_resp)

            # --- Respiratory Section (this logic is unchanged) ---
            if respiratory_key:
                st.success(f"{respiratory_key.upper()} selected as the best RESPIRATORY signal.")
                
                resp_min_hz = 0.15
                resp_max_hz = 0.40
                peak_freq_hz_resp, f_fft_resp, mag_fft_resp = processor.calculate_peak_frequency_fft(
                    processor.respiratory_signal_auto, effective_fs_resp, 
                    min_freq_hz=resp_min_hz, max_freq_hz=resp_max_hz
                )
                average_br = peak_freq_hz_resp * 60.0
                
                if average_br > 0:
                    st.metric("Average Breathing Rate (FFT)", f"{average_br:.2f} breaths/min")
                else:
                    st.warning("Could not calculate breathing rate from FFT.")
                
                st.subheader(f"DWT Respiratory Signal ({respiratory_key.upper()}) - Time Domain")
                time_axis_resp = np.arange(len(processor.respiratory_signal_auto)) / effective_fs_resp
                df_resp = pd.DataFrame({'Time (s)': time_axis_resp, 'Amplitude': processor.respiratory_signal_auto})
                fig_resp = px.line(df_resp, x='Time (s)', y='Amplitude', title=f"DWT Respiratory Signal ({respiratory_key.upper()})")
                st.plotly_chart(fig_resp, use_container_width=True)
                
                st.subheader("Respiratory FFT")
                df_fft = pd.DataFrame({"Frequency (Hz)": f_fft_resp, "Magnitude": mag_fft_resp})
                
                df_fft['Mag_Inside'] = df_fft['Magnitude'].where((df_fft["Frequency (Hz)"] >= resp_min_hz) & (df_fft["Frequency (Hz)"] <= resp_max_hz))
                df_fft['Mag_Outside'] = df_fft['Magnitude'].where((df_fft["Frequency (Hz)"] < resp_min_hz) | (df_fft["Frequency (Hz)"] > resp_max_hz))
                
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(x=df_fft["Frequency (Hz)"], y=df_fft['Mag_Outside'], mode='lines', line=dict(color='grey'), name='Outside Range'))
                fig_fft.add_trace(go.Scatter(x=df_fft["Frequency (Hz)"], y=df_fft['Mag_Inside'], mode='lines', line=dict(color='blue'), name=f'Search Range ({resp_min_hz}-{resp_max_hz} Hz)'))
                
                if average_br > 0:
                    fig_fft.add_vline(x=peak_freq_hz_resp, line_dash="dash", line_color="red", annotation_text=f"Detected: {average_br:.2f} bpm")
                
                fig_fft.add_vrect(x0=resp_min_hz, x1=resp_max_hz, fillcolor="green", opacity=0.1, line_width=0)
                fig_fft.update_layout(title="FFT of Respiratory Signal", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", showlegend=True)
                st.plotly_chart(fig_fft, use_container_width=True)

            # --- Vasomotor Section (this logic is unchanged) ---
            if vasomotor_key:
                st.info(f"{vasomotor_key.upper()} selected as the best VASOMOTOR signal.")

                vaso_min_hz = 0.04
                vaso_max_hz = 0.15
                
                peak_freq_hz_vaso, f_fft_vaso, mag_fft_vaso = processor.calculate_peak_frequency_fft(
                    processor.vasomotor_signal_auto, effective_fs_resp, 
                    min_freq_hz=vaso_min_hz, max_freq_hz=vaso_max_hz
                )
                
                if peak_freq_hz_vaso > 0:
                    st.metric("Peak Vasomotor Frequency (FFT)", f"{peak_freq_hz_vaso:.4f} Hz")
                else:
                    st.warning("Could not calculate peak vasomotor frequency from FFT.")

                time_axis_vaso = np.arange(len(processor.vasomotor_signal_auto)) / effective_fs_resp
                df_vaso = pd.DataFrame({'Time (s)': time_axis_vaso, 'Amplitude': processor.vasomotor_signal_auto})
                fig_vaso = px.line(df_vaso, x='Time (s)', y='Amplitude', title=f"Vasomotor Signal ({vasomotor_key.upper()})")
                st.plotly_chart(fig_vaso, use_container_width=True)
                
                st.subheader("Vasomotor FFT")
                df_fft_vaso = pd.DataFrame({"Frequency (Hz)": f_fft_vaso, "Magnitude": mag_fft_vaso})
                
                df_fft_vaso['Mag_Inside'] = df_fft_vaso['Magnitude'].where((df_fft_vaso["Frequency (Hz)"] >= vaso_min_hz) & (df_fft_vaso["Frequency (Hz)"] <= vaso_max_hz))
                df_fft_vaso['Mag_Outside'] = df_fft_vaso['Magnitude'].where((df_fft_vaso["Frequency (Hz)"] < vaso_min_hz) | (df_fft_vaso["Frequency (Hz)"] > vaso_max_hz))
                
                fig_fft_vaso = go.Figure()
                fig_fft_vaso.add_trace(go.Scatter(x=df_fft_vaso["Frequency (Hz)"], y=df_fft_vaso['Mag_Outside'], mode='lines', line=dict(color='grey'), name='Outside Range'))
                fig_fft_vaso.add_trace(go.Scatter(x=df_fft_vaso["Frequency (Hz)"], y=df_fft_vaso['Mag_Inside'], mode='lines', line=dict(color='blue'), name=f'Search Range ({vaso_min_hz}-{vaso_max_hz} Hz)'))
                
                if peak_freq_hz_vaso > 0:
                    fig_fft_vaso.add_vline(x=peak_freq_hz_vaso, line_dash="dash", line_color="red", 
                                          annotation_text=f"Detected: {peak_freq_hz_vaso:.4f} Hz")
                
                fig_fft_vaso.add_vrect(x0=vaso_min_hz, x1=vaso_max_hz, fillcolor="green", opacity=0.1, line_width=0)
                fig_fft_vaso.update_layout(title="FFT of Vasomotor Signal", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", showlegend=True)
                st.plotly_chart(fig_fft_vaso, use_container_width=True)