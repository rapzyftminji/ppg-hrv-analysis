import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.ppgProcessing import ppgSignalProcessing, peakDetection
from lib.hrv_calculation import rrInterval, hrvCalculation

st.set_page_config(layout="wide")
st.title("Cardiac Analysis & HRV Tachograms")

required_keys = ['processor', 'sampling_rate']
if not all(key in st.session_state for key in required_keys):
    st.error("Please upload and process a PPG signal on the 'Main Page' first.")
    st.stop()

processor = st.session_state['processor']
SAMPLING_RATE = st.session_state['sampling_rate']

with st.sidebar.form(key='cardiac_form'):
    st.header("Cardiac Parameters")
    DOWNSAMPLE_FACTOR_CARD = st.number_input(
        "Downsample Factor (Cardiac)", 
        value=1, min_value=1
    )

    st.header("RR Peak Detection")
    rr_params = {
        'min_peak_dist_ms': st.number_input(
            "Min Peak Dist (ms)", value=450.0
        ),
        'initial_th_multiplier': st.number_input(
            "Initial Threshold Mult", value=0.1, format="%.3f"
        ),
        'threshold_m_factor': st.number_input(
            "Threshold M Factor", value=0.5, format="%.2f"
        )
    }
    
    run_analysis_button = st.form_submit_button("Run Cardiac Analysis")

if run_analysis_button:
    
    st.header("Cardiac Signal Decomposition")
    with st.spinner(f"Re-decomposing for Cardiac (Downsample x{DOWNSAMPLE_FACTOR_CARD})..."):
        processor.dwtSignal(downsample_factor=DOWNSAMPLE_FACTOR_CARD)
        effective_fs_card = SAMPLING_RATE / DOWNSAMPLE_FACTOR_CARD
        cardiac_key = processor.find_cardiac_signal(effective_fs_card)

        if not (cardiac_key and processor.cardiac_signal_auto is not None):
            st.error("Could not find a suitable DWT cardiac signal.")
            st.stop() 
        
        st.success(f"---> {cardiac_key.upper()} selected as the best CARDIAC signal.")
        cardiac_delay_samples = processor.cardiac_signal_delay_samples
        total_delay_s = cardiac_delay_samples / effective_fs_card
    
    with st.spinner("Detecting RR intervals..."):
        rr_peaks, maxima_rr, minima_rr, rr_intervals_ms = rrInterval(
            processor.cardiac_signal_auto,
            SAMPLING_RATE,
            effective_fs_card,
            **rr_params
        )
    st.success("Peak detection complete.")
    
    st.session_state['cardiac_results'] = {
        'cardiac_signal': processor.cardiac_signal_auto,
        'cardiac_key': cardiac_key,
        'effective_fs_card': effective_fs_card,
        'total_delay_s': total_delay_s,
        'rr_peaks': rr_peaks,
        'maxima_rr': maxima_rr,
        'minima_rr': minima_rr,
        'rr_intervals_ms': rr_intervals_ms
    }
    st.session_state['hrv_calc'] = hrvCalculation()
    st.session_state['rr_intervals_ms'] = rr_intervals_ms 

if 'cardiac_results' in st.session_state:
    results = st.session_state['cardiac_results']
    rr_intervals_ms = results['rr_intervals_ms']
    
    if rr_intervals_ms.size == 0:
        st.warning("No RR intervals found. Cannot plot tachograms.")
        st.stop()

    hrv_calc = st.session_state['hrv_calc']
    mean_rr_ms = hrv_calc._mean(list(rr_intervals_ms))
    mean_hr_bpm = 60000.0 / mean_rr_ms if mean_rr_ms > 0 else 0
    
    col1, col2 = st.columns(2)
    col1.metric("Total Detected Peaks", f"{len(results['rr_peaks'])} peaks")
    col2.metric("Mean Heart Rate (Nominal BPM)", f"{mean_hr_bpm:.2f} bpm")
    
    st.header("Pre-Processed vs. DWT Cardiac Signal (Aligned)")
    
    time_axis_pre = np.arange(len(processor.pre_processed_signal)) / SAMPLING_RATE
    delay_correction_s = results['total_delay_s'] 
    
    time_axis_card = (np.arange(len(results['cardiac_signal'])) / results['effective_fs_card']) + delay_correction_s

    fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Pre-Processed Signal (Input to DWT)", 
                                             f"DWT Cardiac Signal ({results['cardiac_key'].upper()}) & Peak Detection (Aligned)"))
    
    fig_comp.add_trace(go.Scatter(x=time_axis_pre, y=processor.pre_processed_signal, 
                                  name='Pre-Processed', line=dict(color='grey')), row=1, col=1)
    
    fig_comp.add_trace(go.Scatter(x=time_axis_card, y=results['cardiac_signal'], 
                                  name=f'DWT Cardiac', line=dict(color='black')), row=2, col=1)
    
    fig_comp.add_trace(go.Scatter(x=time_axis_card[results['maxima_rr']], y=results['cardiac_signal'][results['maxima_rr']], 
                                  name='Local Maxima', mode='markers', marker=dict(color='dodgerblue', symbol='triangle-up', size=8)), row=2, col=1)
    
    fig_comp.add_trace(go.Scatter(x=time_axis_card[results['minima_rr']], y=results['cardiac_signal'][results['minima_rr']], 
                                  name='Local Minima', mode='markers', marker=dict(color='limegreen', symbol='triangle-down', size=8)), row=2, col=1)
    
    fig_comp.add_trace(go.Scatter(x=time_axis_card[results['rr_peaks']], y=results['cardiac_signal'][results['rr_peaks']], 
                                  name='Detected Peaks', mode='markers', marker=dict(color='red', symbol='x', size=10, line_width=2)), row=2, col=1)

    fig_comp.update_layout(height=700, legend_tracegroupgap=180)
    fig_comp.update_xaxes(title_text="Time (s)", row=2, col=1)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.header("HRV Tachograms (Plotly)")
    rr_list = list(rr_intervals_ms)
    beat_numbers = np.arange(len(rr_list))
    
    st.subheader("RR Interval Tachogram")
    mean_rr = hrv_calc._mean(rr_list)
    median_rr = hrv_calc._median(rr_list)
    df_rr = pd.DataFrame({"Beat Number": beat_numbers, "RR Interval (ms)": rr_list})
    fig_tacho = px.line(df_rr, x="Beat Number", y="RR Interval (ms)", title="RR Tachogram (Beat-to-Beat Intervals)", markers=True)

    fig_tacho.update_layout(
        shapes=[
            dict(type='line', y0=mean_rr, y1=mean_rr, x0=0, x1=1, xref='paper', yref='y', line=dict(color='red', dash='dash')),
            dict(type='line', y0=median_rr, y1=median_rr, x0=0, x1=1, xref='paper', yref='y', line=dict(color='orange', dash='dash'))
        ],
        annotations=[
            dict(x=0.95, y=mean_rr, xref='paper', yref='y', text=f'Mean RR: {mean_rr:.2f} ms', showarrow=False, yshift=5, bgcolor="white"),
            dict(x=0.95, y=median_rr, xref='paper', yref='y', text=f'Median RR: {median_rr:.2f} ms', showarrow=False, yshift=5, bgcolor="white")
        ]
    )
    st.plotly_chart(fig_tacho, use_container_width=True)
    
    st.subheader("Heart Rate Tachogram")
    hr_bpm = 60000.0 / rr_intervals_ms
    hr_list = list(hr_bpm)
    mean_hr = hrv_calc._mean(hr_list)
    median_hr = hrv_calc._median(hr_list)
    std_hr = hrv_calc._std(hr_list, mean_val=mean_hr)
    df_hr = pd.DataFrame({"Beat Number": beat_numbers, "Heart Rate (BPM)": hr_list})
    
    fig_hr_tacho = px.line(df_hr, x="Beat Number", y="Heart Rate (BPM)", title="HRV Tachogram", markers=True)

    fig_hr_tacho.update_layout(
        shapes=[
            dict(type='line', y0=mean_hr, y1=mean_hr, x0=0, x1=1, xref='paper', yref='y', line=dict(color='red', dash='solid')),
            dict(type='line', y0=median_hr, y1=median_hr, x0=0, x1=1, xref='paper', yref='y', line=dict(color='green', dash='dot')),
            dict(type='line', y0=mean_hr + std_hr, y1=mean_hr + std_hr, x0=0, x1=1, xref='paper', yref='y', line=dict(color='red', dash='dash')),
            dict(type='line', y0=mean_hr - std_hr, y1=mean_hr - std_hr, x0=0, x1=1, xref='paper', yref='y', line=dict(color='red', dash='dash'))
        ],
        annotations=[
            dict(x=0.95, y=mean_hr, xref='paper', yref='y', text=f'Mean HR: {mean_hr:.2f} BPM', showarrow=False, yshift=5, bgcolor="#E0E0E0"),
            dict(x=0.95, y=median_hr, xref='paper', yref='y', text=f'Median HR: {median_hr:.2f} BPM', showarrow=False, yshift=5, bgcolor="#E0E0E0"),
            dict(x=0.95, y=mean_hr + std_hr, xref='paper', yref='y', text=f'+1 SD: {mean_hr + std_hr:.2f} BPM', showarrow=False, yshift=5, bgcolor="#E0E0E0"),
            dict(x=0.95, y=mean_hr - std_hr, xref='paper', yref='y', text=f'-1 SD: {mean_hr - std_hr:.2f} BPM', showarrow=False, yshift=-5, bgcolor="#E0E0E0")
        ],
        plot_bgcolor='#E0E0E0'
    )
    
    fig_hr_tacho.update_yaxes(range=[40, 140], gridcolor='white')
    fig_hr_tacho.update_xaxes(gridcolor='white')
    st.plotly_chart(fig_hr_tacho, use_container_width=True)
