import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lib.hrv_calculation import hrvCalculation, frequencyDomainHRV

st.set_page_config(layout="wide")
st.title("Frequency Domain Analysis (Welch's PSD)")

if 'rr_intervals_ms' not in st.session_state or len(st.session_state['rr_intervals_ms']) == 0:
    st.error("RR Interval data not found.")
    st.warning("Please return to the 'TCardiac Analysis' page and calculate RR Intervals first.")
else:
    rr_intervals = st.session_state['rr_intervals_ms']
    
    st.info(f"Found {len(rr_intervals)} RR intervals from session state. Starting analysis...")
    
    try:
        freq_analyzer = frequencyDomainHRV(rr_intervals_ms=rr_intervals, interp_fs=4)

        with st.spinner("Calculating Power Spectral Density (PSD) with Welch's method..."):
            f, Pxx = freq_analyzer.calculate_welch_psd(nperseg=256, noverlap=128)

        if f is None or Pxx is None or len(f) == 0:
            st.error("Failed to calculate PSD. The signal might be too short or another error occurred.")
        else:
            
            with st.spinner("Calculating frequency parameters..."):
                parameters = freq_analyzer.calculate_frequency_parameters()
            
            st.header("Key Metrics") 

            param_list = list(parameters.items())
            num_params = len(param_list)
            
            num_cols = 4 
            
            num_rows = (num_params + num_cols - 1) // num_cols

            for row in range(num_rows):
                cols = st.columns(num_cols)
                row_params = param_list[row * num_cols : (row + 1) * num_cols]
                
                for i, (param, value) in enumerate(row_params):
                    with cols[i]:
                        if isinstance(value, (float, np.float64)):
                            formatted_value = f"{value:.6f}" 
                        elif isinstance(value, (int, np.int64)):
                            formatted_value = f"{value:,}" 
                        else:
                            formatted_value = str(value)
                            
                        st.metric(label=param, value=formatted_value)

            st.header("Welch's Power Spectral Density (PSD) Plot")

            df_psd = pd.DataFrame({'Frequency (Hz)': f, 'Power Density (s²/Hz)': Pxx})

            fig_psd = go.Figure()

            fig_psd.add_trace(go.Scatter(
                x=df_psd['Frequency (Hz)'], 
                y=df_psd['Power Density (s²/Hz)'],
                mode='lines',
                name='HRV Power Spectral Density (Welch\'s Method)',
                line=dict(color='black', width=2),
                fill='tozeroy' 
            ))
            
            fig_psd.add_vrect(
                x0=0.0001, x1=0.003,
                fillcolor="purple", opacity=0.1,
                line_width=0, annotation_text="ULF", annotation_position="top left"
            )
            
            fig_psd.add_vrect(
                x0=0.003, x1=0.04,
                fillcolor="blue", opacity=0.2,
                line_width=0, annotation_text="VLF", annotation_position="top left"
            )
            
            fig_psd.add_vrect(
                x0=0.04, x1=0.15,
                fillcolor="green", opacity=0.2,
                line_width=0, annotation_text="LF", annotation_position="top left"
            )
            
            fig_psd.add_vrect(
                x0=0.15, x1=0.4,
                fillcolor="orange", opacity=0.2,
                line_width=0, annotation_text="HF", annotation_position="top left"
            )
            
            max_power = df_psd['Power Density (s²/Hz)'].max()
            
            fig_psd.update_layout(
                title="HRV Power Spectral Density (Welch's Method)",
                xaxis_title="Frequency (Hz)",            
                yaxis_title="Power Density (s²/Hz)",      
                xaxis_type="linear",                    
                yaxis_type="linear",                   
                hovermode="x unified",
                xaxis_range=[0, 0.5],                   
                yaxis_range=[0, max_power * 1.1]         
            )
            
            fig_psd.update_yaxes(zeroline=False, zerolinewidth=0, zerolinecolor='rgba(0,0,0,0)')
            fig_psd.update_xaxes(zeroline=False, zerolinewidth=0, zerolinecolor='rgba(0,0,0,0)')
            
            st.plotly_chart(fig_psd, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during frequency analysis: {e}")
        st.exception(e)