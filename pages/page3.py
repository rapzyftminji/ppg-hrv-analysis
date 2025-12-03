import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math

from lib.hrv_calculation import hrvCalculation

st.set_page_config(layout="wide")
st.title("HRV Time-Domain Analysis")

required_keys = ['rr_intervals_ms', 'hrv_calc']
if not all(key in st.session_state for key in required_keys):
    st.error("Please run 'Cardiac Analysis' on the previous page (Page 2) to generate RR intervals first.")
    st.stop()

rr_intervals_ms = st.session_state['rr_intervals_ms']
hrv_calc = st.session_state['hrv_calc']
rr_list = list(rr_intervals_ms)

st.header("Time-Domain HRV Metrics")
with st.spinner("Calculating HRV metrics..."):
    hrv_metrics = hrv_calc.calculate_time_domain(rr_intervals_ms)
    st.session_state['hrv_metrics'] = hrv_metrics

if 'hrv_metrics' in st.session_state:
    
    hrv_metrics = st.session_state['hrv_metrics']
    
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean RR (ms)", f"{hrv_metrics['Mean RR (ms)']:.2f}")
    col2.metric("Mean HR (bpm)", f"{hrv_metrics['Mean HR (bpm)']:.2f}")
    col3.metric("SDNN (ms)", f"{hrv_metrics['SDNN (ms)']:.2f}")
    col4.metric("RMSSD (ms)", f"{hrv_metrics['RMSSD (ms)']:.2f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("pNN50 (%)", f"{hrv_metrics['pNN50 (%)']:.1f}")
    col2.metric("HRV Triangular Index", f"{hrv_metrics['HRV Triangular Index']:.2f}")
    col3.metric("TINN (ms)", f"{hrv_metrics['TINN (ms)']:.1f}")
    col4.metric("Skewness", f"{hrv_metrics['Skewness']:.3f}")
    
    col1, col2 = st.columns(2)
    col1.metric("CVNN", f"{hrv_metrics['CVNN']:.2f}")
    col2.metric("CVSD", f"{hrv_metrics['CVSD']:.2f}")

    with st.expander("Show All Calculated Metrics"):
        st.json(hrv_metrics)

    st.header("HRV Plots (Plotly)")
    df_rr = pd.DataFrame({"RR Interval (ms)": rr_list})

    st.subheader("HRV Triangular Index (HTI) Histogram")
    bin_width_ms = 8.0
    hti_val = hrv_metrics['HRV Triangular Index']

    fig_hti = px.histogram(df_rr, 
                           x="RR Interval (ms)", 
                           nbins=int((max(rr_list) - min(rr_list)) / bin_width_ms),
                           title=f"Histogram RR Intervals (HTI ≈ {hti_val:.2f})")

    fig_hti.update_layout(
        xaxis_title=f'RR Interval (ms) [Bin Width = {bin_width_ms:.1f} ms]',
        yaxis_title='Frequency (Count)',
        bargap=0.01
    )
    fig_hti.update_traces(marker_color='yellow', marker_line_color='black', marker_line_width=1)
    st.plotly_chart(fig_hti, use_container_width=True)

    st.subheader("TINN Visualization (Triangular Interpolation)")
    tinn_val = hrv_metrics['TINN (ms)']

    min_rr = min(rr_list)
    max_rr = max(rr_list)
    N_plot = math.floor(min_rr / bin_width_ms) * bin_width_ms
    M_plot = math.ceil(max_rr / bin_width_ms) * bin_width_ms
    if N_plot >= M_plot:
        M_plot = N_plot + bin_width_ms

    counts, bin_edges = np.histogram(rr_list, bins=int((M_plot - N_plot) / bin_width_ms), range=(N_plot, M_plot))
    Y = np.max(counts)
    X_index = np.argmax(counts)
    X = (bin_edges[X_index] + bin_edges[X_index+1]) / 2 

    fig_tinn = px.histogram(df_rr, 
                            x="RR Interval (ms)", 
                            nbins=int((M_plot - N_plot) / bin_width_ms), 
                            range_x=(N_plot, M_plot),
                            title="TINN Visualization")

    fig_tinn.add_trace(go.Scatter(
        x=[N_plot, X, M_plot, N_plot],
        y=[0, Y, 0, 0],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name=f'TINN Triangle (Base = {M_plot - N_plot:.1f} ms)'
    ))
    fig_tinn.add_annotation(x=N_plot, y=0, text=f"N={N_plot:.1f}", showarrow=False, yshift=-10)
    fig_tinn.add_annotation(x=M_plot, y=0, text=f"M={M_plot:.1f}", showarrow=False, yshift=-10)
    fig_tinn.add_annotation(
        x=X, y=Y, text=f"Peak (Y) = {Y}", bgcolor="wheat", opacity=0.8, ax=60, ay=-30
    )
    fig_tinn.add_annotation(
        x=M_plot, y=Y*0.9, text=f"TINN (M-N): {tinn_val:.1f} ms", showarrow=False, 
        bordercolor="black", borderwidth=1, bgcolor="wheat", opacity=0.8
    )
    fig_tinn.update_layout(
        xaxis_title=f'RR Interval (ms) [Bin Width ≈ {bin_width_ms:.1f} ms]',
        yaxis_title='Number of RR Intervals (Count)',
    )
    st.plotly_chart(fig_tinn, use_container_width=True)

    st.subheader("RR Interval Histogram (Density)")
    mean_rr = hrv_metrics['Mean RR (ms)']
    skewness = hrv_metrics['Skewness']
    num_bins = int(math.sqrt(len(rr_list)))

    fig_rr_hist = px.histogram(df_rr, 
                               x="RR Interval (ms)", 
                               nbins=num_bins,
                               histnorm='probability density',
                               title="RR Interval Histogram")

    fig_rr_hist.update_layout(shapes=[
        dict(type='line', y0=0, y1=1, xref='x', yref='paper', x0=mean_rr, x1=mean_rr, line=dict(color='red', dash='dash'))
    ])
    fig_rr_hist.add_annotation(x=mean_rr, y=0.95, yref='paper', text=f'Mean: {mean_rr:.2f} ms', showarrow=False, xshift=5)
                          
    fig_rr_hist.add_annotation(
        x=0.95, y=0.95, text=f'Skewness: {skewness:.3f}', showarrow=False, 
        xref="paper", yref="paper", bordercolor="black", borderwidth=1, bgcolor="wheat", opacity=0.8
    )

    fig_rr_hist.update_traces(marker_color='skyblue', marker_line_color='black', marker_line_width=1)
    fig_rr_hist.update_layout(
        xaxis_title='RR Interval (ms)',
        yaxis_title='Frequency Density'
    )
    st.plotly_chart(fig_rr_hist, use_container_width=True)

else:
    st.info("Click 'Calculate All Metrics' in the sidebar to see results.")
