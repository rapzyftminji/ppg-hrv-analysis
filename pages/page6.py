import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lib.hrv_calculation import nonLinearHRV 

st.set_page_config(layout="wide")
st.title("Non-Linear Method HRV Analysis Page")

required_keys = ['rr_intervals_ms', 'hrv_calc']
if not all(key in st.session_state for key in required_keys):
    st.error("Please run 'Cardiac Analysis' on the previous page (Page 2) to generate RR intervals first.")
    st.stop()

if 'hrv_metrics' not in st.session_state:
    st.error("Please run 'Calculate All Metrics' on 'Page 3 (HRV Time-Domain)' first.")
    st.info("Non-linear metrics (SD1, SD2) depend on SDNN and RMSSD from the time-domain analysis.")
    st.stop()

rr_intervals_ms = st.session_state['rr_intervals_ms']
rr_list = list(rr_intervals_ms)
time_domain_metrics = st.session_state['hrv_metrics']
# hrv_calc = st.session_state['hrv_calc']

with st.spinner("Calculating Non-Linear Metrics (SD1, SD2)..."):
    
    try:
        nonlinear_calc = nonLinearHRV()
        nonlinear_metrics = nonlinear_calc.calculate_poincare_metrics(time_domain_metrics)
        
        if not nonlinear_metrics:
            st.error("Could not calculate non-linear metrics. Check console for errors.")
        else:
            st.session_state['hrv_nonlinear_metrics'] = nonlinear_metrics
            st.success("Calculation of non-linear metrics complete.")
    except Exception as e:
        st.error(f"An error occurred during non-linear calculation: {e}")


if 'hrv_nonlinear_metrics' in st.session_state:
    metrics = st.session_state['hrv_nonlinear_metrics']
    mean_rr = time_domain_metrics['Mean RR (ms)']

    st.header("Poincaré Plot Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("SD1 (ms)", f"{metrics['SD1 (ms)']:.2f}", help="Short-term variability (Parasympathetic Activity) [cite: 1333, 1338]")
    col2.metric("SD2 (ms)", f"{metrics['SD2 (ms)']:.2f}", help="Long-term variability (Sympathetic + Parasympathetic) [cite: 1335, 1339]")
    col3.metric("SD1/SD2 Ratio", f"{metrics['SD1/SD2 Ratio']:.3f}", help="Ratio of short-term to long-term variability")

    rr_n = rr_list[:-1]
    rr_n_plus_1 = rr_list[1:]
    
    df_poincare = pd.DataFrame({
        "RRn (ms)": rr_n,
        "RRn+1 (ms)": rr_n_plus_1
    })

    st.header("Poincaré Plot")
    fig = go.Figure()

    min_val = min(min(rr_n), min(rr_n_plus_1)) * 0.95
    max_val = max(max(rr_n), max(rr_n_plus_1)) * 1.05
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(color='grey', dash='dash'),
        name='Identity Line (y=x)'
    ))

    fig.add_trace(go.Scatter(
        x=df_poincare["RRn (ms)"],
        y=df_poincare["RRn+1 (ms)"],
        mode='markers',
        marker=dict(color='blue', opacity=0.5),
        name='RR(n) vs RR(n+1)'
    ))

    sd1 = metrics['SD1 (ms)']
    sd2 = metrics['SD2 (ms)']

    t = np.linspace(0, 2 * np.pi, 100)

    ellipse_x_std = sd2 * np.cos(t) 
    ellipse_y_std = sd1 * np.sin(t)

    angle = np.pi / 4
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    ellipse_x_rot = (ellipse_x_std * cos_a) - (ellipse_y_std * sin_a)
    ellipse_y_rot = (ellipse_x_std * sin_a) + (ellipse_y_std * cos_a)
    
    ellipse_x_final = ellipse_x_rot + mean_rr
    ellipse_y_final = ellipse_y_rot + mean_rr
    
    fig.add_trace(go.Scatter(
        x=ellipse_x_final,
        y=ellipse_y_final,
        mode='lines',
        line=dict(color='red', width=2),
        name='SD1/SD2 Ellipse'
    ))

    fig.add_trace(go.Scatter(
        x=[mean_rr - sd1*cos_a, mean_rr + sd1*cos_a],
        y=[mean_rr + sd1*sin_a, mean_rr - sd1*sin_a],
        mode='lines',
        line=dict(color='red', dash='dot', width=1.5),
        name='SD1 Axis'
    ))
    fig.add_trace(go.Scatter(
        x=[mean_rr - sd2*cos_a, mean_rr + sd2*cos_a],
        y=[mean_rr - sd2*sin_a, mean_rr + sd2*sin_a],
        mode='lines',
        line=dict(color='red', dash='dot', width=1.5),
        name='SD2 Axis'
    ))
    
    fig.add_trace(go.Scatter(
        x=[mean_rr], y=[mean_rr],
        mode='markers',
        marker=dict(color='red', size=8, symbol='x'),
        name='Center (Mean RR)'
    ))
    
    fig.update_layout(
        title="Poincaré Plot (RRn vs RRn+1)",
        xaxis_title="RRn (ms)",
        yaxis_title="RRn+1 (ms)",
        xaxis=dict(scaleanchor="y", scaleratio=1), 
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=700,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click 'Run Non-Linear Analysis' in the sidebar to calculate and display the Poincaré plot results.")