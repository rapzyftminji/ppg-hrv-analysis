import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("DWT Filter Performance Analysis")

required_keys = ['processor', 'sampling_rate']
if not all(key in st.session_state for key in required_keys):
    st.error("Please upload and process a PPG signal on the 'Main Page' first.")
    st.stop()

processor = st.session_state['processor']
sampling_rate = st.session_state['sampling_rate']
downsample_factor = processor.last_downsample_factor
effective_fs = sampling_rate / downsample_factor

tab_freq, tab_impulse, tab_decomp = st.tabs([
    "1. Frequency Response",
    "2. Impulse Response",
    "3. Decomposition Results (Signal)"
])

def dirac(x):
    return 1 if x == 0 else 0

@st.cache_data
def calculate_base_filters_freq_response(fs_samples):  
    k_indices = [-2, -1, 0, 1]
    h = {}
    g = {}
    
    for n in k_indices:
        h[n] = 1/8 * (dirac(n-1) + 3*dirac(n) + 3*dirac(n+1) + dirac(n+2))
        g[n] = -2 * (dirac(n) - dirac(n+1))
        

    fs_const = fs_samples 
    Hw = np.zeros(fs_const + 1, dtype=float)
    Gw = np.zeros(fs_const + 1, dtype=float)
    
    for i in range(fs_const + 1):
        reG, imG, reH, imH = 0.0, 0.0, 0.0, 0.0
        for k in k_indices:
            angle = k * 2 * np.pi * i / fs_const
            
            reG += g[k] * np.cos(angle)
            imG -= g[k] * np.sin(angle)
            
            reH += h[k] * np.cos(angle)
            imH -= h[k] * np.sin(angle)
        
        Hw[i] = np.sqrt(reH**2 + imH**2)
        Gw[i] = np.sqrt(reG**2 + imG**2)
        
    return Hw, Gw, fs_const

@st.cache_data
def build_q_filters(Hw, Gw, fs_const):
    Q = {} 
    i_list = np.arange(0, (fs_const // 2) + 1)
    
    for j in range(1, 9):
        Q[j] = np.zeros_like(i_list, dtype=float)
        
    max_hw_gw_index = len(Hw) - 1

    for i in i_list:
        idx_1 = i
        if idx_1 <= max_hw_gw_index:
            Q[1][i] = Gw[idx_1]
        
        idx_2 = 2 * i
        if idx_2 <= max_hw_gw_index and idx_1 <= max_hw_gw_index:
            Q[2][i] = Gw[idx_2] * Hw[idx_1]
        
        idx_4 = 4 * i
        if idx_4 <= max_hw_gw_index and idx_2 <= max_hw_gw_index and idx_1 <= max_hw_gw_index:
            Q[3][i] = Gw[idx_4] * Hw[idx_2] * Hw[idx_1]
            
        idx_8 = 8 * i
        if idx_8 <= max_hw_gw_index and idx_4 <= max_hw_gw_index and idx_2 <= max_hw_gw_index and idx_1 <= max_hw_gw_index:
            Q[4][i] = Gw[idx_8] * Hw[idx_4] * Hw[idx_2] * Hw[idx_1]

        idx_16 = 16 * i
        if idx_16 <= max_hw_gw_index and idx_8 <= max_hw_gw_index and idx_4 <= max_hw_gw_index and idx_2 <= max_hw_gw_index and idx_1 <= max_hw_gw_index:
            Q[5][i] = Gw[idx_16] * Hw[idx_8] * Hw[idx_4] * Hw[idx_2] * Hw[idx_1]

        idx_32 = 32 * i
        if idx_32 <= max_hw_gw_index and idx_16 <= max_hw_gw_index and idx_8 <= max_hw_gw_index and idx_4 <= max_hw_gw_index and idx_2 <= max_hw_gw_index and idx_1 <= max_hw_gw_index:
            Q[6][i] = Gw[idx_32] * Hw[idx_16] * Hw[idx_8] * Hw[idx_4] * Hw[idx_2] * Hw[idx_1]

        idx_64 = 64 * i
        if idx_64 <= max_hw_gw_index and idx_32 <= max_hw_gw_index and idx_16 <= max_hw_gw_index and idx_8 <= max_hw_gw_index and idx_4 <= max_hw_gw_index and idx_2 <= max_hw_gw_index and idx_1 <= max_hw_gw_index:
            Q[7][i] = Gw[idx_64] * Hw[idx_32] * Hw[idx_16] * Hw[idx_8] * Hw[idx_4] * Hw[idx_2] * Hw[idx_1]

        idx_128 = 128 * i
        if idx_128 <= max_hw_gw_index and idx_64 <= max_hw_gw_index and idx_32 <= max_hw_gw_index and idx_16 <= max_hw_gw_index and idx_8 <= max_hw_gw_index and idx_4 <= max_hw_gw_index and idx_2 <= max_hw_gw_index and idx_1 <= max_hw_gw_index:
            Q[8][i] = Gw[idx_128] * Hw[idx_64] * Hw[idx_32] * Hw[idx_16] * Hw[idx_8] * Hw[idx_4] * Hw[idx_2] * Hw[idx_1]
    return i_list, Q

with tab_freq:
    st.header("1. Frequency Response")
    with st.spinner("Calculating base filters 'Hw' and 'Gw' from scratch..."):
        Hw, Gw, fs_const = calculate_base_filters_freq_response(fs_samples=1024)
    with st.spinner("Building Q[1]-Q[8] from base filters..."):
        i_list, Q = build_q_filters(Hw, Gw, fs_const)
    
    fig_freq = go.Figure()
    
    x_axis_hz = i_list * (effective_fs / fs_const)
    
    for j in range(1, 9):
        fig_freq.add_trace(go.Scatter(
            x=x_axis_hz, 
            y=Q[j], 
            name=f"Q{j}"
        ))
    
    fig_freq.update_layout(
        title="Frequency Response of DWT Filter Bank",
        xaxis_title=f"Frequency (Hz) - Scaled to {effective_fs:.2f} Hz Nyquist",
        yaxis_title="Magnitude",
        legend_title="Filter",
        xaxis_range=[0, effective_fs / 2] 
    )
    st.plotly_chart(fig_freq, use_container_width=True)

with tab_impulse:
    st.header("2. Impulse Response")
    
    q_coeffs = processor.q_coeffs
    sorted_keys = sorted(q_coeffs.keys(), key=lambda x: int(x.lstrip('q')))
    
    fig_impulse = make_subplots(
        rows=4, 
        cols=2, 
        subplot_titles=[k.upper() for k in sorted_keys],
        vertical_spacing=0.1
    )

    for i, key in enumerate(sorted_keys):
        coeffs = q_coeffs[key]

        if len(coeffs) > 0:
            a = - (len(coeffs) // 2)
            k_list = np.arange(a, a + len(coeffs))
        else:
            k_list = []
        
        row, col = (i % 4) + 1, (i // 4) + 1
        
        fig_impulse.add_trace(
            go.Bar(x=k_list, y=coeffs, name=key),
            row=row, col=col
        )
    
    fig_impulse.update_layout(
        height=900, 
        title_text="DWT Filter Impulse Responses (q1-q8)", 
        showlegend=False
    )
    st.plotly_chart(fig_impulse, use_container_width=True)


with tab_decomp:
    st.header("3. Decomposition Results (Signal)")
    
    j_signals = processor.j_signals
    sorted_keys = sorted(j_signals.keys(), key=lambda x: int(x.lstrip('j')))
    
    if not sorted_keys:
        st.warning("No decomposed signals found. Please re-run processing on the main page.")
    else:
        num_keys = len(sorted_keys)
        rows = 4
        if num_keys >= 8:
            left_col_keys = sorted_keys[0:rows]
            right_col_keys = sorted_keys[rows:rows*2]
            titles_in_plotly_order = [item for pair in zip(left_col_keys, right_col_keys) for item in pair]
        else:
            titles_in_plotly_order = sorted_keys

        fig_decomp = make_subplots(
            rows=rows, 
            cols=2, 
            subplot_titles=[k.upper() for k in titles_in_plotly_order], 
            shared_xaxes=True, 
            vertical_spacing=0.05
        )
        
        for i, key in enumerate(sorted_keys):
            signal = j_signals[key]
            row, col = (i % rows) + 1, (i // rows) + 1 
            if signal is not None:
                time_axis = np.arange(len(signal)) / effective_fs
                fig_decomp.add_trace(go.Scatter(x=time_axis, y=signal, name=key), row=row, col=col)
        
        fig_decomp.update_layout(height=800, title_text="Decomposed Signals (j-signals)", showlegend=False)
        fig_decomp.update_xaxes(title_text="Time (s)", row=rows, col=1)
        fig_decomp.update_xaxes(title_text="Time (s)", row=rows, col=2)
        st.plotly_chart(fig_decomp, use_container_width=True)
