# PPG-HRV Analysis Tool

A comprehensive Streamlit web application for analyzing Photoplethysmography (PPG) signals and calculating Heart Rate Variability (HRV) metrics using Discrete Wavelet Transform (DWT) decomposition.

## Overview

This tool provides a complete pipeline for PPG signal processing and HRV analysis, featuring multiple analysis methods including time-domain, frequency-domain, and non-linear HRV metrics. The application uses custom DWT implementation with quadrature mirror filters for signal decomposition.

## Features

### Signal Processing
- **PPG Signal Normalization** - Standardizes input signals for consistent analysis
- **DWT Decomposition** - Multi-level wavelet decomposition to separate signal components:
  - Respiratory signals (breathing patterns)
  - Vasomotor signals (blood vessel tone variations)
  - Cardiac signals (heart activity)
- **Adaptive Peak Detection** - Robust systolic peak detection with customizable parameters
- **Signal Downsampling** - Reduces computational load while preserving signal characteristics

### Analysis Modules

1. **DWT Decomposition Process and Breath Analysis**
   - Interactive visualization of wavelet decomposition levels
   - Respiratory rate calculation from extracted breathing patterns
   - Real-time signal processing controls

2. **DWT Performance Analysis**
   - Evaluation of decomposition quality
   - Signal-to-noise ratio analysis
   - Reconstruction error metrics

3. **Cardiac Analysis**
   - Heart rate calculation
   - RR interval extraction
   - Beat-to-beat analysis

4. **HRV Time-Domain Analysis**
   - SDNN (Standard Deviation of NN intervals)
   - RMSSD (Root Mean Square of Successive Differences)
   - pNN50 (Percentage of successive NN intervals differing by > 50ms)
   - Additional time-domain metrics

5. **HRV Frequency-Domain Analysis (Welch's PSD)**
   - Power Spectral Density estimation
   - VLF (Very Low Frequency) power
   - LF (Low Frequency) power
   - HF (High Frequency) power
   - LF/HF ratio calculation

6. **Non-Linear HRV Analysis**
   - Poincaré plot analysis (SD1, SD2)
   - Approximate Entropy (ApEn)
   - Sample Entropy (SampEn)
   - Detrended Fluctuation Analysis (DFA)

7. **Manual EMD with Custom Peak Detection**
   - Empirical Mode Decomposition implementation
   - Custom spline interpolation
   - Intrinsic Mode Functions (IMF) extraction

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rapzyftminji/hrv-analysis-tool.git
cd ppg-hrv-analysis
```

2. Install required packages:
```bash
pip install streamlit numpy pandas scipy matplotlib plotly
```

## Usage

### Starting the Application

Run the Streamlit app from the project directory:
```bash
streamlit run streamlitApp.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Input Data Format

The application accepts PPG signal data in `.txt` format with the following specifications:
- One sample per line
- Plain text format
- Numerical values only
- Maximum duration: 300 seconds (configurable)

Example data format:
```
123.45
124.67
125.89
...
```

### Processing Workflow

1. **Upload Data**: Use the sidebar file uploader to load your PPG data file
2. **Set Parameters**: 
   - Adjust sampling rate (default: 50 Hz)
   - Configure downsample factor
   - Set peak detection parameters
3. **Select Analysis**: Navigate between different analysis modules using the sidebar
4. **View Results**: Interactive plots and metrics are displayed in real-time
5. **Export Results**: Download processed data and analysis results as needed

## Project Structure

```
ppg-hrv-analysis/
├── streamlitApp.py          # Main application entry point
├── lib/                      # Core processing libraries
│   ├── ppgProcessing.py     # Signal processing and DWT implementation
│   ├── hrv_calculation.py   # HRV metrics calculation
│   └── __pycache__/
├── pages/                    # Streamlit page modules
│   ├── mainPage.py          # DWT decomposition and breath analysis
│   ├── page1.py             # DWT performance analysis
│   ├── page2.py             # Cardiac analysis
│   ├── page3.py             # Time-domain HRV
│   ├── page5.py             # Frequency-domain HRV
│   ├── page6.py             # Non-linear HRV
│   └── page7.py             # EMD implementation
├── .gitignore
└── README.md
```

## Key Components

### ppgSignalProcessing Class
Handles signal preprocessing and DWT decomposition:
- Quadrature mirror filter implementation (Q1-Q8)
- Multi-level wavelet decomposition
- Signal normalization and detrending
- Downsampling operations

### hrvCalculation Class
Computes comprehensive HRV metrics:
- Time-domain statistics
- Frequency-domain power analysis
- Non-linear complexity measures
- Statistical operations without external dependencies

### Peak Detection Module
Robust peak detection using **PD (Proportional-Derivative) Controller Method**:
- **Adaptive thresholding** with PD control for dynamic threshold adjustment
- **Proportional component** responds to signal amplitude variations
- **Derivative component** tracks rate of change for enhanced peak sensitivity
- Local maxima/minima identification
- Artifact rejection
- Customizable detection parameters

The PD controller approach provides real-time adaptive threshold control, improving peak detection accuracy across varying signal conditions and noise levels.

## Configuration

Key parameters that can be adjusted:

- **Sampling Rate**: Original signal sampling frequency (Hz)
- **Downsample Factor**: Factor for reducing sampling rate
- **Peak Detection Threshold**: Sensitivity for detecting systolic peaks
- **Filter Levels**: Number of DWT decomposition levels
- **Window Size**: For frequency domain analysis

## Technical Details

### Discrete Wavelet Transform
The application uses a custom DWT implementation with quadrature mirror filters based on the Mallat algorithm. Six levels of filters (Q1-Q8) are available for different frequency band separations.

### PD Controller-Based Peak Detection
The peak detection algorithm employs a **Proportional-Derivative (PD) controller** strategy for adaptive threshold control:

- **Proportional Term (P)**: Adjusts threshold based on current signal amplitude, providing immediate response to signal level changes
- **Derivative Term (D)**: Monitors the rate of signal change, enhancing detection of sharp transitions characteristic of systolic peaks
- **Controlled Threshold Strategy**: Dynamically adapts to signal variations, reducing false positives/negatives
- **Real-Time Performance**: Enables efficient processing suitable for continuous monitoring applications

This approach, based on control theory principles, offers superior performance compared to fixed-threshold methods, particularly for PPG signals with varying baseline and amplitude characteristics.

### HRV Metrics

**Time Domain:**
- Mean RR, SDNN, RMSSD, NN50, pNN50, SDSD, HTI, TINN, Skewness of NN Distribution, CVNN, CVSD

**Frequency Domain:**
- Welch's PSD (VLF: 0.003-0.04; HzLF: 0.04-0.15 Hz; HF: 0.15-0.4 Hz)
- Total Power
- Total Power of LF
- Total Power of HF
- LF (Normalized Unit)
- HF (Normalized Unit)
- Peak Frequency of LF
- Peak Frequency of HF

**Non-Linear:**
- SD1
- SD2
- Ratio of SD1 & SD2
- Poincaré plot indices

## Troubleshooting

### Common Issues

**Application won't start:**
- Ensure all dependencies are installed
- Check Python version compatibility
- Verify you're in the correct directory

**File upload fails:**
- Confirm file format is plain text (.txt)
- Check that values are numeric
- Ensure file size is reasonable

**Peak detection issues:**
- Adjust threshold parameters
- Check signal quality
- Verify sampling rate is correct

**Memory issues with large files:**
- Reduce maximum duration
- Increase downsample factor
- Process data in smaller segments

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is available for academic and research purposes.

## Authors

- Repository: [rapzyftminji/hrv-analysis-tool](https://github.com/rapzyftminji/hrv-analysis-tool)

## Acknowledgments

- Based on research in PPG signal processing and HRV analysis
- Implements standard HRV metrics as defined by Task Force guidelines
- Uses custom DWT implementation for educational purposes

## References

- Bahoura, M., Hassani, M., & Hubin, M. (1997). DSP implementation of wavelet transform for real time ECG wave forms detection and heart rate analysis. *Computer Methods and Programs in Biomedicine*, 52(1), 35-44. https://doi.org/10.1016/S0169-2607(97)01780-X

- Chen, A., Zhang, Y., Zhang, M., Liu, W., Chang, S., Wang, H., He, J., & Huang, Q. (2020). A real time QRS detection algorithm based on ET and PD controlled threshold strategy. *Sensors*, 20(20), 4003. https://doi.org/10.3390/s20144003

- Hikmah, N. F., Arifin, A., & Sardjono, T. A. (2020). Delineation of ECG feature extraction using multiresolution analysis framework. *JUTI: Jurnal Ilmiah Teknologi Informasi*, 18(2), 135-146. https://doi.org/10.12962/j24068535.v18i2.a992

- Mallat, S. (1989). A theory for multiresolution signal decomposition: The wavelet representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674-693. https://doi.org/10.1109/34.192463

- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. *Circulation*, 93(5), 1043-1065. https://doi.org/10.1161/01.CIR.93.5.1043

## Contact

For questions or support, please open an issue on the GitHub repository.

---

**Last Updated:** December 2025
