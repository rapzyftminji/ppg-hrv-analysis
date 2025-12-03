import streamlit as st

main_page = st.Page("pages/mainPage.py", title="DWT Decomposition Process and Breath Analysis")
page_1 = st.Page("pages/page1.py", title="DWT Performance Analysis")
page_2 = st.Page("pages/page2.py", title="Cardiac Analysis")
page_3 = st.Page("pages/page3.py", title="HRV Time-Domain Analysis")
page_5 = st.Page("pages/page5.py", title="HRV Frequency-Domain Analysis (Welch's PSD)")
page_6 = st.Page("pages/page6.py", title="Non-Linear Method HRV Analysis")
page_7 = st.Page("pages/page7.py", title="Manual EMD with Custom Peak Detection and Spline Interpolation")

pg = st.navigation([main_page, page_1, page_2, page_3, page_5, page_6, page_7])

pg.run()
