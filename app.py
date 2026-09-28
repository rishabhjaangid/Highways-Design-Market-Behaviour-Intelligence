import streamlit as st
import pandas as pd

from Market_Behaviour_Intelligence_TFIDF_v7 import (
    run_market_analysis
)

st.set_page_config(
    page_title="Market Behaviour Intelligence Engine",
    layout="wide"
)

st.title("Market Behaviour Intelligence Engine")

st.markdown("""
This prototype analyses publicly available procurement data
to identify similar projects and provide market intelligence insights.

⚠️ Experimental tool for discussion and research purposes only.
""")

tender_description = st.text_area(
    "Tender Description",
    height=250,
    placeholder="Paste tender description here..."
)

if st.button("Analyse Tender"):

    if not tender_description.strip():
        st.warning("Please enter a tender description.")
    else:

        with st.spinner("Analysing historic projects..."):

            result = run_market_analysis(
                tender_description
            )

        st.success("Analysis Complete")

        st.subheader("Fee Benchmark")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "P25 Fee",
                f"£{result['fee_p25']:,.0f}"
            )

        with col2:
            st.metric(
                "Median Fee",
                f"£{result['fee_median']:,.0f}"
            )

        with col3:
            st.metric(
                "P75 Fee",
                f"£{result['fee_p75']:,.0f}"
            )

        st.subheader("Confidence")

        st.write(
            result["confidence"]
        )

        st.subheader("Top Comparable Projects
