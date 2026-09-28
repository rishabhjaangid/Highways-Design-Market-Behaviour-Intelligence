import streamlit as st
from market_behaviour_engine import run_market_analysis

st.title(
    "Market Behaviour Intelligence Engine"
)

description = st.text_area(
    "Tender Description"
)

if st.button("Analyse"):

    result = run_market_analysis(
        description
    )

    st.metric(
        "Confidence",
        result["confidence"]
    )

    st.metric(
        "Winning Fee Median",
        f"£{result['fee_median']:,.0f}"
    )

    st.metric(
        "Winning Fee P25",
        f"£{result['fee_p25']:,.0f}"
    )

    st.metric(
        "Winning Fee P75",
        f"£{result['fee_p75']:,.0f}"
    )

    st.subheader(
        "Top Comparable Projects"
    )

    st.dataframe(
        result["comparables"]
    )
