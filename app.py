import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Market Behaviour Intelligence Engine",
    layout="wide"
)

st.title("Market Behaviour Intelligence Engine")

st.markdown("""
This prototype analyses historic procurement data to identify
similar projects and provide market intelligence insights.

⚠️ Experimental tool for discussion and research purposes only.
""")

# ----------------------------
# USER INPUTS
# ----------------------------

tender_title = st.text_input(
    "Tender Title",
    placeholder="e.g. A123 Junction Improvement Detailed Design"
)

tender_description = st.text_area(
    "Tender Description",
    height=200,
    placeholder="Paste tender description here..."
)

estimated_fee = st.number_input(
    "Estimated Fee (£)",
    min_value=0.0,
    value=100000.0,
    step=1000.0
)

# ----------------------------
# ANALYSE BUTTON
# ----------------------------

if st.button("Analyse Tender"):

    st.success("Analysis started")

    st.subheader("Tender Summary")

    st.write("Title:", tender_title)
    st.write("Estimated Fee:", f"£{estimated_fee:,.0f}")

    st.subheader("Market Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Confidence", "Medium")

    with col2:
        st.metric("Expected Winning Fee", "£120,000")

    with col3:
        st.metric("Market Uncertainty", "18%")

    with col4:
        st.metric("Award Position Index", "0.62")

    st.subheader("Comparable Projects")

    sample = pd.DataFrame({
        "Project": [
            "Active Travel Scheme",
            "Detailed Design Framework",
            "Junction Improvement"
        ],
        "Similarity": [0.91, 0.84, 0.80]
    })

    st.dataframe(sample, use_container_width=True)

    st.subheader("Notes")

    st.info(
        "This is currently a demonstration interface. "
        "The next step is to connect it to the intelligence engine."
    )
