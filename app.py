import streamlit as st
import pandas as pd

from Market_Behaviour_Intelligence_TFIDF_v7 import (
    tender_intelligence,
    tag_dictionary
)

st.set_page_config(
    page_title="Market Behaviour Intelligence Engine",
    layout="wide"
)

st.title("Market Behaviour Intelligence Engine")

st.markdown("""
This prototype analyses publicly available procurement data to identify
similar projects and provide market behaviour intelligence.

**Experimental tool:** Results are intended for research, benchmarking
and discussion only. They should not be treated as a prediction of a
winning bid or relied upon as the sole basis for commercial decisions.
""")


# ============================================================
# INPUTS
# ============================================================

st.subheader("Tender Input")

tender_description = st.text_area(
    "Tender Description",
    height=250,
    placeholder=(
        "Example: Detailed design of an active travel corridor "
        "including junction improvements, drainage design, traffic "
        "modelling and stakeholder engagement."
    )
)

estimated_fee = st.number_input(
    "Client Estimated Value / Reference Fee (£)",
    min_value=0.0,
    value=500000.0,
    step=10000.0,
    format="%.0f"
)

selected_tags = st.multiselect(
    "Relevant Project Tags",
    options=sorted(tag_dictionary.keys()),
    help=(
        "Optional. Select tags that best describe the tender. "
        "These influence the comparable-project search."
    )
)


# ============================================================
# FORMATTING FUNCTIONS
# ============================================================

def format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"£{value:,.0f}"


def format_percentage(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def format_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


# ============================================================
# RUN ANALYSIS
# ============================================================

if st.button("Analyse Tender", type="primary"):

    if not tender_description.strip():
        st.warning("Please enter a tender description.")
        st.stop()

    with st.spinner("Analysing historic procurement records..."):

        try:

            (
                summary,
                comparable_projects,
                winner_summary,
                classification_mix
            ) = tender_intelligence(
                description=tender_description,
                live_tender_estimate=estimated_fee,
                top_n=50,
                similarity_threshold=0.10,
                remove_fee_outliers=True,
                selected_tags=selected_tags
            )

        except Exception as error:

            st.error(
                "The Market Behaviour Intelligence Engine "
                "encountered an error."
            )

            st.exception(error)

            st.stop()


    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if summary["Comparable Projects Displayed"] == 0:

        st.warning(
            "No comparable projects met the current search criteria. "
            "Try removing some tags or providing a more detailed "
            "tender description."
        )

        st.stop()


    st.success("Analysis Complete")


    # ========================================================
    # CONFIDENCE
    # ========================================================

    st.subheader("Evidence & Confidence")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Confidence",
            summary["Confidence"]
        )

    with col2:
        st.metric(
            "Comparable Projects",
            summary["Comparable Projects Displayed"]
        )

    with col3:
        st.metric(
            "Average Similarity",
            format_percentage(
                summary["Average Similarity"]
            )
        )

    with col4:
        st.metric(
            "Top Similarity",
            format_percentage(
                summary["Top Similarity"]
            )
        )


    # ========================================================
    # FEE BENCHMARK
    # ========================================================

    st.subheader("Historic Winning Fee Benchmark")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Adjusted P25",
            format_currency(
                summary["Adjusted Winning Fee P25"]
            )
        )

    with col2:
        st.metric(
            "Adjusted Median",
            format_currency(
                summary["Adjusted Winning Fee Median"]
            )
        )

    with col3:
        st.metric(
            "Adjusted P75",
            format_currency(
                summary["Adjusted Winning Fee P75"]
            )
        )

    with col4:
        st.metric(
            "Similarity Weighted Fee",
            format_currency(
                summary["Similarity Weighted Adjusted Fee"]
            )
        )


    # ========================================================
    # INDICATIVE FEE SCENARIOS
    # ========================================================

    st.subheader("Indicative Fee Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Price-Led Scenario",
            format_currency(
                summary["Price Led Indicative Fee"]
            )
        )

    with col2:
        st.metric(
            "Blended Scenario",
            format_currency(
                summary["Blended Indicative Winner Fee"]
            )
        )

    with col3:
        st.metric(
            "Premium Scenario",
            format_currency(
                summary[
                    "Premium Scenario Indicative Winner Fee"
                ]
            )
        )


    # ========================================================
    # BENCHMARK MESSAGE
    # ========================================================

    st.subheader("Benchmark Interpretation")

    st.info(
        summary["Estimate Benchmark Message"]
    )


    # ========================================================
    # MARKET BEHAVIOUR
    # ========================================================

    st.subheader("Historic Market Behaviour")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Win at Lowest Bid",
            format_percentage(
                summary["Win At Low Probability"]
            )
        )

    with col2:
        st.metric(
            "Weighted Premium",
            format_percentage(
                summary[
                    "Weighted Premium Factor All Low-Win Records"
                ]
            )
        )

    with col3:
        st.metric(
            "Historic Bid Spread",
            format_percentage(
                summary[
                    "Historic Bid Spread Ratio Median"
                ]
            )
        )

    with col4:
        st.metric(
            "Award Position Index",
            format_number(
                summary["Award Position Index Median"]
            )
        )


    # ========================================================
    # DATA QUALITY
    # ========================================================

    st.caption(
        f"Behaviour calculations use "
        f"{summary['Behaviour Valid Records']} valid records. "
        f"{summary['Behaviour Excluded Records']} comparable "
        f"records were displayed but excluded from behaviour "
        f"calculations by commercial data-quality controls."
    )


    # ========================================================
    # MORE BEHAVIOUR ANALYSIS
    # ========================================================

    with st.expander("View Detailed Behaviour Analysis"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Winner Premium Over Low",
                format_percentage(
                    summary[
                        "Winner Premium Over Low Median"
                    ]
                )
            )

        with col2:
            st.metric(
                "Positive Premium Median",
                format_percentage(
                    summary[
                        "Positive Premium Median"
                    ]
                )
            )

        with col3:
            st.metric(
                "Winner Discount From High",
                format_percentage(
                    summary[
                        "Winner Discount From High Median"
                    ]
                )
            )

        st.write(
            "Low + Win records:",
            summary["Low and Win Records Used"]
        )

        st.write(
            "Won at lowest bid:",
            summary["Won At Low Records Used"]
        )

        st.write(
            "Won above lowest bid:",
            summary["Won Above Low Records Used"]
        )

        st.write(
            "Full bid-range records:",
            summary["Full Bid Range Records Used"]
        )


    # ========================================================
    # COMPARABLE PROJECTS
    # ========================================================

    st.subheader("Top Comparable Projects")

    st.dataframe(
        comparable_projects,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # HISTORIC WINNERS
    # ========================================================

    st.subheader("Top Historic Winners")

    if winner_summary.empty:

        st.write(
            "No winner information is available "
            "for the selected comparable projects."
        )

    else:

        winners_table = (
            winner_summary
            .rename_axis("Winner")
            .reset_index(name="Comparable Project Count")
        )

        st.dataframe(
            winners_table,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # CLASSIFICATION MIX
    # ========================================================

    st.subheader("Project Classification Mix")

    if classification_mix.empty:

        st.write(
            "No project classification information is available."
        )

    else:

        classification_table = (
            classification_mix
            .rename_axis("Project Classification")
            .reset_index(name="Comparable Project Count")
        )

        st.dataframe(
            classification_table,
            use_container_width=True,
            hide_index=True
        )
