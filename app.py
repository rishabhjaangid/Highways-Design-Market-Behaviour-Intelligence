import pandas as pd
import streamlit as st

from Market_Behaviour_Intelligence_TFIDF_v7 import (
    tag_dictionary,
    tender_intelligence,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Market Behaviour Intelligence Engine",
    page_icon="📊",
    layout="wide",
)

st.title("Market Behaviour Intelligence Engine")

st.markdown(
    """
This experimental prototype uses publicly available procurement data
to identify comparable projects and explore historic market behaviour.

The results are intended for research, benchmarking and discussion.
They should not be treated as a prediction of a winning bid or relied
upon as the sole basis for pricing or commercial decisions.
"""
)


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
    ),
)

estimated_fee = st.number_input(
    "Client Estimated Value / Reference Fee (£)",
    min_value=0.0,
    value=500000.0,
    step=10000.0,
    format="%.0f",
)

selected_tags = st.multiselect(
    "Relevant Project Tags",
    options=sorted(tag_dictionary.keys()),
    help=(
        "Select the tags that best describe the tender. "
        "The tags influence the comparable-project search."
    ),
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

if st.button(
    "Analyse Tender",
    type="primary",
):

    if not tender_description.strip():
        st.warning(
            "Please enter a tender description."
        )
        st.stop()

    with st.spinner(
        "Analysing historic procurement records..."
    ):

        try:
            (
                summary,
                comparable_projects,
                winner_summary,
                classification_mix,
            ) = tender_intelligence(
                description=tender_description,
                live_tender_estimate=estimated_fee,
                top_n=50,
                similarity_threshold=0.10,
                remove_fee_outliers=True,
                selected_tags=selected_tags,
            )

        except Exception as error:
            st.error(
                "The analysis could not be completed."
            )
            st.exception(error)
            st.stop()

    if summary["Comparable Projects Displayed"] == 0:
        st.warning(
            "No comparable projects met the current similarity "
            "and tag filters. Try removing some tags or entering "
            "a more detailed tender description."
        )
        st.stop()

    st.success("Analysis Complete")


    # ========================================================
    # EVIDENCE AND CONFIDENCE
    # ========================================================

    st.subheader("Evidence & Confidence")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Confidence "
            "(strength of comparable evidence)",
            summary["Confidence"],
        )

    with col2:
        st.metric(
            "Comparable Projects "
            "(projects used after filters)",
            summary["Comparable Projects Displayed"],
        )

    with col3:
        st.metric(
            "Average Similarity "
            "(average Term Frequency–Inverse Document "
            "Frequency (TF-IDF) text match)",
            format_percentage(
                summary["Average Similarity"]
            ),
        )

    with col4:
        st.metric(
            "Top Similarity "
            "(best Term Frequency–Inverse Document "
            "Frequency (TF-IDF) text match)",
            format_percentage(
                summary["Top Similarity"]
            ),
        )


    # ========================================================
    # HISTORIC WINNING FEE BENCHMARK
    # ========================================================

    st.subheader("Historic Winning Fee Benchmark")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Adjusted P25 "
            "(25% of awards below this)",
            format_currency(
                summary["Adjusted Winning Fee P25"]
            ),
        )

    with col2:
        st.metric(
            "Adjusted Median "
            "(50% of awards below this)",
            format_currency(
                summary[
                    "Adjusted Winning Fee Median"
                ]
            ),
        )

    with col3:
        st.metric(
            "Adjusted P75 "
            "(75% of awards below this)",
            format_currency(
                summary["Adjusted Winning Fee P75"]
            ),
        )

    with col4:
        st.metric(
            "Similarity-Weighted Historic Fee "
            "(closer matches weighted more)",
            format_currency(
                summary[
                    "Similarity Weighted Adjusted Fee"
                ]
            ),
        )


    # ========================================================
    # INDICATIVE MARKET SCENARIOS
    # ========================================================

    st.subheader("Indicative Market Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Price-Led Scenario "
            "(client estimate/reference fee)",
            format_currency(
                summary["Price Led Indicative Fee"]
            ),
        )

    with col2:
        st.metric(
            "Blended Scenario "
            "(estimate + historic weighted premium)",
            format_currency(
                summary[
                    "Blended Indicative Winner Fee"
                ]
            ),
        )

    with col3:
        st.metric(
            "Premium Scenario "
            "(estimate + premium-only behaviour)",
            format_currency(
                summary[
                    "Premium Scenario Indicative "
                    "Winner Fee"
                ]
            ),
        )


    # ========================================================
    # BENCHMARK INTERPRETATION
    # ========================================================

    st.subheader("Benchmark Interpretation")

    st.info(
        summary["Estimate Benchmark Message"]
    )


    # ========================================================
    # HISTORIC MARKET BEHAVIOUR
    # ========================================================

    st.subheader("Historic Market Behaviour")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Win at Lowest Bid "
            "(historic % awarded at lowest)",
            format_percentage(
                summary["Win At Low Probability"]
            ),
        )

    with col2:
        st.metric(
            "Weighted Premium "
            "(historic premium above lowest bid)",
            format_percentage(
                summary[
                    "Weighted Premium Factor "
                    "All Low-Win Records"
                ]
            ),
        )

    with col3:
        st.metric(
            "Historic Bid Spread "
            "(high-low range ÷ lowest bid)",
            format_percentage(
                summary[
                    "Historic Bid Spread Ratio Median"
                ]
            ),
        )

    with col4:
        st.metric(
            "Award Position Index "
            "(0 = lowest bid, 1 = highest bid)",
            format_number(
                summary["Award Position Index Median"]
            ),
        )

    st.caption(
        f"Behaviour calculations use "
        f"{summary['Behaviour Valid Records']} valid records. "
        f"{summary['Behaviour Excluded Records']} displayed "
        f"comparables were excluded from behaviour calculations "
        f"by the commercial data-quality controls."
    )


    # ========================================================
    # DETAILED BEHAVIOUR ANALYSIS
    # ========================================================

    with st.expander(
        "View Detailed Behaviour Analysis"
    ):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Winner Premium Over Low "
                "(median amount above lowest bid)",
                format_percentage(
                    summary[
                        "Winner Premium Over Low Median"
                    ]
                ),
            )

        with col2:
            st.metric(
                "Positive Premium Median "
                "(cases won above lowest only)",
                format_percentage(
                    summary["Positive Premium Median"]
                ),
            )

        with col3:
            st.metric(
                "Winner Discount From High "
                "(median amount below highest bid)",
                format_percentage(
                    summary[
                        "Winner Discount From High Median"
                    ]
                ),
            )

        st.write(
            "Low and win records used:",
            summary["Low and Win Records Used"],
        )

        st.write(
            "Won at lowest bid:",
            summary["Won At Low Records Used"],
        )

        st.write(
            "Won above lowest bid:",
            summary["Won Above Low Records Used"],
        )

        st.write(
            "Full bid-range records used:",
            summary["Full Bid Range Records Used"],
        )


    # ========================================================
    # COMPARABLE PROJECTS
    # ========================================================

    st.subheader("Top Comparable Projects")

    display_data = comparable_projects.copy()

    currency_columns = [
        "Awarded Value",
        "Inflation Adjusted Awarded Value",
        "Value Low",
        "Value High",
    ]

    for column in currency_columns:
        if column in display_data.columns:
            display_data[column] = (
                display_data[column].apply(
                    lambda value: (
                        f"£{value:,.0f}"
                        if pd.notna(value)
                        else ""
                    )
                )
            )

    percentage_columns = [
        "Winner Premium Over Low",
        "Winner Discount From High",
        "Historic Bid Spread Ratio",
        "Similarity",
    ]

    for column in percentage_columns:
        if column in display_data.columns:
            display_data[column] = (
                display_data[column].apply(
                    lambda value: (
                        f"{value:.1%}"
                        if pd.notna(value)
                        else ""
                    )
                )
            )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # TOP HISTORIC WINNERS
    # ========================================================

    st.subheader("Top Historic Winners")

    if winner_summary.empty:
        st.write(
            "No winner information is available "
            "for the comparable projects."
        )

    else:
        winners_table = (
            winner_summary
            .rename_axis("Winner")
            .reset_index(
                name="Comparable Project Count"
            )
        )

        st.dataframe(
            winners_table,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # PROJECT CLASSIFICATION MIX
    # ========================================================

    st.subheader("Project Classification Mix")

    if classification_mix.empty:
        st.write(
            "No project classification information "
            "is available."
        )

    else:
        classification_table = (
            classification_mix
            .rename_axis("Project Classification")
            .reset_index(
                name="Comparable Project Count"
            )
        )

        st.dataframe(
            classification_table,
            use_container_width=True,
            hide_index=True,
        )
