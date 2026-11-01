from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "notices3_shortlisted_sorted.xlsx"

DEFAULT_ANNUAL_INFLATION_RATE = 0.03
CURRENT_YEAR = 2026

TOP_N = 50
SIMILARITY_THRESHOLD = 0.10
TAG_MATCH_BONUS = 0.05
REQUIRE_SELECTED_TAG_MATCH = True

EXCLUDE_FRAMEWORKS = False
EXCLUDE_CONSTRUCTION_WORKS = False
EXCLUDE_NON_HIGHWAYS_BUILDING = True
EXCLUDE_VERY_LARGE_AWARDS = False

MAX_AWARD_VALUE = 1_000_000_000
REMOVE_FEE_OUTLIERS = True
PREMIUM_ZERO_TOLERANCE = 0.000001

pd.set_option("display.max_colwidth", 220)
pd.set_option("display.float_format", lambda value: f"{value:,.2f}")

warnings.filterwarnings(
    "default",
    category=pd.errors.PerformanceWarning,
)


# ============================================================
# TEXT FUNCTIONS
# ============================================================

def clean_text(text):
    """Clean text for classification and TF-IDF matching."""

    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("_x000D_", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def classify_project(text):
    """Assign one primary classification to a historic project."""

    text = clean_text(text)

    if any(
        term in text
        for term in [
            "framework",
            "framework agreement",
            "professional services partnership",
            "term arrangement",
            "term consultancy",
            "panel",
        ]
    ):
        return "Framework / Term Consultancy"

    if any(
        term in text
        for term in [
            "construction",
            "demolition",
            "scaffolding",
            "supply and installation",
            "installation of",
            "replacement of lighting",
            "new build",
            "building works",
        ]
    ):
        return "Construction / Physical Works"

    if any(
        term in text
        for term in [
            "business case",
            "outline business case",
            "obc",
            "fbc",
            "sopc",
            "economic assessment",
            "funding approval",
            "technical evaluator",
            "local growth fund",
        ]
    ):
        return "Business Case / Funding"

    if any(
        term in text
        for term in [
            "technical partner",
            "programme support",
            "programme director",
            "mobilisation",
            "portfolio",
            "governance",
        ]
    ):
        return "Strategic / Technical Partner"

    if any(
        term in text
        for term in [
            "independent technical advisor",
            "technical advisor",
            "assurance",
            "review of detailed design",
            "technical evaluation",
        ]
    ):
        return "Assurance / Technical Advisor"

    if any(
        term in text
        for term in [
            "road condition monitoring",
            "rcm",
            "scrim",
            "scanner",
            "ukpms",
            "pavement survey",
            "condition survey",
            "network survey",
        ]
    ):
        return "Road Condition Monitoring / Asset Survey"

    if any(
        term in text
        for term in [
            "traffic signal",
            "signals",
            "signal scheme",
            "ibus",
            "vehicle detection",
            "hadecs",
            "enforcement",
            "its",
        ]
    ):
        return "Traffic Signals / ITS"

    if any(
        term in text
        for term in [
            "traffic modelling",
            "transport modelling",
            "traffic model",
            "transport model",
            "modelling",
            "model update",
        ]
    ):
        return "Traffic / Transport Modelling"

    if any(
        term in text
        for term in [
            "active travel",
            "cycling",
            "cycleway",
            "cycle route",
            "walking",
            "pedestrian",
            "lcwip",
        ]
    ):
        return "Active Travel / Walking / Cycling"

    if any(
        term in text
        for term in [
            "bus priority",
            "bsip",
            "bus service improvement",
            "bus route",
            "public transport",
        ]
    ):
        return "Bus Priority / Public Transport"

    if any(
        term in text
        for term in [
            "public realm",
            "urban realm",
            "town centre",
            "city centre",
            "streetscape",
            "event spaces",
        ]
    ):
        return "Public Realm / Urban Realm"

    if any(
        term in text
        for term in [
            "detailed design",
            "design changes",
            "as-built drawings",
            "construction support",
            "buildability",
        ]
    ):
        return "Detailed Design"

    if any(
        term in text
        for term in [
            "preliminary design",
            "concept design",
            "option appraisal",
            "options appraisal",
            "feasibility",
            "study",
        ]
    ):
        return "Preliminary / Concept / Feasibility"

    if any(
        term in text
        for term in [
            "junction",
            "corridor",
            "road improvement",
            "highway improvement",
            "carriageway",
            "roundabout",
        ]
    ):
        return "Junction / Corridor Improvement"

    if any(
        term in text
        for term in [
            "bridge",
            "viaduct",
            "retaining wall",
            "culvert",
            "structure",
            "archway",
        ]
    ):
        return "Structures / Bridges"

    if any(
        term in text
        for term in [
            "drainage",
            "flood",
            "surface water",
            "suds",
            "attenuation",
            "watercourse",
        ]
    ):
        return "Drainage / Flood / Water"

    if any(
        term in text
        for term in [
            "asset valuation",
            "asset management",
            "mrm",
            "survey",
        ]
    ):
        return "Asset / Survey / Data Collection"

    if any(
        term in text
        for term in [
            "housing",
            "homes",
            "building",
            "redevelopment",
            "cemetery",
        ]
    ):
        return "Building / Non-Highways Development"

    return "Other"


# ============================================================
# TAG DICTIONARY
# ============================================================

tag_dictionary = {
    "active travel": [
        "active travel",
        "cycling",
        "cycleway",
        "cycle route",
        "walking",
        "pedestrian",
        "lcwip",
    ],
    "detailed design": [
        "detailed design",
        "as-built",
        "design changes",
    ],
    "preliminary design": [
        "preliminary design",
        "concept design",
    ],
    "business case": [
        "business case",
        "obc",
        "fbc",
        "funding",
    ],
    "traffic modelling": [
        "traffic modelling",
        "transport modelling",
        "traffic model",
        "transport model",
        "modelling",
    ],
    "traffic signals": [
        "traffic signal",
        "signals",
        "signal scheme",
        "ibus",
        "vehicle detection",
    ],
    "bus priority": [
        "bus priority",
        "bsip",
        "bus service improvement",
    ],
    "junction": [
        "junction",
        "roundabout",
    ],
    "corridor": [
        "corridor",
    ],
    "public realm": [
        "public realm",
        "urban realm",
        "town centre",
        "city centre",
    ],
    "structures": [
        "bridge",
        "viaduct",
        "culvert",
        "retaining wall",
        "structure",
    ],
    "drainage": [
        "drainage",
        "flood",
        "surface water",
        "suds",
    ],
    "framework": [
        "framework",
        "panel",
        "term arrangement",
    ],
    "assurance": [
        "assurance",
        "technical advisor",
        "technical evaluator",
        "governance",
    ],
    "survey": [
        "survey",
        "surveys",
        "condition survey",
        "network survey",
        "pavement survey",
        "scrim",
        "scanner",
        "mrm",
        "ukpms",
    ],
    "road condition monitoring": [
        "road condition monitoring",
        "rcm",
        "scrim",
        "scanner",
        "ukpms",
        "mrm",
        "condition survey",
        "network survey",
        "pavement survey",
    ],
    "asset management": [
        "asset management",
        "road condition",
        "network condition",
        "ukpms",
        "scrim",
        "scanner",
        "rcm",
    ],
    "construction": [
        "construction",
        "demolition",
        "scaffolding",
        "installation",
        "new build",
    ],
}


def create_tags(text):
    """Create a comma-separated tag list for a project."""

    text = clean_text(text)
    tags = []

    for tag, keywords in tag_dictionary.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)

    return ", ".join(tags) if tags else "none"


def suggest_tags(description):
    """Return tags whose keywords appear in a live description."""

    description = clean_text(description)
    scores = {}

    for tag, keywords in tag_dictionary.items():
        score = sum(keyword in description for keyword in keywords)

        if score > 0:
            scores[tag] = score

    return [
        tag
        for tag, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

required_columns = [
    "Organisation Name",
    "Title",
    "Description",
    "Value Low",
    "Value High",
    "Awarded Date",
    "Awarded Value",
    "Winner",
]


def load_and_prepare_data():
    """Load the spreadsheet and create all model fields."""

    if not FILE.exists():
        raise FileNotFoundError(
            f"Could not find '{FILE.name}'. "
            "Place the spreadsheet in the same GitHub folder "
            "as this Python file."
        )

    data = pd.read_excel(FILE, engine="openpyxl")

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Spreadsheet is missing required columns: "
            f"{missing_columns}"
        )

    data = data.copy()

    for column in [
        "Value Low",
        "Value High",
        "Awarded Value",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    for column in [
        "Organisation Name",
        "Title",
        "Description",
        "Winner",
    ]:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
        )

    data["Awarded Date"] = pd.to_datetime(
        data["Awarded Date"],
        dayfirst=True,
        errors="coerce",
    )

    data["Award Year"] = data["Awarded Date"].dt.year

    data["Project Age"] = (
        CURRENT_YEAR - data["Award Year"]
    )

    data.loc[
        data["Project Age"].lt(0),
        "Project Age",
    ] = np.nan

    data["Inflation Rate Used"] = (
        DEFAULT_ANNUAL_INFLATION_RATE
    )

    data["Inflation Factor"] = np.where(
        data["Project Age"].notna(),
        (1 + DEFAULT_ANNUAL_INFLATION_RATE)
        ** data["Project Age"],
        np.nan,
    )

    data["Inflation Adjusted Awarded Value"] = np.where(
        data["Inflation Factor"].notna(),
        data["Awarded Value"] * data["Inflation Factor"],
        data["Awarded Value"],
    )

    low = data["Value Low"]
    high = data["Value High"]
    win = data["Awarded Value"]

    has_low = low.notna() & low.gt(0)
    has_high = high.notna() & high.gt(0)
    has_win = win.notna() & win.gt(0)

    full_range = (
        has_low
        & has_high
        & has_win
        & high.gt(low)
    )

    low_win = has_low & has_win
    win_high = has_high & has_win

    conditions = [
        full_range,
        low_win & ~full_range,
        win_high & ~full_range,
        low.eq(0),
        has_win,
    ]

    choices = [
        "Full bid range",
        "Low and win only",
        "Win and high only",
        "Zero-low record",
        "Awarded only",
    ]

    data["Analysis Type"] = np.select(
        conditions,
        choices,
        default="Data issue or missing awarded value",
    )

    has_low_win = (
        low.notna()
        & low.gt(0)
        & win.notna()
        & win.gt(0)
    )

    has_win_high = (
        high.notna()
        & high.gt(0)
        & win.notna()
        & win.gt(0)
    )

    has_full_range = (
        low.notna()
        & high.notna()
        & win.notna()
        & low.gt(0)
        & high.gt(low)
    )

    data["Winner Premium Over Low"] = np.where(
        has_low_win,
        (win - low) / low,
        np.nan,
    )

    data["Winner Discount From High"] = np.where(
        has_win_high,
        (high - win) / high,
        np.nan,
    )

    data["Market Midpoint"] = np.where(
        has_full_range,
        (low + high) / 2,
        np.nan,
    )

    data["Bid Spread"] = np.where(
        has_full_range,
        high - low,
        np.nan,
    )

    data["Historic Bid Spread Ratio"] = np.where(
        has_full_range,
        (high - low) / low,
        np.nan,
    )

    data["Market Uncertainty Index"] = np.where(
        has_full_range,
        data["Bid Spread"] / data["Market Midpoint"],
        np.nan,
    )

    data["Market Consensus Score"] = np.where(
        has_full_range,
        np.maximum(
            0,
            100 - (
                data["Market Uncertainty Index"] * 100
            ),
        ),
        np.nan,
    )

    data["Award Position Index"] = np.where(
        has_full_range,
        (win - low) / (high - low),
        np.nan,
    )

    data["Midpoint Deviation %"] = np.where(
        has_full_range,
        (win - data["Market Midpoint"])
        / data["Market Midpoint"],
        np.nan,
    )

    data["Commercial Data Quality"] = "Valid"

    same_low_awarded_high = (
        low.notna()
        & win.notna()
        & high.notna()
        & low.eq(win)
        & win.eq(high)
    )

    data.loc[
        same_low_awarded_high,
        "Commercial Data Quality",
    ] = "Low = Awarded = High"

    awarded_equals_high = (
        win.notna()
        & high.notna()
        & win.eq(high)
        & ~same_low_awarded_high
    )

    data.loc[
        awarded_equals_high,
        "Commercial Data Quality",
    ] = "Awarded = High"

    awarded_equals_low = (
        win.notna()
        & low.notna()
        & win.eq(low)
        & data["Commercial Data Quality"].eq("Valid")
    )

    data.loc[
        awarded_equals_low,
        "Commercial Data Quality",
    ] = "Awarded = Low"

    data["Use For Behaviour Analysis"] = True

    data.loc[
        data["Commercial Data Quality"].isin(
            [
                "Low = Awarded = High",
                "Awarded = High",
            ]
        ),
        "Use For Behaviour Analysis",
    ] = False

    data["Search_Text"] = (
        data["Organisation Name"].apply(clean_text)
        + " "
        + data["Title"].apply(clean_text)
        + " "
        + data["Description"].apply(clean_text)
        + " "
        + data["Description"].apply(clean_text)
    )

    data["Project Classification"] = (
        data["Search_Text"].apply(classify_project)
    )

    data["Project Tags"] = (
        data["Search_Text"].apply(create_tags)
    )

    data["Is Framework"] = (
        data["Project Classification"]
        .eq("Framework / Term Consultancy")
    )

    data["Is Construction Work"] = (
        data["Project Classification"]
        .eq("Construction / Physical Works")
    )

    data["Is Non Highways Building"] = (
        data["Project Classification"]
        .eq("Building / Non-Highways Development")
    )

    data["Is Very Large Award"] = (
        data["Awarded Value"] > MAX_AWARD_VALUE
    )

    data["Benchmark Eligible"] = True

    if EXCLUDE_FRAMEWORKS:
        data.loc[
            data["Is Framework"],
            "Benchmark Eligible",
        ] = False

    if EXCLUDE_CONSTRUCTION_WORKS:
        data.loc[
            data["Is Construction Work"],
            "Benchmark Eligible",
        ] = False

    if EXCLUDE_NON_HIGHWAYS_BUILDING:
        data.loc[
            data["Is Non Highways Building"],
            "Benchmark Eligible",
        ] = False

    if EXCLUDE_VERY_LARGE_AWARDS:
        data.loc[
            data["Is Very Large Award"],
            "Benchmark Eligible",
        ] = False

    return data.copy()


df = load_and_prepare_data()


# ============================================================
# TF-IDF MODEL
# ============================================================

custom_stop_words = [
    "services",
    "service",
    "contract",
    "project",
    "scheme",
    "support",
    "consultancy",
    "consultant",
    "council",
    "authority",
    "framework",
    "award",
    "procurement",
    "provide",
    "delivery",
    "works",
    "lot",
    "stage",
    "phase",
    "including",
    "required",
    "appoint",
    "seeking",
    "undertake",
]

all_stop_words = list(
    ENGLISH_STOP_WORDS.union(custom_stop_words)
)

vectorizer = TfidfVectorizer(
    stop_words=all_stop_words,
    ngram_range=(1, 3),
    max_features=20000,
    min_df=2,
)

historic_vectors = vectorizer.fit_transform(
    df["Search_Text"]
)


# ============================================================
# COMPARABLE-PROJECT SEARCH
# ============================================================

def normalise_selected_tags(selected_tags):
    if selected_tags is None:
        return []

    return [
        str(tag).strip().lower()
        for tag in selected_tags
        if str(tag).strip()
    ]


def count_tag_matches(
    project_tags,
    project_classification,
    selected_tags,
):
    text = (
        f"{project_tags} {project_classification}"
        .lower()
    )

    return sum(tag in text for tag in selected_tags)


def find_similar_projects(
    description,
    top_n=TOP_N,
    only_benchmark_eligible=True,
    restrict_to_classification=None,
    selected_tags=None,
    require_selected_tag_match=REQUIRE_SELECTED_TAG_MATCH,
    tag_match_bonus=TAG_MATCH_BONUS,
):
    """Find and rank comparable historic projects."""

    selected_tags = normalise_selected_tags(
        selected_tags
    )

    query_text = clean_text(description)

    if not query_text:
        return df.head(0).copy()

    query_vector = vectorizer.transform(
        [query_text]
    )

    similarities = cosine_similarity(
        query_vector,
        historic_vectors,
    )[0]

    results = df.copy()
    results["Similarity"] = similarities

    if only_benchmark_eligible:
        results = results[
            results["Benchmark Eligible"]
        ].copy()

    if restrict_to_classification is not None:
        results = results[
            results["Project Classification"].eq(
                restrict_to_classification
            )
        ].copy()

    if selected_tags:
        results["Tag Bonus"] = results.apply(
            lambda row: count_tag_matches(
                row.get("Project Tags", ""),
                row.get("Project Classification", ""),
                selected_tags,
            ),
            axis=1,
        )
    else:
        results["Tag Bonus"] = 0

    if (
        selected_tags
        and require_selected_tag_match
    ):
        results = results[
            results["Tag Bonus"] > 0
        ].copy()

    results["Final Score"] = (
        results["Similarity"]
        + tag_match_bonus * results["Tag Bonus"]
    )

    results = results.sort_values(
        "Final Score",
        ascending=False,
    )

    return results.head(top_n).copy()


# ============================================================
# ANALYTICAL HELPERS
# ============================================================

def remove_award_value_outliers(
    data,
    value_column="Inflation Adjusted Awarded Value",
):
    """Remove fee outliers using the 1.5 x IQR rule."""

    clean_count = data[value_column].dropna().shape[0]

    if data.empty or clean_count < 4:
        return data.copy()

    q1 = data[value_column].quantile(0.25)
    q3 = data[value_column].quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return data.copy()

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return data[
        data[value_column].between(
            lower_bound,
            upper_bound,
            inclusive="both",
        )
    ].copy()


def calculate_confidence(similar_projects):
    """Calculate evidence confidence."""

    if similar_projects.empty:
        return "No evidence", 0.0, 0.0

    match_count = len(similar_projects)
    average_similarity = (
        similar_projects["Similarity"].mean()
    )
    top_similarity = (
        similar_projects["Similarity"].max()
    )

    fee_series = (
        similar_projects[
            "Inflation Adjusted Awarded Value"
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if fee_series.empty or fee_series.mean() == 0:
        fee_cv = np.nan
    else:
        fee_cv = fee_series.std() / fee_series.mean()

    high_confidence = (
        match_count >= 20
        and average_similarity >= 0.16
        and top_similarity >= 0.25
        and pd.notna(fee_cv)
        and fee_cv <= 1.0
    )

    medium_confidence = (
        match_count >= 10
        and average_similarity >= 0.10
    )

    if high_confidence:
        confidence = "High"
    elif medium_confidence:
        confidence = "Medium"
    else:
        confidence = "Low"

    return (
        confidence,
        average_similarity,
        top_similarity,
    )


def weighted_average(values, weights):
    """Calculate a weighted average with validity checks."""

    valid = (
        values.notna()
        & weights.notna()
        & weights.gt(0)
    )

    if valid.sum() == 0:
        return np.nan

    return np.average(
        values[valid],
        weights=weights[valid],
    )


def safe_percentiles(series):
    """Calculate P25, median and P75 safely."""

    clean = (
        series
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if clean.empty:
        return {
            "P25": np.nan,
            "Median": np.nan,
            "P75": np.nan,
        }

    return {
        "P25": clean.quantile(0.25),
        "Median": clean.median(),
        "P75": clean.quantile(0.75),
    }


def estimate_vs_benchmark_message(
    live_estimate,
    adjusted_median,
):
    """Interpret the live estimate against the historic median."""

    if (
        live_estimate is None
        or pd.isna(live_estimate)
        or pd.isna(adjusted_median)
        or adjusted_median == 0
    ):
        return "A benchmark comparison is not available."

    ratio = live_estimate / adjusted_median

    if ratio >= 2.0:
        return (
            "The live estimate is materially higher than the "
            "historic inflation-adjusted median. Check whether "
            "the scope, duration, procurement structure or risk "
            "profile is materially different."
        )

    if ratio <= 0.5:
        return (
            "The live estimate is materially lower than the "
            "historic inflation-adjusted median. Check whether "
            "the tender scope is narrower or potentially "
            "under-budgeted."
        )

    return (
        "The live estimate is broadly aligned with the historic "
        "inflation-adjusted median."
    )


# ============================================================
# MAIN INTELLIGENCE FUNCTION
# ============================================================

def tender_intelligence(
    description,
    live_tender_estimate=None,
    top_n=TOP_N,
    similarity_threshold=SIMILARITY_THRESHOLD,
    remove_fee_outliers=REMOVE_FEE_OUTLIERS,
    restrict_to_classification=None,
    selected_tags=None,
    require_selected_tag_match=REQUIRE_SELECTED_TAG_MATCH,
    tag_match_bonus=TAG_MATCH_BONUS,
):
    """
    Run the complete Market Behaviour Intelligence analysis.

    Returns:
        summary
        comparable_projects
        winner_summary
        classification_mix
    """

    selected_tags = normalise_selected_tags(
        selected_tags
    )

    similar = find_similar_projects(
        description=description,
        top_n=top_n,
        only_benchmark_eligible=True,
        restrict_to_classification=(
            restrict_to_classification
        ),
        selected_tags=selected_tags,
        require_selected_tag_match=(
            require_selected_tag_match
        ),
        tag_match_bonus=tag_match_bonus,
    )

    similar = similar[
        similar["Similarity"] >= similarity_threshold
    ].copy()

    before_outlier_count = len(similar)

    if remove_fee_outliers and not similar.empty:
        similar = remove_award_value_outliers(
            similar
        )

    after_outlier_count = len(similar)

    behaviour_records = similar[
        similar["Use For Behaviour Analysis"]
    ].copy()

    low_win = behaviour_records[
        behaviour_records[
            "Winner Premium Over Low"
        ].notna()
    ].copy()

    positive_premium = low_win[
        low_win["Winner Premium Over Low"]
        > PREMIUM_ZERO_TOLERANCE
    ].copy()

    won_at_low = low_win[
        low_win["Winner Premium Over Low"].abs()
        <= PREMIUM_ZERO_TOLERANCE
    ].copy()

    win_high = behaviour_records[
        behaviour_records[
            "Winner Discount From High"
        ].notna()
    ].copy()

    full_range = behaviour_records[
        behaviour_records["Analysis Type"].eq(
            "Full bid range"
        )
    ].copy()

    (
        confidence,
        average_similarity,
        top_similarity,
    ) = calculate_confidence(similar)

    if not similar.empty:
        similar["Fee Weight"] = similar["Final Score"]
    else:
        similar["Fee Weight"] = pd.Series(dtype=float)

    similarity_weighted_fee = weighted_average(
        similar["Inflation Adjusted Awarded Value"],
        similar["Fee Weight"],
    )

    low_win["Premium Weight"] = (
        low_win["Inflation Adjusted Awarded Value"]
        * low_win["Final Score"]
    )

    positive_premium["Premium Weight"] = (
        positive_premium[
            "Inflation Adjusted Awarded Value"
        ]
        * positive_premium["Final Score"]
    )

    weighted_premium_factor_all = weighted_average(
        low_win["Winner Premium Over Low"],
        low_win["Premium Weight"],
    )

    weighted_premium_factor_positive = weighted_average(
        positive_premium["Winner Premium Over Low"],
        positive_premium["Premium Weight"],
    )

    if len(low_win) > 0:
        win_at_low_probability = (
            len(won_at_low) / len(low_win)
        )
        above_low_probability = (
            len(positive_premium) / len(low_win)
        )
    else:
        win_at_low_probability = np.nan
        above_low_probability = np.nan

    if (
        live_tender_estimate is not None
        and pd.notna(live_tender_estimate)
    ):
        price_led_indicative_fee = (
            live_tender_estimate
        )
    else:
        price_led_indicative_fee = np.nan

    if (
        live_tender_estimate is not None
        and pd.notna(live_tender_estimate)
        and pd.notna(weighted_premium_factor_all)
    ):
        blended_indicative_winner_fee = (
            live_tender_estimate
            * (1 + weighted_premium_factor_all)
        )
    else:
        blended_indicative_winner_fee = np.nan

    if (
        live_tender_estimate is not None
        and pd.notna(live_tender_estimate)
        and pd.notna(
            weighted_premium_factor_positive
        )
    ):
        premium_scenario_indicative_winner_fee = (
            live_tender_estimate
            * (
                1
                + weighted_premium_factor_positive
            )
        )
    else:
        premium_scenario_indicative_winner_fee = np.nan

    premium_stats = safe_percentiles(
        low_win["Winner Premium Over Low"]
    )

    positive_premium_stats = safe_percentiles(
        positive_premium[
            "Winner Premium Over Low"
        ]
    )

    discount_stats = safe_percentiles(
        win_high["Winner Discount From High"]
    )

    spread_stats = safe_percentiles(
        full_range["Historic Bid Spread Ratio"]
    )

    api_stats = safe_percentiles(
        full_range["Award Position Index"]
    )

    winner_summary = (
        similar["Winner"]
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(10)
    )

    classification_mix = (
        similar["Project Classification"]
        .value_counts()
        .head(10)
    )

    if similar.empty:
        adjusted_median = np.nan
    else:
        adjusted_median = similar[
            "Inflation Adjusted Awarded Value"
        ].median()

    if (
        live_tender_estimate is not None
        and pd.notna(live_tender_estimate)
        and pd.notna(adjusted_median)
        and adjusted_median != 0
    ):
        estimate_ratio = (
            live_tender_estimate
            / adjusted_median
        )
    else:
        estimate_ratio = np.nan

    estimate_message = (
        estimate_vs_benchmark_message(
            live_tender_estimate,
            adjusted_median,
        )
    )

    summary = {
        "Comparable Projects Displayed": len(similar),
        "Behaviour Valid Records": len(
            behaviour_records
        ),
        "Behaviour Excluded Records": (
            len(similar) - len(behaviour_records)
        ),
        "Before Outlier Filter": before_outlier_count,
        "After Outlier Filter": after_outlier_count,
        "Confidence": confidence,
        "Average Similarity": average_similarity,
        "Top Similarity": top_similarity,
        "Selected Tags": selected_tags,
        "Live Tender Estimate": live_tender_estimate,
        "Adjusted Winning Fee Median": (
            adjusted_median
        ),
        "Estimate to Adjusted Median Ratio": (
            estimate_ratio
        ),
        "Estimate Benchmark Message": (
            estimate_message
        ),
        "Winning Fee P25": (
            similar["Awarded Value"].quantile(0.25)
            if not similar.empty
            else np.nan
        ),
        "Winning Fee Median": (
            similar["Awarded Value"].median()
            if not similar.empty
            else np.nan
        ),
        "Winning Fee P75": (
            similar["Awarded Value"].quantile(0.75)
            if not similar.empty
            else np.nan
        ),
        "Adjusted Winning Fee P25": (
            similar[
                "Inflation Adjusted Awarded Value"
            ].quantile(0.25)
            if not similar.empty
            else np.nan
        ),
        "Adjusted Winning Fee P75": (
            similar[
                "Inflation Adjusted Awarded Value"
            ].quantile(0.75)
            if not similar.empty
            else np.nan
        ),
        "Similarity Weighted Adjusted Fee": (
            similarity_weighted_fee
        ),
        "Low and Win Records Used": len(low_win),
        "Won At Low Records Used": len(won_at_low),
        "Won Above Low Records Used": len(
            positive_premium
        ),
        "Win At Low Probability": (
            win_at_low_probability
        ),
        "Above Low Probability": (
            above_low_probability
        ),
        "Winner Premium Over Low Median": (
            premium_stats["Median"]
        ),
        "Positive Premium Median": (
            positive_premium_stats["Median"]
        ),
        "Weighted Premium Factor All Low-Win Records": (
            weighted_premium_factor_all
        ),
        "Weighted Positive Premium Factor": (
            weighted_premium_factor_positive
        ),
        "Price Led Indicative Fee": (
            price_led_indicative_fee
        ),
        "Blended Indicative Winner Fee": (
            blended_indicative_winner_fee
        ),
        "Premium Scenario Indicative Winner Fee": (
            premium_scenario_indicative_winner_fee
        ),
        "Win and High Records Used": len(win_high),
        "Winner Discount From High Median": (
            discount_stats["Median"]
        ),
        "Full Bid Range Records Used": len(
            full_range
        ),
        "Historic Bid Spread Ratio Median": (
            spread_stats["Median"]
        ),
        "Award Position Index Median": (
            api_stats["Median"]
        ),
    }

    display_columns = [
        "Organisation Name",
        "Title",
        "Description",
        "Winner",
        "Project Classification",
        "Project Tags",
        "Commercial Data Quality",
        "Use For Behaviour Analysis",
        "Awarded Value",
        "Inflation Adjusted Awarded Value",
        "Value Low",
        "Value High",
        "Analysis Type",
        "Winner Premium Over Low",
        "Winner Discount From High",
        "Historic Bid Spread Ratio",
        "Award Position Index",
        "Similarity",
        "Tag Bonus",
        "Final Score",
    ]

    comparable_projects = similar[
        display_columns
    ].copy()

    return (
        summary,
        comparable_projects,
        winner_summary,
        classification_mix,
    )
