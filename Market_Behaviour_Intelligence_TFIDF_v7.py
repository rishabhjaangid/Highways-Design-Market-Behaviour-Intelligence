# %% [markdown]
# # Market Behaviour Intelligence Engine v7
# 
# This version adds commercial data-quality controls. Comparable projects are still displayed, but records where `Low = Awarded = High` or `Awarded = High` are excluded from behaviour calculations. `Awarded = Low` is retained because it may show genuine price-led procurement behaviour.

# %%
# Cell 1: Import required Python libraries
import pandas as pd
import numpy as np
import re
import warnings
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_colwidth", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
warnings.filterwarnings("default", category=pd.errors.PerformanceWarning)
print("Libraries loaded successfully")

# %%
# Cell 2: Set user-editable model configuration
FILE = "notices3_shortlisted_sorted.xlsx"
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
print("Configuration set")

# %%
# Cell 3: Load the historic tender spreadsheet safely
try:
    df = pd.read_excel(FILE, engine="openpyxl")
    print("Spreadsheet loaded successfully")
    print("Rows and columns:", df.shape)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {FILE}. Put the spreadsheet in the same folder as this notebook.")
except PermissionError:
    raise PermissionError(f"Windows is denying access to {FILE}. Close the workbook in Excel or save a local copy and try again.")
df.head()

# %%
# Cell 4: Check required columns exist
required_columns = ["Organisation Name", "Title", "Description", "Value Low", "Value High", "Awarded Date", "Awarded Value", "Winner"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
print("All required columns found")
print(df.columns.tolist())

# %%
# Cell 5: Clean numeric and text fields
for col in ["Value Low", "Value High", "Awarded Value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in ["Organisation Name", "Title", "Description", "Winner"]:
    df[col] = df[col].fillna("").astype(str)
df = df.copy()
print("Numeric and text fields cleaned")
df[["Value Low", "Value High", "Awarded Value"]].describe()

# %%
# Cell 6: Convert awarded date into award year and project age
df["Awarded Date"] = pd.to_datetime(df["Awarded Date"], dayfirst=True, errors="coerce")
df["Award Year"] = df["Awarded Date"].dt.year
df["Project Age"] = CURRENT_YEAR - df["Award Year"]
df.loc[df["Project Age"].lt(0), "Project Age"] = np.nan
print("Award year and project age created")
df[["Awarded Date", "Award Year", "Project Age"]].head()

# %%
# Cell 7: Apply configurable inflation uplift to awarded values
df["Inflation Rate Used"] = DEFAULT_ANNUAL_INFLATION_RATE
df["Inflation Factor"] = np.where(df["Project Age"].notna(), (1 + DEFAULT_ANNUAL_INFLATION_RATE) ** df["Project Age"], np.nan)
df["Inflation Adjusted Awarded Value"] = np.where(df["Inflation Factor"].notna(), df["Awarded Value"] * df["Inflation Factor"], df["Awarded Value"])
print("Inflation-adjusted awarded value created")
df[["Awarded Value", "Award Year", "Project Age", "Inflation Adjusted Awarded Value"]].head()

# %%
# Cell 8: Classify what commercial evidence is available per record
low = df["Value Low"]
high = df["Value High"]
win = df["Awarded Value"]
has_low = low.notna() & low.gt(0)
has_high = high.notna() & high.gt(0)
has_win = win.notna() & win.gt(0)
full_range = has_low & has_high & has_win & high.gt(low)
low_win = has_low & has_win
win_high = has_high & has_win
conditions = [full_range, low_win & ~full_range, win_high & ~full_range, df["Value Low"].eq(0), has_win]
choices = ["Full bid range", "Low and win only", "Win and high only", "Zero-low record", "Awarded only"]
df["Analysis Type"] = np.select(conditions, choices, default="Data issue or missing awarded value")
df = df.copy()
print("Analysis type created")
print(df["Analysis Type"].value_counts(dropna=False))

# %%
# Cell 9: Calculate winner-position factors, supporting metrics and commercial data-quality flags
low = df["Value Low"]
high = df["Value High"]
win = df["Awarded Value"]
has_low_win = low.notna() & low.gt(0) & win.notna() & win.gt(0)
has_win_high = high.notna() & high.gt(0) & win.notna() & win.gt(0)
has_full_range = low.notna() & high.notna() & win.notna() & low.gt(0) & high.gt(low)

df["Winner Premium Over Low"] = np.where(has_low_win, (win - low) / low, np.nan)
df["Winner Discount From High"] = np.where(has_win_high, (high - win) / high, np.nan)
df["Market Midpoint"] = np.where(has_full_range, (low + high) / 2, np.nan)
df["Bid Spread"] = np.where(has_full_range, high - low, np.nan)
df["Historic Bid Spread Ratio"] = np.where(has_full_range, (high - low) / low, np.nan)
df["Market Uncertainty Index"] = np.where(has_full_range, df["Bid Spread"] / df["Market Midpoint"], np.nan)
df["Market Consensus Score"] = np.where(has_full_range, np.maximum(0, 100 - (df["Market Uncertainty Index"] * 100)), np.nan)
df["Award Position Index"] = np.where(has_full_range, (win - low) / (high - low), np.nan)
df["Midpoint Deviation %"] = np.where(has_full_range, (win - df["Market Midpoint"]) / df["Market Midpoint"], np.nan)

# Commercial data quality. Awarded = Low is retained. Awarded = High and Low = Awarded = High are excluded from behaviour calculations.
df["Commercial Data Quality"] = "Valid"
same_low_awarded_high = df["Value Low"].notna() & df["Awarded Value"].notna() & df["Value High"].notna() & df["Value Low"].eq(df["Awarded Value"]) & df["Awarded Value"].eq(df["Value High"])
df.loc[same_low_awarded_high, "Commercial Data Quality"] = "Low = Awarded = High"
awarded_equals_high = df["Awarded Value"].notna() & df["Value High"].notna() & df["Awarded Value"].eq(df["Value High"]) & ~same_low_awarded_high
df.loc[awarded_equals_high, "Commercial Data Quality"] = "Awarded = High"
awarded_equals_low = df["Awarded Value"].notna() & df["Value Low"].notna() & df["Awarded Value"].eq(df["Value Low"]) & df["Commercial Data Quality"].eq("Valid")
df.loc[awarded_equals_low, "Commercial Data Quality"] = "Awarded = Low"

df["Use For Behaviour Analysis"] = True
df.loc[df["Commercial Data Quality"].isin(["Low = Awarded = High", "Awarded = High"]), "Use For Behaviour Analysis"] = False
df = df.copy()
print("Winner-position factors, supporting metrics and commercial data-quality flags created")
print(df["Commercial Data Quality"].value_counts(dropna=False))
df[["Analysis Type", "Commercial Data Quality", "Use For Behaviour Analysis", "Winner Premium Over Low", "Winner Discount From High", "Historic Bid Spread Ratio", "Award Position Index"]].head()

# %%
# Cell 10: Clean and prepare text for classification and TF-IDF matching
def clean_text(text):
    text = str(text)
    text = text.replace("_x000D_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

df["Search_Text"] = df["Organisation Name"].apply(clean_text) + " " + df["Title"].apply(clean_text) + " " + df["Description"].apply(clean_text) + " " + df["Description"].apply(clean_text)
print("Search text created")
df[["Title", "Search_Text"]].head()

# %%
# Cell 11: Create rich primary project classifications from title and description
def classify_project(text):
    text = str(text).lower()
    if any(t in text for t in ["framework", "framework agreement", "professional services partnership", "term arrangement", "term consultancy", "panel"]): return "Framework / Term Consultancy"
    if any(t in text for t in ["construction", "demolition", "scaffolding", "supply and installation", "installation of", "replacement of lighting", "new build", "building works"]): return "Construction / Physical Works"
    if any(t in text for t in ["business case", "outline business case", "obc", "fbc", "sopc", "economic assessment", "funding approval", "technical evaluator", "local growth fund"]): return "Business Case / Funding"
    if any(t in text for t in ["technical partner", "programme support", "programme director", "mobilisation", "portfolio", "governance"]): return "Strategic / Technical Partner"
    if any(t in text for t in ["independent technical advisor", "technical advisor", "assurance", "governance", "review of detailed design", "technical evaluation"]): return "Assurance / Technical Advisor"
    if any(t in text for t in ["road condition monitoring", "rcm", "scrim", "scanner", "ukpms", "pavement survey", "condition survey", "network survey"]): return "Road Condition Monitoring / Asset Survey"
    if any(t in text for t in ["traffic signal", "signals", "signal scheme", "ibus", "vehicle detection", "hadecs", "enforcement", "its"]): return "Traffic Signals / ITS"
    if any(t in text for t in ["traffic modelling", "transport modelling", "traffic model", "transport model", "modelling", "model update"]): return "Traffic / Transport Modelling"
    if any(t in text for t in ["active travel", "cycling", "cycleway", "cycle route", "walking", "pedestrian", "lcwip"]): return "Active Travel / Walking / Cycling"
    if any(t in text for t in ["bus priority", "bsip", "bus service improvement", "bus route", "public transport"]): return "Bus Priority / Public Transport"
    if any(t in text for t in ["public realm", "urban realm", "town centre", "city centre", "streetscape", "event spaces"]): return "Public Realm / Urban Realm"
    if any(t in text for t in ["detailed design", "design changes", "as-built drawings", "construction support", "buildability"]): return "Detailed Design"
    if any(t in text for t in ["preliminary design", "concept design", "option appraisal", "options appraisal", "feasibility", "study"]): return "Preliminary / Concept / Feasibility"
    if any(t in text for t in ["junction", "corridor", "road improvement", "highway improvement", "carriageway", "roundabout"]): return "Junction / Corridor Improvement"
    if any(t in text for t in ["bridge", "viaduct", "retaining wall", "culvert", "structure", "archway"]): return "Structures / Bridges"
    if any(t in text for t in ["drainage", "flood", "surface water", "suds", "attenuation", "watercourse"]): return "Drainage / Flood / Water"
    if any(t in text for t in ["asset valuation", "asset management", "mrm", "survey"]): return "Asset / Survey / Data Collection"
    if any(t in text for t in ["housing", "homes", "building", "redevelopment", "cemetery"]): return "Building / Non-Highways Development"
    return "Other"

df["Project Classification"] = df["Search_Text"].apply(classify_project)
print("Project classifications created")
df["Project Classification"].value_counts().head(30)

# %%
# Cell 12: Create secondary project tags for explainability
tag_dictionary = {
    "active travel": ["active travel", "cycling", "cycleway", "cycle route", "walking", "pedestrian", "lcwip"],
    "detailed design": ["detailed design", "as-built", "design changes"],
    "preliminary design": ["preliminary design", "concept design"],
    "business case": ["business case", "obc", "fbc", "funding"],
    "traffic modelling": ["traffic modelling", "transport modelling", "traffic model", "modelling"],
    "traffic signals": ["traffic signal", "signals", "ibus", "vehicle detection"],
    "bus priority": ["bus priority", "bsip", "bus service improvement"],
    "junction": ["junction", "roundabout"],
    "corridor": ["corridor"],
    "public realm": ["public realm", "urban realm", "town centre", "city centre"],
    "structures": ["bridge", "viaduct", "culvert", "retaining wall", "structure"],
    "drainage": ["drainage", "flood", "surface water", "suds"],
    "framework": ["framework", "panel", "term arrangement"],
    "assurance": ["assurance", "technical advisor", "technical evaluator", "governance"],
    "survey": ["survey", "surveys", "condition survey", "network survey", "pavement survey", "scrim", "scanner", "mrm", "ukpms"],
    "road condition monitoring": ["road condition monitoring", "rcm", "scrim", "scanner", "ukpms", "mrm", "condition survey", "network survey", "pavement survey"],
    "asset management": ["asset management", "road condition", "network condition", "ukpms", "scrim", "scanner", "rcm"],
    "construction": ["construction", "demolition", "scaffolding", "installation", "new build"],
}

def create_tags(text):
    text = str(text).lower()
    tags = []
    for tag, keywords in tag_dictionary.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return ", ".join(tags) if tags else "none"

df["Project Tags"] = df["Search_Text"].apply(create_tags)
print("Project tags created")
df[["Title", "Project Classification", "Project Tags"]].head(20)

# %%
# Cell 13: Create benchmark eligibility flags to control which records enter the benchmark pool
df["Is Framework"] = df["Project Classification"].eq("Framework / Term Consultancy")
df["Is Construction Work"] = df["Project Classification"].eq("Construction / Physical Works")
df["Is Non Highways Building"] = df["Project Classification"].eq("Building / Non-Highways Development")
df["Is Very Large Award"] = df["Awarded Value"] > MAX_AWARD_VALUE

df["Benchmark Eligible"] = True
if EXCLUDE_FRAMEWORKS: df.loc[df["Is Framework"], "Benchmark Eligible"] = False
if EXCLUDE_CONSTRUCTION_WORKS: df.loc[df["Is Construction Work"], "Benchmark Eligible"] = False
if EXCLUDE_NON_HIGHWAYS_BUILDING: df.loc[df["Is Non Highways Building"], "Benchmark Eligible"] = False
if EXCLUDE_VERY_LARGE_AWARDS: df.loc[df["Is Very Large Award"], "Benchmark Eligible"] = False

df = df.copy()
print("Benchmark eligibility created")
print(df["Benchmark Eligible"].value_counts())

# %%
# Cell 14: Build the TF-IDF model using English and custom procurement stop words
custom_stop_words = ["services", "service", "contract", "project", "scheme", "support", "consultancy", "consultant", "council", "authority", "framework", "award", "procurement", "provide", "delivery", "works", "lot", "stage", "phase", "including", "required", "appoint", "seeking", "undertake"]
all_stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words))
vectorizer = TfidfVectorizer(stop_words=all_stop_words, ngram_range=(1, 3), max_features=20000, min_df=2)
historic_vectors = vectorizer.fit_transform(df["Search_Text"])
print("Historic TF-IDF vectors created:", historic_vectors.shape)

# %%
# Cell 15: Define the function that finds similar historic projects with tag-weighted ranking
def normalise_selected_tags(selected_tags):
    if selected_tags is None:
        return []
    return [str(tag).strip().lower() for tag in selected_tags if str(tag).strip()]

def count_tag_matches(project_tags, project_classification, selected_tags):
    text = f"{project_tags} {project_classification}".lower()
    return sum(tag in text for tag in selected_tags)

def find_similar_projects(description, top_n=TOP_N, only_benchmark_eligible=True, restrict_to_classification=None, selected_tags=None, require_selected_tag_match=REQUIRE_SELECTED_TAG_MATCH, tag_match_bonus=TAG_MATCH_BONUS):
    selected_tags = normalise_selected_tags(selected_tags)
    query_text = clean_text(description)
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, historic_vectors)[0]
    results = df.copy()
    results["Similarity"] = similarities
    if only_benchmark_eligible:
        results = results[results["Benchmark Eligible"] == True]
    if restrict_to_classification is not None:
        results = results[results["Project Classification"].eq(restrict_to_classification)]
    if selected_tags:
        results["Tag Bonus"] = results.apply(lambda row: count_tag_matches(row.get("Project Tags", ""), row.get("Project Classification", ""), selected_tags), axis=1)
    else:
        results["Tag Bonus"] = 0
    if selected_tags and require_selected_tag_match:
        results = results[results["Tag Bonus"] > 0]
    results["Final Score"] = results["Similarity"] + (tag_match_bonus * results["Tag Bonus"])
    results = results.sort_values("Final Score", ascending=False)
    return results.head(top_n)

print("find_similar_projects function with tag-weighted ranking created")

# %%
# Cell 16: Define helper functions for outliers, confidence, weighting and formatting
def remove_award_value_outliers(data, value_column="Inflation Adjusted Awarded Value"):
    if data.empty or data[value_column].dropna().shape[0] < 4:
        return data
    q1 = data[value_column].quantile(0.25)
    q3 = data[value_column].quantile(0.75)
    iqr = q3 - q1
    return data[(data[value_column] >= q1 - 1.5 * iqr) & (data[value_column] <= q3 + 1.5 * iqr)]

def calculate_confidence(similar_projects):
    if similar_projects.empty:
        return "No evidence", 0, 0
    match_count = len(similar_projects)
    average_similarity = similar_projects["Similarity"].mean()
    top_similarity = similar_projects["Similarity"].max()
    fee_cv = similar_projects["Inflation Adjusted Awarded Value"].std() / similar_projects["Inflation Adjusted Awarded Value"].mean()
    if match_count >= 20 and average_similarity >= 0.16 and top_similarity >= 0.25 and fee_cv <= 1.0:
        confidence = "High"
    elif match_count >= 10 and average_similarity >= 0.10:
        confidence = "Medium"
    else:
        confidence = "Low"
    return confidence, average_similarity, top_similarity

def weighted_average(values, weights):
    valid = values.notna() & weights.notna() & weights.gt(0)
    if valid.sum() == 0:
        return np.nan
    return np.average(values[valid], weights=weights[valid])

def safe_percentiles(series):
    clean = series.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"P25": np.nan, "Median": np.nan, "P75": np.nan}
    return {"P25": clean.quantile(0.25), "Median": clean.median(), "P75": clean.quantile(0.75)}

def estimate_vs_benchmark_message(live_estimate, adjusted_median):
    if pd.isna(live_estimate) or pd.isna(adjusted_median) or adjusted_median == 0:
        return "Not available"
    ratio = live_estimate / adjusted_median
    if ratio >= 2.0:
        return "Live estimate is materially higher than historic adjusted median. Check whether scope, duration, framework structure or risk profile differs."
    if ratio <= 0.5:
        return "Live estimate is materially lower than historic adjusted median. Check whether tender is narrower or under-budgeted."
    return "Live estimate is broadly aligned with historic adjusted median."

def fmt_currency(value):
    if pd.isna(value): return "Not available"
    return f"GBP {value:,.0f}"

def fmt_percent(value):
    if pd.isna(value): return "Not available"
    return f"{value:.2%}"

def fmt_number(value):
    if pd.isna(value): return "Not available"
    return f"{value:.2f}"

print("Helper functions created")

# %%
# Cell 17: Define the main Market Behaviour Intelligence function with data-quality exclusions from calculations
def tender_intelligence(description, live_tender_estimate=None, top_n=TOP_N, similarity_threshold=SIMILARITY_THRESHOLD, remove_fee_outliers=REMOVE_FEE_OUTLIERS, restrict_to_classification=None, selected_tags=None, require_selected_tag_match=REQUIRE_SELECTED_TAG_MATCH, tag_match_bonus=TAG_MATCH_BONUS):
    similar = find_similar_projects(description, top_n=top_n, only_benchmark_eligible=True, restrict_to_classification=restrict_to_classification, selected_tags=selected_tags, require_selected_tag_match=require_selected_tag_match, tag_match_bonus=tag_match_bonus)
    similar = similar[similar["Similarity"] >= similarity_threshold].copy()
    before_outlier_count = len(similar)
    if remove_fee_outliers:
        similar = remove_award_value_outliers(similar).copy()
    after_outlier_count = len(similar)

    behaviour_records = similar[similar["Use For Behaviour Analysis"] == True].copy()
    low_win = behaviour_records[behaviour_records["Winner Premium Over Low"].notna()].copy()
    positive_premium = low_win[low_win["Winner Premium Over Low"] > PREMIUM_ZERO_TOLERANCE].copy()
    won_at_low = low_win[low_win["Winner Premium Over Low"].abs() <= PREMIUM_ZERO_TOLERANCE].copy()
    win_high = behaviour_records[behaviour_records["Winner Discount From High"].notna()].copy()
    full_range = behaviour_records[behaviour_records["Analysis Type"].eq("Full bid range")].copy()

    confidence, avg_similarity, top_similarity = calculate_confidence(similar)
    similar["Fee Weight"] = similar["Final Score"]
    similarity_weighted_fee = weighted_average(similar["Inflation Adjusted Awarded Value"], similar["Fee Weight"])

    low_win["Premium Weight"] = low_win["Inflation Adjusted Awarded Value"] * low_win["Final Score"]
    positive_premium["Premium Weight"] = positive_premium["Inflation Adjusted Awarded Value"] * positive_premium["Final Score"]
    weighted_premium_factor_all = weighted_average(low_win["Winner Premium Over Low"], low_win["Premium Weight"])
    weighted_premium_factor_positive = weighted_average(positive_premium["Winner Premium Over Low"], positive_premium["Premium Weight"])
    win_at_low_probability = len(won_at_low) / len(low_win) if len(low_win) else np.nan
    above_low_probability = len(positive_premium) / len(low_win) if len(low_win) else np.nan

    price_led_indicative_fee = live_tender_estimate if live_tender_estimate is not None else np.nan
    blended_indicative_winner_fee = live_tender_estimate * (1 + weighted_premium_factor_all) if live_tender_estimate is not None and pd.notna(weighted_premium_factor_all) else np.nan
    premium_scenario_indicative_winner_fee = live_tender_estimate * (1 + weighted_premium_factor_positive) if live_tender_estimate is not None and pd.notna(weighted_premium_factor_positive) else np.nan

    premium_stats = safe_percentiles(low_win["Winner Premium Over Low"])
    positive_premium_stats = safe_percentiles(positive_premium["Winner Premium Over Low"])
    discount_stats = safe_percentiles(win_high["Winner Discount From High"])
    spread_stats = safe_percentiles(full_range["Historic Bid Spread Ratio"])
    api_stats = safe_percentiles(full_range["Award Position Index"])

    winner_summary = similar["Winner"].replace("", np.nan).dropna().value_counts().head(10)
    classification_mix = similar["Project Classification"].value_counts().head(10)
    data_quality_summary = similar["Commercial Data Quality"].value_counts(dropna=False)
    behaviour_use_summary = similar["Use For Behaviour Analysis"].value_counts(dropna=False)

    adjusted_median = similar["Inflation Adjusted Awarded Value"].median() if len(similar) else np.nan
    estimate_ratio = live_tender_estimate / adjusted_median if live_tender_estimate is not None and pd.notna(adjusted_median) and adjusted_median != 0 else np.nan
    estimate_message = estimate_vs_benchmark_message(live_tender_estimate, adjusted_median)

    summary = {
        "Comparable Projects Displayed": len(similar),
        "Behaviour Valid Records": len(behaviour_records),
        "Behaviour Excluded Records": len(similar) - len(behaviour_records),
        "Before Outlier Filter": before_outlier_count,
        "After Outlier Filter": after_outlier_count,
        "Confidence": confidence,
        "Average Similarity": avg_similarity,
        "Top Similarity": top_similarity,
        "Selected Tags": selected_tags,
        "Live Tender Estimate": live_tender_estimate,
        "Adjusted Winning Fee Median": adjusted_median,
        "Estimate to Adjusted Median Ratio": estimate_ratio,
        "Estimate Benchmark Message": estimate_message,
        "Winning Fee P25": similar["Awarded Value"].quantile(0.25) if len(similar) else np.nan,
        "Winning Fee Median": similar["Awarded Value"].median() if len(similar) else np.nan,
        "Winning Fee P75": similar["Awarded Value"].quantile(0.75) if len(similar) else np.nan,
        "Adjusted Winning Fee P25": similar["Inflation Adjusted Awarded Value"].quantile(0.25) if len(similar) else np.nan,
        "Adjusted Winning Fee P75": similar["Inflation Adjusted Awarded Value"].quantile(0.75) if len(similar) else np.nan,
        "Similarity Weighted Adjusted Fee": similarity_weighted_fee,
        "Low and Win Records Used": len(low_win),
        "Won At Low Records Used": len(won_at_low),
        "Won Above Low Records Used": len(positive_premium),
        "Win At Low Probability": win_at_low_probability,
        "Above Low Probability": above_low_probability,
        "Winner Premium Over Low Median": premium_stats["Median"],
        "Positive Premium Median": positive_premium_stats["Median"],
        "Weighted Premium Factor All Low-Win Records": weighted_premium_factor_all,
        "Weighted Positive Premium Factor": weighted_premium_factor_positive,
        "Price Led Indicative Fee": price_led_indicative_fee,
        "Blended Indicative Winner Fee": blended_indicative_winner_fee,
        "Premium Scenario Indicative Winner Fee": premium_scenario_indicative_winner_fee,
        "Win and High Records Used": len(win_high),
        "Winner Discount From High Median": discount_stats["Median"],
        "Full Bid Range Records Used": len(full_range),
        "Historic Bid Spread Ratio Median": spread_stats["Median"],
        "Award Position Index Median": api_stats["Median"],
    }

    print() ; print("=" * 70) ; print("MARKET BEHAVIOUR INTELLIGENCE") ; print("=" * 70)
    print(f"Comparable projects displayed: {summary['Comparable Projects Displayed']}")
    print(f"Behaviour-valid records used in calculations: {summary['Behaviour Valid Records']}")
    print(f"Behaviour-excluded records still shown: {summary['Behaviour Excluded Records']}")
    print(f"Confidence: {summary['Confidence']}")
    print(f"Average similarity: {summary['Average Similarity']:.3f}")
    print(f"Top similarity: {summary['Top Similarity']:.3f}")
    print(f"Selected tags: {summary['Selected Tags']}")
    print() ; print("Commercial data quality in displayed comparables") ; print(data_quality_summary)
    print() ; print("Use for behaviour analysis") ; print(behaviour_use_summary)
    print() ; print("Headline winner behaviour")
    print(f"Low and win records used: {summary['Low and Win Records Used']}")
    print(f"Won at lowest bid: {summary['Won At Low Records Used']} ({fmt_percent(summary['Win At Low Probability'])})")
    print(f"Won above lowest bid: {summary['Won Above Low Records Used']} ({fmt_percent(summary['Above Low Probability'])})")
    print(f"Weighted premium factor, all low-win records: {fmt_percent(summary['Weighted Premium Factor All Low-Win Records'])}")
    print(f"Weighted positive premium factor: {fmt_percent(summary['Weighted Positive Premium Factor'])}")
    print() ; print("Indicative fee guidance from live tender estimate")
    print(f"Live tender estimate: {fmt_currency(summary['Live Tender Estimate'])}")
    print(f"Price-led indicative fee: {fmt_currency(summary['Price Led Indicative Fee'])}")
    print(f"Blended indicative winner fee: {fmt_currency(summary['Blended Indicative Winner Fee'])}")
    print(f"Premium scenario indicative winner fee: {fmt_currency(summary['Premium Scenario Indicative Winner Fee'])}")
    print(f"Estimate to historic adjusted median ratio: {fmt_number(summary['Estimate to Adjusted Median Ratio'])}")
    print(f"Benchmark check: {summary['Estimate Benchmark Message']}")
    print() ; print("Historic winning fee benchmark")
    print(f"Original P25: {fmt_currency(summary['Winning Fee P25'])}")
    print(f"Original median: {fmt_currency(summary['Winning Fee Median'])}")
    print(f"Original P75: {fmt_currency(summary['Winning Fee P75'])}")
    print(f"Adjusted P25: {fmt_currency(summary['Adjusted Winning Fee P25'])}")
    print(f"Adjusted median: {fmt_currency(summary['Adjusted Winning Fee Median'])}")
    print(f"Adjusted P75: {fmt_currency(summary['Adjusted Winning Fee P75'])}")
    print(f"Similarity weighted adjusted fee: {fmt_currency(summary['Similarity Weighted Adjusted Fee'])}")
    print() ; print("Supporting diagnostics from behaviour-valid records")
    print(f"Winner premium over low median: {fmt_percent(summary['Winner Premium Over Low Median'])}")
    print(f"Positive premium median: {fmt_percent(summary['Positive Premium Median'])}")
    print(f"Win and high records used: {summary['Win and High Records Used']}")
    print(f"Winner discount from high median: {fmt_percent(summary['Winner Discount From High Median'])}")
    print(f"Full bid-range records used: {summary['Full Bid Range Records Used']}")
    print(f"Historic bid spread ratio median: {fmt_percent(summary['Historic Bid Spread Ratio Median'])}")
    print(f"Award position index median: {fmt_number(summary['Award Position Index Median'])}")
    print() ; print("Project classification mix") ; print(classification_mix)
    print() ; print("Top historic winners among comparable projects") ; print(winner_summary)

    display_columns = ["Organisation Name", "Title", "Description", "Winner", "Project Classification", "Project Tags", "Commercial Data Quality", "Use For Behaviour Analysis", "Awarded Value", "Inflation Adjusted Awarded Value", "Value Low", "Value High", "Analysis Type", "Winner Premium Over Low", "Winner Discount From High", "Historic Bid Spread Ratio", "Award Position Index", "Similarity", "Tag Bonus", "Final Score"]
    return summary, similar[display_columns], winner_summary, classification_mix

print("Main tender_intelligence function created")

# %%
# Cell 18: Test the model with an example live tender
example_description = """
Detailed design of an active travel corridor including junction improvements,
drainage design, traffic modelling and stakeholder engagement.
"""
summary, comparable_projects, winner_summary, classification_mix = tender_intelligence(
    example_description,
    live_tender_estimate=500000,
    top_n=50,
    similarity_threshold=0.10,
    remove_fee_outliers=True,
    selected_tags=["active travel", "detailed design", "junction", "drainage", "traffic modelling"],
)
comparable_projects

# %%
# Cell 19: View the example intelligence summary as a table
summary_df = pd.DataFrame([summary]).T
summary_df.columns = ["Value"]
summary_df

# %%
# Cell 20: Enter live tender description and estimate, then suggest classification tags
print("=" * 70)
print("LIVE TENDER INPUT")
print("=" * 70)
live_description = input("Paste live tender description: ")
live_estimate_input = input("Enter client estimated value in pounds, for example 500000: ")
try:
    live_tender_estimate = float(live_estimate_input.replace(",", ""))
except ValueError:
    live_tender_estimate = None
    print("No valid estimate entered. Fee guidance based on live estimate will not be calculated.")

live_description_lower = live_description.lower()
tag_scores = Counter()
for tag, keywords in tag_dictionary.items():
    for keyword in keywords:
        if keyword in live_description_lower:
            tag_scores[tag] += 1
suggested_tags = [tag for tag, score in tag_scores.most_common()]
print() ; print("Suggested tags ranked by keyword matches:")
if suggested_tags:
    for i, (tag, score) in enumerate(tag_scores.most_common(), start=1):
        print(f"{i}. {tag} (matched {score} keywords)")
else:
    print("No tags were detected automatically. You can manually enter tags in the next cell.")
print() ; print("Available tags you can choose from:")
for tag in sorted(tag_dictionary.keys()):
    print("-", tag)

# %%
# Cell 21: Choose classification tags and run live tender market intelligence
selected_tags_input = input("Enter tags to use, separated by commas. Press ENTER to use suggested tags. Type none to use no tag filter: ")
if selected_tags_input.strip().lower() == "none":
    selected_tags = []
elif selected_tags_input.strip() == "":
    selected_tags = suggested_tags
else:
    selected_tags = [tag.strip().lower() for tag in selected_tags_input.split(",") if tag.strip()]
print() ; print("Selected tags:") ; print(selected_tags)
live_summary, live_comparable_projects, live_winner_summary, live_classification_mix = tender_intelligence(
    live_description,
    live_tender_estimate=live_tender_estimate,
    top_n=50,
    similarity_threshold=0.10,
    remove_fee_outliers=True,
    selected_tags=selected_tags,
    require_selected_tag_match=REQUIRE_SELECTED_TAG_MATCH,
    tag_match_bonus=TAG_MATCH_BONUS,
)
live_comparable_projects

# %%
# Cell 22: View the live tender intelligence summary as a table
live_summary_df = pd.DataFrame([live_summary]).T
live_summary_df.columns = ["Value"]
live_summary_df

# %%
# Cell 23: Optional export of enriched dataset and latest comparable projects to Excel
EXPORT_RESULTS = False
if EXPORT_RESULTS:
    df.to_excel("enriched_market_behaviour_dataset_v7.xlsx", index=False)
    comparable_projects.to_excel("latest_comparable_projects_v7.xlsx", index=False)
    summary_df.to_excel("latest_market_intelligence_summary_v7.xlsx")
    print("Export complete")
else:
    print("Export skipped. Set EXPORT_RESULTS = True to export files.")

# %% [markdown]
# # Cell 24: Interpretation notes
# 
# The headline logic in v7 is:
# 
# 1. Comparable projects are still displayed even when commercial data is weak.
# 2. Behaviour calculations exclude records where `Low = Awarded = High` or `Awarded = High`.
# 3. Records where `Awarded = Low` are retained because they may show genuine price-led procurement behaviour.
# 4. Win-at-low probability shows how often comparable projects were awarded at the lowest bid.
# 5. Weighted premium factor across behaviour-valid low-win records blends zero-premium and positive-premium outcomes and is used for a central indicative fee.
# 6. Weighted positive premium factor only looks at cases where the winning fee was above the lowest bid and is used for a premium scenario.
# 7. Indicative winner fee applies the weighted premium factor to the live tender estimate.


