import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

FILE = "notices3_shortlisted_sorted.xlsx"

def clean_text(text):
    text = str(text)
    text = text.replace("_x000D_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

# Load dataset
df = pd.read_excel(FILE, engine="openpyxl")

for col in ["Value Low", "Value High", "Awarded Value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["Organisation Name", "Title", "Description", "Winner"]:
    df[col] = df[col].fillna("").astype(str)

df["Search_Text"] = (
    df["Organisation Name"].apply(clean_text)
    + " "
    + df["Title"].apply(clean_text)
    + " "
    + df["Description"].apply(clean_text)
)

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
]

all_stop_words = list(
    ENGLISH_STOP_WORDS.union(custom_stop_words)
)

vectorizer = TfidfVectorizer(
    stop_words=all_stop_words,
    ngram_range=(1, 3),
    max_features=20000,
    min_df=2
)

historic_vectors = vectorizer.fit_transform(
    df["Search_Text"]
)

def find_similar_projects(
    description,
    top_n=20
):

    query = clean_text(description)

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        historic_vectors
    )[0]

    results = df.copy()

    results["Similarity"] = similarities

    results = results.sort_values(
        "Similarity",
        ascending=False
    )

    return results.head(top_n)

def run_market_analysis(
    description
):

    matches = find_similar_projects(
        description
    )

    fees = matches["Awarded Value"].dropna()

    result = {
        "confidence":
            "High" if len(matches) > 10 else "Low",

        "fee_p25":
            fees.quantile(0.25),

        "fee_median":
            fees.median(),

        "fee_p75":
            fees.quantile(0.75),

        "comparables":
            matches[
                [
                    "Organisation Name",
                    "Title",
                    "Awarded Value",
                    "Similarity",
                ]
            ]
    }

    return result
