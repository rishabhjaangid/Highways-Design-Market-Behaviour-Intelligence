import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity

FILE = "notices3_shortlisted_sorted.xlsx"


def clean_text(text):
    text = str(text)
    text = text.replace("_x000D_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_excel(
    FILE,
    engine="openpyxl"
)

# Convert numeric fields

for col in [
    "Value Low",
    "Value High",
    "Awarded Value"
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Clean text fields

for col in [
    "Organisation Name",
    "Title",
    "Description",
    "Winner"
]:
    df[col] = (
        df[col]
        .fillna("")
        .astype(str)
    )

# Create searchable text

df["Search_Text"] = (
    df["
