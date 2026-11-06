import pandas as pd
from google.colab import files
from io import StringIO
import numpy as np

def compute_and_clean_sus(df):
    """
    Compute SUS scores, clean neutral responses, and show stats before/after cleaning.
    """
    # Ensure correct format
    if df.shape[1] != 10:
        raise ValueError("Data must have exactly 10 columns (one per SUS question).")

    # Copy original data for comparison
    original_df = df.copy()

    # STEP 1 — Detect lazy responses (same value across all 10 questions)
    lazy_mask = df.nunique(axis=1) == 1
    lazy_responses = df[lazy_mask]
    cleaned_df = df[~lazy_mask]

    # STEP 2 — Apply SUS scoring adjustments
    positive_items = [1, 3, 5, 7, 9]
    negative_items = [2, 4, 6, 8, 10]
    adjusted = cleaned_df.copy()

    for i in positive_items:
        adjusted.iloc[:, i - 1] = cleaned_df.iloc[:, i - 1] - 1
    for i in negative_items:
        adjusted.iloc[:, i - 1] = 5 - cleaned_df.iloc[:, i - 1]

    # STEP 3 — Compute SUS score per respondent
    cleaned_df['SUS_Score'] = adjusted.sum(axis=1) * 2.5

    # STEP 4 — Compute statistics for both datasets
    def summarize_sus(df_subset, label):
        mean_sus = df_subset['SUS_Score'].mean()
        std_sus = df_subset['SUS_Score'].std()
        n = len(df_subset)
        moe = 1.96 * (std_sus / np.sqrt(n)) if n > 1 else 0
        error_percent = (moe / mean_sus) * 100 if mean_sus != 0 else 0

        print(f"\n===== {label} =====")
        print(f"Number of respondents : {n}")
        print(f"Mean SUS Score        : {mean_sus:.2f}")
        print(f"Standard Deviation    : {std_sus:.2f}")
        print(f"Margin of Error (95%) : ±{moe:.2f}")
        print(f"Error %               : {error_percent:.2f}%")
        return mean_sus, error_percent

    # Compute stats
    print("===== SYSTEM USABILITY SCALE (SUS) ANALYSIS =====")
    print(f"Total responses loaded : {len(original_df)}")
    print(f"Neutral (uniform) responses removed : {len(lazy_responses)}")

    # Compute lazy group SUS (for comparison)
    if len(lazy_responses) > 0:
        lazy_adj = lazy_responses.copy()
        for i in positive_items:
            lazy_adj.iloc[:, i - 1] = lazy_responses.iloc[:, i - 1] - 1
        for i in negative_items:
            lazy_adj.iloc[:, i - 1] = 5 - lazy_responses.iloc[:, i - 1]
        lazy_responses['SUS_Score'] = lazy_adj.sum(axis=1) * 2.5
        summarize_sus(lazy_responses, "Neutral Response Group")

    summarize_sus(cleaned_df, "Cleaned Dataset (Adjusted)")

    # STEP 5 — Save results
    results_file = "cleaned_sus_results.csv"
    cleaned_df[['SUS_Score']].to_csv(results_file, index=False)
    files.download(results_file)
    print(f"\nCleaned SUS results saved as '{results_file}'")

    return cleaned_df, lazy_responses


# -----------------------------
# Upload CSV in Google Colab
# -----------------------------
uploaded = files.upload()
csv_file = list(uploaded.keys())[0]
df = pd.read_csv(StringIO(uploaded[csv_file].decode('utf-8')), header=None)

# Run analysis
cleaned, lazy = compute_and_clean_sus(df)
