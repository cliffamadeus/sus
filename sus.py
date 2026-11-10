import pandas as pd
from google.colab import files
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt


def visualize_sus_scores(df, label, filename="sus_score_distribution.png"):
    """
    Plot a histogram + boxplot of SUS_Score with the standard SUS
    benchmark lines (50.9 = marginally acceptable, 68 = above average).
    Saves the figure to `filename` and downloads it.
    """
    if 'SUS_Score' not in df.columns or len(df) == 0:
        print(f"[skip] No SUS scores to plot for '{label}'.")
        return

    scores = df['SUS_Score'].dropna()
    if len(scores) == 0:
        print(f"[skip] SUS_Score column is empty for '{label}'.")
        return

    mean_sus = scores.mean()

    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )

    # --- Histogram ---
    ax_hist.hist(scores, bins=10, color='steelblue',
                 edgecolor='black', alpha=0.85)
    ax_hist.axvline(50.9, color='orange', linestyle='--', linewidth=1.5,
                    label='Marginal (50.9)')
    ax_hist.axvline(68, color='green', linestyle='--', linewidth=1.5,
                    label='Above avg (68)')
    ax_hist.axvline(mean_sus, color='red', linestyle='-', linewidth=2,
                    label=f'Mean ({mean_sus:.1f})')
    ax_hist.set_ylabel('Respondents')
    ax_hist.set_title(f'SUS Score Distribution — {label} (n={len(scores)})')
    ax_hist.legend(loc='upper left', fontsize=9)

    # --- Boxplot ---
    ax_box.boxplot(scores, vert=False, widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor='lightsteelblue'))
    ax_box.axvline(50.9, color='orange', linestyle='--', linewidth=1.5)
    ax_box.axvline(68, color='green', linestyle='--', linewidth=1.5)
    ax_box.axvline(mean_sus, color='red', linestyle='-', linewidth=2)
    ax_box.set_yticks([])
    ax_box.set_xlabel('SUS Score (0–100)')
    ax_box.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    files.download(filename)
    print(f"Saved and downloaded figure: '{filename}'")


def compute_and_clean_sus(df):
    """
    Compute SUS scores, clean neutral responses, and show stats before/after cleaning.
    Expects df to have exactly 10 SUS question columns.
    """
    if df.shape[1] != 10:
        raise ValueError(
            f"Data must have exactly 10 columns (one per SUS question). "
            f"Got {df.shape[1]} columns: {list(df.columns)}"
        )

    df = df.copy()

    # STEP 1 — Detect lazy responses
    lazy_mask = df.nunique(axis=1) == 1
    lazy_responses = df[lazy_mask].copy()
    cleaned_df = df[~lazy_mask].copy()

    positive_items = [1, 3, 5, 7, 9]
    negative_items = [2, 4, 6, 8, 10]

    # STEP 2 — SUS scoring adjustments
    adjusted = cleaned_df.copy()
    for i in positive_items:
        adjusted.iloc[:, i - 1] = cleaned_df.iloc[:, i - 1] - 1
    for i in negative_items:
        adjusted.iloc[:, i - 1] = 5 - cleaned_df.iloc[:, i - 1]

    # STEP 3 — Per-respondent SUS score
    cleaned_df['SUS_Score'] = adjusted.sum(axis=1) * 2.5

    # STEP 4 — Summary stats
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

    print("===== SYSTEM USABILITY SCALE (SUS) ANALYSIS =====")
    print(f"Total responses loaded : {len(df)}")
    print(f"Neutral (uniform) responses removed : {len(lazy_responses)}")

    # Lazy group SUS (for comparison)
    if len(lazy_responses) > 0:
        lazy_adj = lazy_responses.copy()
        for i in positive_items:
            lazy_adj.iloc[:, i - 1] = lazy_responses.iloc[:, i - 1] - 1
        for i in negative_items:
            lazy_adj.iloc[:, i - 1] = 5 - lazy_responses.iloc[:, i - 1]
        lazy_responses['SUS_Score'] = lazy_adj.sum(axis=1) * 2.5
        summarize_sus(lazy_responses, "Neutral Response Group")

    summarize_sus(cleaned_df, "Cleaned Dataset (Adjusted)")

    # STEP 5 — Visualize (cleaned dataset + lazy group if any)
    visualize_sus_scores(cleaned_df, "Cleaned Dataset")
    if len(lazy_responses) > 0:
        visualize_sus_scores(
            lazy_responses, "Neutral Response Group",
            filename="sus_score_distribution_lazy.png",
        )

    # STEP 6 — Save results
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

df = pd.read_csv(StringIO(uploaded[csv_file].decode('utf-8')))
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
sus_cols = [f"Q{i}" for i in range(1, 11)]
df = df[sus_cols]

cleaned, lazy = compute_and_clean_sus(df)