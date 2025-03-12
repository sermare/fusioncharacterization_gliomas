# analysis_functions.py

from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations
import statsmodels.stats.multitest as smm

from fcg.common_imports import *


# Arriba_clonal_in_sample
# ------------------------------
# Utility Functions
# ------------------------------
def create_group_label(df, group_cols=["Histology", "Tumor"]):
    """Create a combined group label from specified columns."""
    if not all(col in df.columns for col in group_cols):
        raise ValueError(f"Columns {group_cols} not found in DataFrame")
    df["Histology_Tumor"] = df[group_cols[0]].astype(str) + "-" + df[group_cols[1]].astype(str)
    return sorted(df["Histology_Tumor"].unique())

def perform_kruskal_test(df, group_col, value_col, group_order):
    """Perform Kruskal-Wallis test across groups."""
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col} not found in DataFrame")
    data_by_group = [df.loc[df[group_col] == group, value_col].dropna() for group in group_order]
    if len(data_by_group) < 2:
        raise ValueError("Need at least 2 groups for Kruskal-Wallis test")
    H, p = kruskal(*data_by_group)
    return H, p

def get_significant_pairs(df, group_col, value_col, group_order, alpha=0.05):
    """Perform pairwise Mann-Whitney U tests with FDR correction."""
    pairs = list(combinations(group_order, 2))
    p_vals, comparisons = [], []
    
    for g1, g2 in pairs:
        data1 = df.loc[df[group_col] == g1, value_col].dropna()
        data2 = df.loc[df[group_col] == g2, value_col].dropna()
        if len(data1) > 0 and len(data2) > 0:
            _, p_val = mannwhitneyu(data1, data2, alternative="two-sided")
            p_vals.append(p_val)
            comparisons.append((g1, g2))
    
    if not p_vals:
        return []
    reject, pvals_corrected, _, _ = smm.multipletests(p_vals, alpha=alpha, method="fdr_bh")
    return [(g1, g2, p) for (g1, g2), p, r in zip(comparisons, pvals_corrected, reject) if r]

def compute_group_stats(df, group_col, value_cols, group_order, count_col="SF#"):
    """Compute mean, SEM, and count of unique samples per group."""
    if not all(col in df.columns for col in list(value_cols) + [count_col]):
        raise ValueError(f"Required columns not found in DataFrame")
    stats = df.groupby(group_col).agg(
        **{f"{col}_mean": (col, "mean") for col in value_cols},
        **{f"{col}_sem": (col, "sem") for col in value_cols},
        n=(count_col, "nunique")  # Count unique SF# values per group
    ).reset_index()
    stats["order"] = stats[group_col].apply(lambda x: group_order.index(x))
    return stats.sort_values("order")

def annotate_significance(ax, sig_pairs, measure_col, x, stats, bar_width, y_offset_factor=0.05):
    """Annotate significant pairwise comparisons on the plot."""
    max_mean = max(stats[f"{col}_mean"].max() for col in measures.values())
    for g1, g2, p_val in sig_pairs:
        i1, i2 = group_order.index(g1), group_order.index(g2)
        offset = -bar_width/2 if "STAR" in measure_col else bar_width/2
        x1, x2 = x[i1] + offset, x[i2] + offset
        
        y1 = stats.loc[stats["Histology_Tumor"] == g1, f"{measure_col}_mean"].values[0] + \
             stats.loc[stats["Histology_Tumor"] == g1, f"{measure_col}_sem"].values[0]
        y2 = stats.loc[stats["Histology_Tumor"] == g2, f"{measure_col}_mean"].values[0] + \
             stats.loc[stats["Histology_Tumor"] == g2, f"{measure_col}_sem"].values[0]
        
        y = max(y1, y2) + y_offset_factor * max_mean
        ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], lw=1.5, c="black")
        ax.text((x1 + x2) / 2, y + 0.02, f"p = {p_val:.2e}", ha="center", va="bottom", fontsize=14)

# ------------------------------
# Main Analysis
# ------------------------------
def analyze_and_plot(df, measures):
    """Perform statistical analysis and generate plot."""
    global group_order
    group_order = create_group_label(df)

    # Step 1: Kruskal-Wallis Test
    results = {}
    for name, col in measures.items():
        H, p = perform_kruskal_test(df, "Histology_Tumor", col, group_order)
        results[name] = {"H": H, "p": p}
        print(f"Kruskal–Wallis test for {name} Clonal Fusion across groups:")
        print(f"  H-statistic = {H:.4f}, p-value = {p:.4g}")

    # Step 2: Pairwise Comparisons
    sig_pairs = {}
    for name, col in measures.items():
        if results[name]["p"] > 2 :
            sig_pairs[name] = get_significant_pairs(df, "Histology_Tumor", col, group_order)
            print(f"\nSignificant pairwise comparisons for {name} Clonal Fusion:")
            for g1, g2, p in sig_pairs[name]:
                print(f"  {g1} vs {g2}: corrected p = {p:.4g}")
        else:
            sig_pairs[name] = []
            print(f"\nNo significant differences for {name} Clonal Fusion; skipping pairwise comparisons.")

    # Step 3: Compute Group Statistics
    stats = compute_group_stats(df, "Histology_Tumor", measures.values(), group_order)

    # Step 4: Create Grouped Bar Plot
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased width for readability
    x = np.arange(len(stats))
    bar_width = 0.4

    for i, (name, col) in enumerate(measures.items()):
        ax.bar(
            x + (i - 0.5) * bar_width,
            stats[f"{col}_mean"],
            yerr=stats[f"{col}_sem"],
            width=bar_width,
            label=f"{name} Clonal Fusion",
            capsize=5,
            alpha=0.8,
            edgecolor="black"
        )

    # Step 5: Annotate Significant Comparisons
    for name, col in measures.items():
        if sig_pairs[name]:
            annotate_significance(ax, sig_pairs[name], col, x, stats, bar_width)

    # Final Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(stats["Histology_Tumor"] + "\n(n=" + stats["n"].astype(str) + ")", rotation=45, ha="right")
    ax.set_ylabel("Number of Clonal Fusions", fontsize=12, fontweight="bold")
    ax.set_title("Number of Clonal Fusions by Group", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()