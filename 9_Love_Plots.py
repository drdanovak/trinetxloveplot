"""
TriNetX Love Plot Generator

Usage:
    streamlit run love_plot_app.py

This app expects a TriNetX "Baseline Patient Characteristics" CSV export
(such as the example you provided). It:

- Strips off the header/notes section.
- Parses the baseline table.
- Finds the 'Before' and 'After' standardized mean difference columns.
- Generates a Love plot.
- Computes simple balance metrics and exposes a CSV download of SMDs.
"""

import math
from io import StringIO, BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ------------------------- Parsing helpers ------------------------- #

def load_trinetx_baseline(uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile") -> pd.DataFrame:
    """
    Load a TriNetX 'Baseline Patient Characteristics' CSV.

    Handles the front-matter (title, notes, etc.) by locating the line
    that starts with 'Characteristic ID' and reading from there.
    Falls back to a standard pd.read_csv if that header is not found.
    """
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8", errors="ignore")

    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Characteristic ID"):
            header_idx = i
            break

    if header_idx is not None:
        data_str = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(data_str))
    else:
        # Fallback: assume the file is a straightforward CSV
        df = pd.read_csv(StringIO(text))

    return df


def find_smd_columns(df: pd.DataFrame):
    """
    Identify the 'Before' and 'After' standardized mean difference columns
    in a TriNetX baseline table.
    """
    before_col = None
    after_col = None

    for col in df.columns:
        cname = col.lower()
        if "standardized mean difference" in cname and "before" in cname:
            before_col = col
        if "standardized mean difference" in cname and "after" in cname:
            after_col = col

    return before_col, after_col


def prepare_love_data(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    include_categories: bool = True,
) -> pd.DataFrame:
    """
    Create a tidy dataframe suitable for Love plotting.

    Each row corresponds to a covariate (or covariate-category level),
    with columns:

        label, <before_col>, <after_col>, abs_before, abs_after
    """
    # Keep rows that have an SMD in at least one of the two columns
    mask = df[before_col].notna() | df[after_col].notna()
    df = df.loc[mask].copy()

    # Base label: characteristic name
    if "Characteristic Name" not in df.columns:
        raise ValueError("Expected a 'Characteristic Name' column in the baseline table.")

    df["label"] = df["Characteristic Name"].fillna("").astype(str)

    # Optionally augment with category levels
    if include_categories and "Category" in df.columns:
        cat = df["Category"].fillna("").astype(str)
        df["label"] = np.where(
            cat.str.strip() != "",
            df["label"] + " (" + cat + ")",
            df["label"],
        )

    # Numeric SMDs + absolute values (for sorting)
    df[before_col] = pd.to_numeric(df[before_col], errors="coerce")
    df[after_col] = pd.to_numeric(df[after_col], errors="coerce")

    df["abs_before"] = df[before_col].abs()
    df["abs_after"] = df[after_col].abs()

    love_df = df[["label", before_col, after_col, "abs_before", "abs_after"]].copy()
    # Sort so that largest imbalance appears at the top of the plot
    love_df = love_df.sort_values("abs_before", ascending=True)

    return love_df


# ------------------------- Metrics & plotting ------------------------- #

def compute_love_metrics(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    threshold: float = 0.1,
) -> dict:
    """
    Compute summary balance metrics for 'before' and 'after' SMDs.
    """
    out = {}
    for label, col in [("Before", before_col), ("After", after_col)]:
        s = pd.to_numeric(df[col], errors="coerce").abs().dropna()
        if s.empty:
            stats = {
                "N covariates": 0,
                "Mean |SMD|": float("nan"),
                "Median |SMD|": float("nan"),
                "Max |SMD|": float("nan"),
                f"N(|SMD| > {threshold})": 0,
                f"Prop(|SMD| > {threshold})": float("nan"),
            }
        else:
            stats = {
                "N covariates": int(s.count()),
                "Mean |SMD|": s.mean(),
                "Median |SMD|": s.median(),
                "Max |SMD|": s.max(),
                f"N(|SMD| > {threshold})": int((s > threshold).sum()),
                f"Prop(|SMD| > {threshold})": float((s > threshold).mean()),
            }
        out[label] = stats

    b = out["Before"]["Mean |SMD|"]
    a = out["After"]["Mean |SMD|"]
    if not math.isnan(b) and not math.isnan(a) and b != 0:
        out["Overall"] = {
            "Percent reduction in mean |SMD|": 100.0 * (1.0 - a / b)
        }

    return out


def make_love_plot(
    love_df: pd.DataFrame,
    before_col: str,
    after_col: str,
    before_label: str = "Before matching",
    after_label: str = "After matching",
    threshold: float = 0.1,
):
    """
    Generate the Love plot matplotlib Figure.
    """
    if love_df.empty:
        return None

    fig_height = max(6, len(love_df) * 0.3)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_positions = range(len(love_df))

    ax.scatter(
        love_df[before_col],
        y_positions,
        label=before_label,
        marker="o",
    )
    ax.scatter(
        love_df[after_col],
        y_positions,
        label=after_label,
        marker="s",
    )

    # Vertical reference lines
    ax.axvline(0, linestyle="-", linewidth=1)
    for thr in [threshold, 2 * threshold]:
        ax.axvline(thr, linestyle="--", linewidth=0.7)
        ax.axvline(-thr, linestyle="--", linewidth=0.7)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(love_df["label"])
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Love plot (covariate balance)")

    ax.legend()
    ax.invert_yaxis()  # largest imbalance at top
    fig.tight_layout()

    return fig


# ------------------------- Streamlit UI ------------------------- #

def main():
    st.set_page_config(page_title="TriNetX Love Plot Generator", layout="wide")
    st.title("TriNetX Love Plot Generator")
    st.write(
        "Upload a TriNetX **Baseline Patient Characteristics** CSV from a "
        "propensity score–matched analysis to generate a Love plot and balance metrics."
    )

    uploaded_file = st.file_uploader(
        "Upload TriNetX baseline CSV",
        type=["csv"],
        help="Export from TriNetX: Baseline Patient Characteristics → Download as CSV",
    )

    if uploaded_file is None:
        st.info("Waiting for a TriNetX baseline CSV upload.")
        return

    # Parse baseline table
    df = load_trinetx_baseline(uploaded_file)

    before_col, after_col = find_smd_columns(df)
    if before_col is None or after_col is None:
        st.error(
            "Could not locate 'Before' and 'After' standardized mean difference "
            "columns. Make sure this is a TriNetX Baseline Patient Characteristics "
            "export that includes standardized mean differences."
        )
        st.stop()

    # Sidebar options
    st.sidebar.header("Plot options")
    before_label = st.sidebar.text_input("Label for 'before' group", "Before matching")
    after_label = st.sidebar.text_input("Label for 'after' group", "After matching")
    threshold = st.sidebar.number_input(
        "Reference threshold for |SMD|",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
    )
    include_categories = st.sidebar.checkbox(
        "Show category levels as separate covariates",
        value=True,
    )
    max_covariates = st.sidebar.slider(
        "Max covariates to display in Love plot",
        min_value=5,
        max_value=100,
        value=40,
        step=5,
        help="If your baseline table is large, this keeps the plot readable by showing only the most imbalanced covariates.",
    )

    # Prepare data for Love plot
    love_df_full = prepare_love_data(
        df,
        before_col=before_col,
        after_col=after_col,
        include_categories=include_categories,
    )

    # Keep only the most imbalanced covariates if requested
    if len(love_df_full) > max_covariates:
        love_df_plot = love_df_full.tail(max_covariates)
    else:
        love_df_plot = love_df_full

    # Layout: plot on left, metrics on right
    col_plot, col_metrics = st.columns([2, 1])

    with col_plot:
        st.subheader("Love plot")
        if love_df_plot.empty:
            st.warning("No covariates with non-missing SMDs to plot.")
        else:
            fig = make_love_plot(
                love_df_plot,
                before_col=before_col,
                after_col=after_col,
                before_label=before_label,
                after_label=after_label,
                threshold=threshold,
            )
            st.pyplot(fig)

            # Downloadable PNG
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                "Download Love plot (PNG)",
                data=buf,
                file_name="love_plot.png",
                mime="image/png",
            )

    with col_metrics:
        st.subheader("Balance metrics")
        metrics = compute_love_metrics(
            df,
            before_col=before_col,
            after_col=after_col,
            threshold=threshold,
        )
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.format(precision=3))

        # Download SMD table
        smd_table = love_df_full[["label", before_col, after_col, "abs_before", "abs_after"]]
        csv_bytes = smd_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download SMD table (CSV)",
            data=csv_bytes,
            file_name="smd_table.csv",
            mime="text/csv",
        )

    # Optional: raw baseline table
    with st.expander("Show raw baseline table"):
        st.dataframe(df)


if __name__ == "__main__":
    main()
