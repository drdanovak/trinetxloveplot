"""
TriNetX Love Plot Generator
- Drag-and-drop ordering
- Custom grouping header rows
- Legend outside plot, no overlap
- Manual X-axis, color controls, metrics

Usage:
    streamlit run 9_Love_Plots.py
"""

import math
from io import StringIO, BytesIO

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit / servers
import matplotlib.pyplot as plt

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode


# ------------------------- Parsing helpers ------------------------- #

def load_trinetx_baseline(uploaded_file) -> pd.DataFrame:
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
    with columns: label, <before_col>, <after_col>, abs_before, abs_after
    """
    mask = df[before_col].notna() | df[after_col].notna()
    df = df.loc[mask].copy()

    if "Characteristic Name" not in df.columns:
        raise ValueError("Expected a 'Characteristic Name' column in the baseline table.")

    df["label"] = df["Characteristic Name"].fillna("").astype(str)

    if include_categories and "Category" in df.columns:
        cat = df["Category"].fillna("").astype(str)
        df["label"] = np.where(
            cat.str.strip() != "",
            df["label"] + " (" + cat + ")",
            df["label"],
        )

    df[before_col] = pd.to_numeric(df[before_col], errors="coerce")
    df[after_col] = pd.to_numeric(df[after_col], errors="coerce")

    df["abs_before"] = df[before_col].abs()
    df["abs_after"] = df[after_col].abs()

    love_df = df[["label", before_col, after_col, "abs_before", "abs_after"]].copy()
    love_df = love_df.sort_values("abs_before", ascending=True).reset_index(drop=True)

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
    Header rows (is_header=True) should be excluded before calling this.
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
    before_color: str = "C0",
    after_color: str = "C1",
    x_min=None,
    x_max=None,
    show_legend: bool = True,
    legend_position: str = "Right outside",
    legend_fontsize: float = 10.0,
):
    """
    Generate the Love plot matplotlib Figure.

    `love_df` may contain an 'is_header' boolean column; header rows will be
    rendered as bold y-axis labels but will not have points plotted.
    Legend is always outside the axes so it never overlaps the data.
    """
    if love_df.empty:
        return None

    fig_height = max(6, len(love_df) * 0.3)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y = np.arange(len(love_df))

    if "is_header" in love_df.columns:
        is_header = love_df["is_header"].fillna(False).to_numpy(dtype=bool)
    else:
        is_header = np.zeros(len(love_df), dtype=bool)

    x_before = pd.to_numeric(love_df[before_col], errors="coerce").to_numpy()
    x_after = pd.to_numeric(love_df[after_col], errors="coerce").to_numpy()

    mask_before = (~is_header) & ~np.isnan(x_before)
    mask_after = (~is_header) & ~np.isnan(x_after)

    ax.scatter(
        x_before[mask_before],
        y[mask_before],
        label=before_label,
        marker="o",
        color=before_color,
    )
    ax.scatter(
        x_after[mask_after],
        y[mask_after],
        label=after_label,
        marker="s",
        color=after_color,
    )

    ax.axvline(0, linestyle="-", linewidth=1)
    for thr in [threshold, 2 * threshold]:
        ax.axvline(thr, linestyle="--", linewidth=0.7)
        ax.axvline(-thr, linestyle="--", linewidth=0.7)

    if (x_min is not None) and (x_max is not None):
        if x_min >= x_max:
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        ax.set_xlim(x_min, x_max)

    ax.set_yticks(list(y))
    ax.set_yticklabels(love_df["label"])

    for text, header_flag in zip(ax.get_yticklabels(), is_header):
        if header_flag:
            text.set_fontweight("bold")

    ax.set_xlabel("Standardized mean difference")
    # no title

    if show_legend:
        box = ax.get_position()

        if legend_position == "Right outside":
            ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])
            leg = ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
        elif legend_position == "Left outside":
            ax.set_position([box.x0 + box.width * 0.22, box.y0, box.width * 0.78, box.height])
            leg = ax.legend(
                loc="center right",
                bbox_to_anchor=(-0.02, 0.5),
                borderaxespad=0.0,
            )
        elif legend_position == "Top outside":
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.82])
            leg = ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.03),
                borderaxespad=0.5,
            )
        elif legend_position == "Bottom outside":
            ax.set_position([box.x0, box.y0 + box.height * 0.18, box.width, box.height * 0.82])
            leg = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.03),
                borderaxespad=0.5,
            )
        else:
            ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])
            leg = ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )

        for txt in leg.get_texts():
            txt.set_fontsize(legend_fontsize)
    else:
        fig.tight_layout()

    ax.invert_yaxis()

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

    df = load_trinetx_baseline(uploaded_file)

    before_col, after_col = find_smd_columns(df)
    if before_col is None or after_col is None:
        st.error(
            "Could not locate 'Before' and 'After' standardized mean difference "
            "columns. Make sure this is a TriNetX Baseline Patient Characteristics "
            "export that includes standardized mean differences."
        )
        st.stop()

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
        max_value=150,
        value=60,
        step=5,
        help="If your baseline table is large, this keeps the plot readable "
             "by limiting to the most imbalanced covariates.",
    )

    st.sidebar.subheader("Colors")
    use_bw = st.sidebar.checkbox(
        "Use black & white style",
        value=False,
        help="Overrides custom colors with black/grey markers.",
    )

    if use_bw:
        before_color = "#000000"
        after_color = "#555555"
    else:
        before_color = st.sidebar.color_picker(
            "Color for 'before' points",
            "#1f77b4",
        )
        after_color = st.sidebar.color_picker(
            "Color for 'after' points",
            "#ff7f0e",
        )

    st.sidebar.subheader("X-axis range")
    auto_xlim = st.sidebar.checkbox(
        "Automatic X-axis range",
        value=True,
        help="Uncheck to manually specify the SMD axis limits.",
    )

    x_min = None
    x_max = None
    if not auto_xlim:
        x_min = st.sidebar.number_input(
            "X-axis minimum",
            value=-0.5,
            step=0.05,
        )
        x_max = st.sidebar.number_input(
            "X-axis maximum",
            value=0.5,
            step=0.05,
        )
        if x_min >= x_max:
            st.sidebar.warning("X-axis min should be less than max; values will be swapped.")
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)

    st.sidebar.subheader("Legend")
    show_legend = st.sidebar.checkbox("Show legend", value=True)
    legend_position = st.sidebar.selectbox(
        "Legend position (always outside plot)",
        [
            "Right outside",
            "Left outside",
            "Top outside",
            "Bottom outside",
        ],
        index=0,
    )
    legend_fontsize = st.sidebar.number_input(
        "Legend font size",
        min_value=6.0,
        max_value=20.0,
        value=10.0,
        step=0.5,
    )

    love_df_full = prepare_love_data(
        df,
        before_col=before_col,
        after_col=after_col,
        include_categories=include_categories,
    )

    base_edit_df = love_df_full[["label", before_col, after_col, "abs_before", "abs_after"]].copy()
    base_edit_df.insert(0, "Include", True)
    base_edit_df.insert(1, "Group", "")
    base_edit_df.insert(2, "is_header", False)

    if (
        "edit_cov_df" not in st.session_state
        or st.session_state.get("_current_file_name") != uploaded_file.name
    ):
        st.session_state["_current_file_name"] = uploaded_file.name
        st.session_state["edit_cov_df"] = base_edit_df.copy()

    st.subheader("Covariates")

    st.markdown(
        "Use the table below to **drag and drop rows** to reorder covariates, "
        "mark **header rows** (`is_header`), rename labels, and include/exclude rows."
    )

    with st.expander("Add custom grouping header row", expanded=False):
        new_header_label = st.text_input(
            "Header label (e.g., Demographics, Labs, Medications)",
            value="",
            key="new_header_label",
        )
        if st.button("Add header row"):
            label = new_header_label.strip()
            if label:
                df_current = st.session_state["edit_cov_df"]
                new_row = {
                    "Include": True,
                    "Group": "",
                    "is_header": True,
                    "label": label,
                    before_col: np.nan,
                    after_col: np.nan,
                    "abs_before": np.nan,
                    "abs_after": np.nan,
                }
                st.session_state["edit_cov_df"] = pd.concat(
                    [df_current, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

    edit_df = st.session_state["edit_cov_df"]

    with st.expander("Edit covariate table (drag rows to reorder)", expanded=False):
        gb = GridOptionsBuilder.from_dataframe(edit_df)

        gb.configure_default_column(editable=True, resizable=True)
        gb.configure_column("Include", headerCheckboxSelection=False)
        gb.configure_column("Group")
        gb.configure_column("is_header", headerName="Header row")
        gb.configure_column("label", rowDrag=True)

        for col in [before_col, after_col, "abs_before", "abs_after"]:
            gb.configure_column(col, editable=False)

        grid_options = gb.build()
        grid_options["rowDragManaged"] = True
        grid_options["rowDragEntireRow"] = True
        grid_options["rowDragMultiRow"] = True

        grid_response = AgGrid(
            edit_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            enable_enterprise_modules=False,
            height=400,
        )

    edited_df = grid_response["data"]
    if not isinstance(edited_df, pd.DataFrame):
        edited_df = pd.DataFrame(edited_df)
    edited_df = edited_df.reset_index(drop=True)
    st.session_state["edit_cov_df"] = edited_df

    cov_df = edited_df[edited_df["Include"]].copy()

    cov_df["is_header"] = cov_df["is_header"].fillna(False).astype(bool)
    cov_df["Group"] = cov_df["Group"].fillna("").astype(str)
    cov_df[before_col] = pd.to_numeric(cov_df[before_col], errors="coerce")
    cov_df[after_col] = pd.to_numeric(cov_df[after_col], errors="coerce")
    cov_df["abs_before"] = cov_df[before_col].abs()
    cov_df["abs_after"] = cov_df[after_col].abs()

    data_rows = cov_df[~cov_df["is_header"]].copy()
    header_rows = cov_df[cov_df["is_header"]].copy()

    if len(data_rows) > max_covariates:
        keep_data_idx = data_rows["abs_before"].nlargest(max_covariates).index
        data_rows = data_rows.loc[keep_data_idx].sort_index()

    combined_df = pd.concat([header_rows, data_rows], axis=0)
    combined_df = combined_df.sort_index()
    cov_df_plot = combined_df.copy()

    plot_df = cov_df_plot[["label", before_col, after_col, "abs_before", "abs_after", "is_header"]].copy()

    st.subheader("Love plot")

    if plot_df.empty:
        st.warning("No covariates with non-missing SMDs to plot after filtering.")
        fig = None
    else:
        fig = make_love_plot(
            plot_df,
            before_col=before_col,
            after_col=after_col,
            before_label=before_label,
            after_label=after_label,
            threshold=threshold,
            before_color=before_color,
            after_color=after_color,
            x_min=x_min,
            x_max=x_max,
            show_legend=show_legend,
            legend_position=legend_position,
            legend_fontsize=legend_fontsize,
        )
        st.pyplot(fig)

        if fig is not None:
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                "Download Love plot (PNG)",
                data=buf,
                file_name="love_plot.png",
                mime="image/png",
            )

    st.subheader("Balance metrics")

    metric_df = cov_df[~cov_df["is_header"]].copy()
    metrics = compute_love_metrics(
        metric_df,
        before_col=before_col,
        after_col=after_col,
        threshold=threshold,
    )
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.format(precision=3))

    smd_table = cov_df[
        ["Group", "is_header", "label", before_col, after_col, "abs_before", "abs_after"]
    ].reset_index(drop=True)
    csv_bytes = smd_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download SMD table (CSV)",
        data=csv_bytes,
        file_name="smd_table.csv",
        mime="text/csv",
    )

    with st.expander("Show raw baseline table from TriNetX"):
        st.dataframe(df)


if __name__ == "__main__":
    main()
