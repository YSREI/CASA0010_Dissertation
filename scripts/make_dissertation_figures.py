"""
Generate dissertation figures from the frozen analytical outputs.

Expected project inputs:
- data_processed/centres/centres.gpkg
- data_processed/panel/centre_period_panel.csv
- data_processed/ptal/centre_ptal.csv
- data_processed/analysis/models/model_key_results.csv
- data_processed/analysis/models/model_sensitivity_results.csv
- data_processed/analysis/models/article4_ptal_marginal_effects.csv
"""

import os
import sys
from pathlib import Path

# The active environment can inherit an incompatible PROJ path from PostGIS.
# Point this process to its own Conda PROJ database before importing GeoPandas.
CONDA_PROJ_DIRECTORY = Path(sys.prefix) / "Library" / "share" / "proj"

if CONDA_PROJ_DIRECTORY.is_dir():
    os.environ["PROJ_LIB"] = str(CONDA_PROJ_DIRECTORY)
    os.environ["PROJ_DATA"] = str(CONDA_PROJ_DIRECTORY)

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOROUGH_PATH = (
    PROJECT_ROOT / "data_raw" / "boundaries" / "London_Boroughs.gpkg"
)
BOROUGH_LAYER = "london_boroughs"

CENTRES_PATH = (
    PROJECT_ROOT / "data_processed" / "centres" / "centres.gpkg"
)
CENTRE_LAYER = "centres"

PANEL_PATH = (
    PROJECT_ROOT / "data_processed" / "panel" / "centre_period_panel.csv"
)


TARGET_CRS = "EPSG:27700"

# Quiet spatial context used only in Figure 1
GREATER_LONDON_OUTLINE = "#B8B8B8"
BOROUGH_OUTLINE = "#E3E3E3"
BOROUGH_FILL = "#FAFAFA"
CENTRE_EDGE = "#333333"


KEY_RESULTS_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "analysis"
    / "models"
    / "model_key_results.csv"
)

SENSITIVITY_RESULTS_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "analysis"
    / "models"
    / "model_sensitivity_results.csv"
)

MARGINAL_EFFECTS_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "analysis"
    / "models"
    / "article4_ptal_marginal_effects.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "figures"
)

# Main orange palette.
ORANGE = "#D97706"
ORANGE_DARK = "#9A4F00"
ORANGE_MID = "#E89A3D"
ORANGE_LIGHT = "#F3B562"
ORANGE_PALE = "#FCE3C2"

CHARCOAL = "#333333"
MID_GREY = "#8A8A8A"
LIGHT_GREY = "#D9D9D9"
VERY_LIGHT_GREY = "#F2F2F2"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save_figure(fig, filename):
    """Save one dissertation-ready 300 dpi PNG."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def read_inputs():
    """Read frozen analytical outputs."""

    panel = pd.read_csv(
        PANEL_PATH,
        dtype={
            "boundary_id": "string",
            "period": "string",
        },
    )


    key_results = pd.read_csv(
        KEY_RESULTS_PATH,
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_RESULTS_PATH,
    )

    marginal = pd.read_csv(
        MARGINAL_EFFECTS_PATH,
    )

    return (
        panel,
        key_results,
        sensitivity,
        marginal,
    )


def figure_spatial_article4(panel):
    """
    Figure 1:
    Article 4 exposure in the final observed period, with Greater London
    and borough boundaries providing quiet spatial context.
    """

    for input_path in [BOROUGH_PATH, CENTRES_PATH]:
        if not input_path.is_file():
            raise FileNotFoundError(f"Figure 1 input not found: {input_path}")

    boroughs = gpd.read_file(
        BOROUGH_PATH,
        layer=BOROUGH_LAYER,
    )
    centres = gpd.read_file(
        CENTRES_PATH,
        layer=CENTRE_LAYER,
    )

    if boroughs.crs is None or centres.crs is None:
        raise ValueError("Figure 1 spatial inputs must have a defined CRS.")

    boroughs = boroughs.to_crs(TARGET_CRS)
    centres = centres.to_crs(TARGET_CRS)

    if len(boroughs) != 33:
        raise ValueError(
            f"Expected 33 London borough features, found {len(boroughs)}."
        )

    if centres["boundary_id"].isna().any():
        raise ValueError("centres.gpkg contains missing boundary_id values.")

    if not centres["boundary_id"].is_unique:
        raise ValueError("centres.gpkg contains duplicate boundary_id values.")

    required_panel_columns = [
        "boundary_id",
        "period",
        "period_index",
        "article4_share",
    ]
    missing_columns = [
        column
        for column in required_panel_columns
        if column not in panel.columns
    ]

    if missing_columns:
        raise ValueError(
            "Panel is missing Figure 1 columns: "
            + ", ".join(missing_columns)
        )

    latest_period = (
        panel[
            ["period", "period_index"]
        ]
        .drop_duplicates()
        .sort_values("period_index")
        .iloc[-1]["period"]
    )

    latest = panel.loc[
        panel["period"] == latest_period,
        [
            "boundary_id",
            "article4_share",
        ],
    ].copy()

    if not latest["boundary_id"].is_unique:
        raise ValueError(
            "Latest-period panel rows are not unique by boundary_id."
        )

    mapped = centres.merge(
        latest,
        on="boundary_id",
        how="left",
        validate="one_to_one",
    )

    if mapped["article4_share"].isna().any():
        raise ValueError(
            "At least one centre has no latest-period Article 4 share."
        )

    london_outline = gpd.GeoSeries(
        [boroughs.geometry.union_all()],
        crs=boroughs.crs,
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 7.5)
    )

    boroughs.plot(
        ax=ax,
        facecolor=BOROUGH_FILL,
        edgecolor=BOROUGH_OUTLINE,
        linewidth=0.35,
        zorder=1,
    )

    london_outline.boundary.plot(
        ax=ax,
        color=GREATER_LONDON_OUTLINE,
        linewidth=0.9,
        zorder=2,
    )

    mapped.plot(
        ax=ax,
        column="article4_share",
        cmap="Oranges",
        vmin=0,
        vmax=1,
        linewidth=0.25,
        edgecolor=CENTRE_EDGE,
        zorder=3,
    )

    colour_scale = ScalarMappable(
        norm=Normalize(
            vmin=0,
            vmax=1,
        ),
        cmap="Oranges",
    )
    colour_scale.set_array([])

    colour_bar = fig.colorbar(
        colour_scale,
        ax=ax,
        fraction=0.035,
        pad=0.018,
    )
    colour_bar.set_label(
        "Article 4 share"
    )
    colour_bar.ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1
        )
    )

    ax.set_title(
        f"Article 4 exposure across centre-boundary features, {latest_period}",
        loc="left",
        fontweight="bold",
    )
    ax.set_axis_off()

    save_figure(
        fig,
        "Figure_1_Article4_spatial_exposure",
    )

def figure_article4_adoption(panel):
    """
    Main figure:
    Treatment accumulation through time.
    A step-line is preferable to bars because treatment adoption is ordered
    and persistent.
    """

    treated = (
        panel.groupby(
            ["period", "period_index"]
        )["treated_25"]
        .sum()
        .reset_index()
        .sort_values("period_index")
    )

    x = np.arange(
        len(treated)
    )

    y = treated[
        "treated_25"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(8.2, 4.6)
    )

    ax.step(
        x,
        y,
        where="mid",
        linewidth=2.2,
        color=ORANGE,
    )
    ax.scatter(
        x,
        y,
        s=38,
        color=ORANGE_DARK,
        zorder=3,
    )

    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        treated["period"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Centre-boundary features at or above 25%"
    )
    ax.set_xlabel(
        "Observed period"
    )

    ax.grid(
        axis="y",
        color=LIGHT_GREY,
        linewidth=0.7,
        alpha=0.7,
    )

    ax.text(
        0.02,
        0.95,
        "0 always treated | 911 never treated | 211 switchers",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=CHARCOAL,
    )

    ax.set_title(
        "Article 4 exposure adoption at the 25% threshold",
        loc="left",
        fontweight="bold",
    )

    save_figure(
        fig,
        "Figure_2_Article4_treatment_adoption",
    )


def figure_application_activity(panel):
    """
    Main descriptive figure:
    Total matched commercial-to-residential applications by period.
    A line chart shows the ordered temporal pattern more naturally than bars.
    """

    activity = (
        panel.groupby(
            ["period", "period_index"]
        )["application_count"]
        .sum()
        .reset_index()
        .sort_values("period_index")
    )

    x = np.arange(
        len(activity)
    )

    fig, ax = plt.subplots(
        figsize=(8.2, 4.6)
    )

    # Light shading reminds the reader that the first and last quarters
    # are only partially observed.
    ax.axvspan(
        -0.45,
        0.45,
        color=VERY_LIGHT_GREY,
        zorder=0,
    )
    ax.axvspan(
        len(activity) - 1.45,
        len(activity) - 0.55,
        color=VERY_LIGHT_GREY,
        zorder=0,
    )

    ax.plot(
        x,
        activity["application_count"],
        color=ORANGE,
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        markerfacecolor=ORANGE_DARK,
    )

    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        activity["period"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Matched applications"
    )
    ax.set_xlabel(
        "Observed period"
    )

    ax.grid(
        axis="y",
        color=LIGHT_GREY,
        linewidth=0.7,
        alpha=0.7,
    )

    ax.set_title(
        "Commercial-to-residential planning application activity over time",
        loc="left",
        fontweight="bold",
    )

    save_figure(
        fig,
        "Figure_3_application_activity_over_time",
    )


def figure_marginal_association(marginal):
    """
    Main empirical figure:
    M2 Article 4 association across the PTAL distribution.
    Line + 95% confidence ribbon is the most direct way to communicate an interaction effect.
    """

    fig, ax = plt.subplots(
        figsize=(7.4, 4.8)
    )

    x = marginal[
        "ptal_mean_ai"
    ].to_numpy()

    y = marginal[
        "article4_effect_per_10pp"
    ].to_numpy()

    low = marginal[
        "conf_low"
    ].to_numpy()

    high = marginal[
        "conf_high"
    ].to_numpy()

    ax.fill_between(
        x,
        low,
        high,
        color=ORANGE_PALE,
        alpha=0.85,
        linewidth=0,
        label="95% CI",
    )

    ax.plot(
        x,
        y,
        color=ORANGE_DARK,
        linewidth=2.3,
        label="Estimated association",
    )

    ax.axhline(
        0,
        color=CHARCOAL,
        linewidth=1,
        linestyle="--",
    )

    ax.set_xlabel(
        "Mean TfL Access Index"
    )
    ax.set_ylabel(
        "Change in applications\nper 10pp increase in Article 4 coverage"
    )

    ax.grid(
        axis="y",
        color=LIGHT_GREY,
        linewidth=0.7,
        alpha=0.7,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.set_title(
        "Within-centre Article 4 association across accessibility levels",
        loc="left",
        fontweight="bold",
    )

    save_figure(
        fig,
        "Figure_4_Article4_PTAL_marginal_association",
    )


def figure_continuous_robustness(
    key_results,
    sensitivity,
):
    """
    Robustness figure:
    Comparable continuous-share specifications only.
    Threshold models are excluded because a binary 0/1 treatment coefficient
    is not on the same scale as a continuous 0-1 Article 4 share coefficient.
    """

    primary = key_results.loc[
        (
            key_results["model"] == "M2"
        )
        & (
            key_results["term"] == "article4_share"
        )
    ].copy()

    tier1 = sensitivity.loc[
        (
            sensitivity["model"] == "tier1_only"
        )
        & (
            sensitivity["term"] == "article4_share"
        )
    ].copy()

    complete = sensitivity.loc[
        (
            sensitivity["model"] == "complete_quarters"
        )
        & (
            sensitivity["term"] == "article4_share"
        )
    ].copy()

    rows = pd.concat(
        [
            primary.assign(
                label="Primary: all centres, 11 periods"
            ),
            tier1.assign(
                label="Tier 1 centres only"
            ),
            complete.assign(
                label="Nine complete quarters"
            ),
        ],
        ignore_index=True,
    )

    rows = rows[
        [
            "label",
            "coef",
            "conf_low",
            "conf_high",
        ]
    ].copy()

    # Convert the 0-1 article4_share coefficient to the effect associated
    # with a 10 percentage-point increase in Article 4 coverage
    for column in [
        "coef",
        "conf_low",
        "conf_high",
    ]:
        rows[column] = (
            rows[column] * 0.10
        )

    y = np.arange(
        len(rows)
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 3.6)
    )

    ax.errorbar(
        rows["coef"],
        y,
        xerr=[
            rows["coef"] - rows["conf_low"],
            rows["conf_high"] - rows["coef"],
        ],
        fmt="o",
        color=ORANGE_DARK,
        ecolor=ORANGE,
        elinewidth=2,
        capsize=4,
        markersize=6,
    )

    ax.axvline(
        0,
        color=CHARCOAL,
        linewidth=1,
        linestyle="--",
    )

    ax.set_yticks(
        y
    )
    ax.set_yticklabels(
        rows["label"]
    )
    ax.invert_yaxis()

    ax.set_xlabel(
        "Change in applications per 10pp increase in Article 4 coverage"
    )

    ax.grid(
        axis="x",
        color=LIGHT_GREY,
        linewidth=0.7,
        alpha=0.7,
    )

    ax.set_title(
        "Robustness of the within-centre Article 4 association",
        loc="left",
        fontweight="bold",
    )

    save_figure(
        fig,
        "Figure_5_continuous_robustness",
    )


def figure_threshold_sensitivity(sensitivity):
    """
    Optional / appendix:
    Horizontal coefficient plot for comparable binary thresholds.
    """

    threshold_order = [
        "treated_10",
        "treated_25",
        "treated_50",
    ]

    rows = []

    for treatment in threshold_order:
        row = sensitivity.loc[
            (
                sensitivity["model"] == treatment
            )
            & (
                sensitivity["term"] == treatment
            )
        ].iloc[0]

        rows.append(
            {
                "label": {
                    "treated_10": "10% threshold",
                    "treated_25": "25% threshold",
                    "treated_50": "50% threshold",
                }[treatment],
                "coef": row["coef"],
                "conf_low": row["conf_low"],
                "conf_high": row["conf_high"],
            }
        )

    rows = pd.DataFrame(
        rows
    )

    y = np.arange(
        len(rows)
    )

    fig, ax = plt.subplots(
        figsize=(6.8, 3.5)
    )

    ax.errorbar(
        rows["coef"],
        y,
        xerr=[
            rows["coef"] - rows["conf_low"],
            rows["conf_high"] - rows["coef"],
        ],
        fmt="o",
        color=ORANGE_DARK,
        ecolor=ORANGE,
        elinewidth=2,
        capsize=4,
        markersize=6,
    )

    ax.axvline(
        0,
        color=CHARCOAL,
        linewidth=1,
        linestyle="--",
    )

    ax.set_yticks(
        y
    )
    ax.set_yticklabels(
        rows["label"]
    )
    ax.invert_yaxis()

    ax.set_xlabel(
        "Binary treatment coefficient"
    )

    ax.grid(
        axis="x",
        color=LIGHT_GREY,
        linewidth=0.7,
        alpha=0.7,
    )

    ax.set_title(
        "Sensitivity to alternative Article 4 exposure thresholds",
        loc="left",
        fontweight="bold",
    )

    save_figure(
        fig,
        "Appendix_Figure_A1_Article4_threshold_sensitivity",
    )


def main():
    """Generate only the figures retained for the dissertation."""

    (
        panel,
        key_results,
        sensitivity,
        marginal,
    ) = read_inputs()

    # Main-text figures
    figure_spatial_article4(
        panel
    )
    figure_article4_adoption(
        panel
    )
    figure_application_activity(
        panel
    )
    figure_marginal_association(
        marginal
    )
    figure_continuous_robustness(
        key_results,
        sensitivity,
    )

    # Single retained appendix figure
    figure_threshold_sensitivity(
        sensitivity
    )

    print(
        f"Final dissertation figures saved to: {OUTPUT_DIR}"
    )

if __name__ == "__main__":
    main()