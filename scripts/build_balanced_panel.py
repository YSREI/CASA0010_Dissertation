"""
Construct the final balanced centre-period panel.

The panel skeleton is built independently from the frozen analytical centre
universe and the 11 observed study periods. 

Three frozen components are then merged onto that skeleton:
1. matched commercial-to-residential application counts
2. Article 4 centre-period exposure
3. centre-level TfL accessibility (PTAL Access Index)

No modelling nwo
"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CENTRE_INPUT_PATH = (
    PROJECT_ROOT / "data_processed" / "centres" / "centres.gpkg"
)
CENTRE_SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "centres" / "centre_summary.json"
)
CENTRE_LAYER_NAME = "centres"

SPATIAL_INPUT_PATH = (
    PROJECT_ROOT / "data_processed" / "spatial" / "pld_spatial.csv"
)
SPATIAL_SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "spatial" / "spatial_summary.json"
)

ARTICLE4_INPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "article4"
    / "article4_centre_period.csv"
)
ARTICLE4_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "article4"
    / "article4_centre_period_summary.json"
)

PTAL_INPUT_PATH = (
    PROJECT_ROOT / "data_processed" / "ptal" / "centre_ptal.csv"
)
PTAL_SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "ptal" / "centre_ptal_summary.json"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "panel"
PANEL_OUTPUT_PATH = OUTPUT_DIRECTORY / "centre_period_panel.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIRECTORY / "centre_period_panel_summary.json"

STUDY_START = pd.Timestamp("2021-08-01")
STUDY_END = pd.Timestamp("2024-03-04")

REQUIRED_CENTRE_COLUMNS = [
    "boundary_id",
    "centre_name",
    "centre_borough",
    "tier",
]

REQUIRED_SPATIAL_COLUMNS = [
    "es_id",
    "quarter",
    "spatial_status",
    "boundary_id",
]

REQUIRED_ARTICLE4_COLUMNS = [
    "boundary_id",
    "period",
    "article4_share",
    "has_article4",
    "treated_10",
    "treated_25",
    "treated_50",
]

REQUIRED_PTAL_COLUMNS = [
    "boundary_id",
    "ptal_mean_ai",
    "ptal_mean_category",
]


def read_json(path):
    """Read a required upstream summary."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required upstream summary not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_columns(frame, required_columns, source_name):
    """Require fields used directly by this stage."""

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(missing)
        )


def ensure_outputs_do_not_exist():
    """Prevent the frozen panel snapshot from being silently replaced."""

    existing = []

    for path in [
        PANEL_OUTPUT_PATH,
        SUMMARY_OUTPUT_PATH,
    ]:
        if path.exists():
            existing.append(str(path))

    if existing:
        raise FileExistsError(
            "Panel outputs already exist. Review and move them before rerunning:\n- "
            + "\n- ".join(existing)
        )


def read_centres(centre_summary):
    """Read the frozen analytical centre-boundary universe."""

    centres = gpd.read_file(
        CENTRE_INPUT_PATH,
        layer=CENTRE_LAYER_NAME,
    )

    require_columns(
        centres,
        REQUIRED_CENTRE_COLUMNS,
        "Centre GeoPackage",
    )

    expected_count = int(
        centre_summary["final_centres"]["final_centre_count"]
    )

    if len(centres) != expected_count:
        raise ValueError(
            "Centre input does not match centre_summary.json: "
            f"{len(centres)} != {expected_count}."
        )

    centres = centres[
        REQUIRED_CENTRE_COLUMNS
    ].copy()

    return centres


def build_period_table():
    """Create the 11 observed period labels in chronological order."""

    periods = pd.period_range(
        STUDY_START,
        STUDY_END,
        freq="Q",
    )

    return pd.DataFrame(
        {
            "period": [
                str(period)
                for period in periods
            ],
            "period_index": range(len(periods)),
        }
    )


def build_panel_skeleton(centres, periods):
    """Create every centre-boundary-feature × observed-period combination."""

    centre_rows = centres.copy()
    period_rows = periods.copy()

    centre_rows["_key"] = 1
    period_rows["_key"] = 1

    panel = (
        centre_rows.merge(
            period_rows,
            on="_key",
            how="inner",
        )
        .drop(columns="_key")
    )

    expected_rows = len(centres) * len(periods)

    if len(panel) != expected_rows:
        raise ValueError(
            "Balanced panel skeleton does not reconcile: "
            f"{len(panel)} != {len(centres)} * {len(periods)}."
        )

    return panel


def read_application_counts(spatial_summary, valid_periods):
    """Aggregate matched eligible applications to centre-period counts."""

    applications = pd.read_csv(
        SPATIAL_INPUT_PATH,
        dtype={
            "es_id": "string",
            "boundary_id": "string",
        },
        low_memory=False,
    )

    require_columns(
        applications,
        REQUIRED_SPATIAL_COLUMNS,
        "Spatial application output",
    )

    expected_total = int(
        spatial_summary["inputs"]["input_eligible_applications"]
    )

    expected_matched = int(
        spatial_summary["final_assignment"]["matched_applications"]
    )

    expected_represented_centres = int(
        spatial_summary["final_assignment"]["represented_centre_count"]
    )

    if len(applications) != expected_total:
        raise ValueError(
            "Spatial application output does not match spatial_summary.json: "
            f"{len(applications)} != {expected_total}."
        )

    matched = applications[
        applications["spatial_status"] == "matched"
    ].copy()

    if len(matched) != expected_matched:
        raise ValueError(
            "Matched application count does not match spatial_summary.json: "
            f"{len(matched)} != {expected_matched}."
        )

    matched["period"] = (
        matched["quarter"]
        .astype("string")
        .str.strip()
    )

    invalid_period = ~matched["period"].isin(
        valid_periods
    )

    if invalid_period.any():
        raise ValueError(
            "A matched application falls outside the observed panel periods."
        )

    application_counts = (
        matched.groupby(
            ["boundary_id", "period"]
        )
        .size()
        .rename("application_count")
        .reset_index()
    )

    if int(application_counts["application_count"].sum()) != expected_matched:
        raise ValueError(
            "Aggregated centre-period application counts do not reconcile "
            "to the matched application universe."
        )

    represented_centres = int(
        matched["boundary_id"].nunique()
    )

    if represented_centres != expected_represented_centres:
        raise ValueError(
            "Represented centre count does not match spatial_summary.json: "
            f"{represented_centres} != {expected_represented_centres}."
        )

    return application_counts, matched


def read_article4(article4_summary):
    """Read the frozen Article 4 centre-period exposure table."""

    article4 = pd.read_csv(
        ARTICLE4_INPUT_PATH,
        dtype={
            "boundary_id": "string",
            "period": "string",
        },
    )

    require_columns(
        article4,
        REQUIRED_ARTICLE4_COLUMNS,
        "Article 4 centre-period output",
    )

    expected_rows = int(
        article4_summary["exposure"]["centre_period_rows"]
    )

    if len(article4) != expected_rows:
        raise ValueError(
            "Article 4 row count does not match its summary: "
            f"{len(article4)} != {expected_rows}."
        )

    return article4[
        REQUIRED_ARTICLE4_COLUMNS
    ].copy()


def read_ptal(ptal_summary):
    """Read the frozen centre-level PTAL accessibility table."""

    ptal = pd.read_csv(
        PTAL_INPUT_PATH,
        dtype={
            "boundary_id": "string",
        },
    )

    require_columns(
        ptal,
        REQUIRED_PTAL_COLUMNS,
        "PTAL centre output",
    )

    expected_count = int(
        ptal_summary["inputs"]["centre_count"]
    )

    if len(ptal) != expected_count:
        raise ValueError(
            "PTAL row count does not match its summary: "
            f"{len(ptal)} != {expected_count}."
        )

    if ptal["ptal_mean_ai"].isna().any():
        raise ValueError(
            "PTAL input contains a missing ptal_mean_ai."
        )

    return ptal[
        REQUIRED_PTAL_COLUMNS
    ].copy()


def merge_components(
    skeleton,
    application_counts,
    article4,
    ptal,
):
    """Merge outcomes, Article 4 exposure and PTAL onto the panel skeleton."""

    valid_keys = set(
        zip(
            skeleton["boundary_id"].astype(str),
            skeleton["period"].astype(str),
        )
    )

    application_keys = set(
        zip(
            application_counts["boundary_id"].astype(str),
            application_counts["period"].astype(str),
        )
    )

    unknown_application_keys = (
        application_keys - valid_keys
    )

    if unknown_application_keys:
        raise ValueError(
            "Matched application counts contain centre-period keys "
            "outside the balanced panel skeleton."
        )

    panel = skeleton.merge(
        article4,
        on=["boundary_id", "period"],
        how="left",
        validate="one_to_one",
    )

    if panel["article4_share"].isna().any():
        raise ValueError(
            "Article 4 exposure does not cover every panel key."
        )

    panel = panel.merge(
        ptal,
        on="boundary_id",
        how="left",
        validate="many_to_one",
    )

    if panel["ptal_mean_ai"].isna().any():
        raise ValueError(
            "PTAL accessibility does not cover every analytical centre."
        )

    panel = panel.merge(
        application_counts,
        on=["boundary_id", "period"],
        how="left",
        validate="one_to_one",
    )

    panel["application_count"] = (
        panel["application_count"]
        .fillna(0)
        .astype(int)
    )

    return panel


def treatment_variation(panel):
    """Summarise the temporal variation in Article 4 exposure."""

    by_centre = (
        panel.sort_values(
            ["boundary_id", "period_index"]
        )
        .groupby("boundary_id")
    )

    treated_sum = by_centre[
        "treated_25"
    ].sum()

    period_count = int(
        panel["period"].nunique()
    )

    always_treated = int(
        treated_sum.eq(period_count).sum()
    )
    never_treated = int(
        treated_sum.eq(0).sum()
    )
    switchers = int(
        (
            treated_sum.gt(0)
            & treated_sum.lt(period_count)
        ).sum()
    )

    first_treated = (
        panel.loc[
            panel["treated_25"] == 1,
            ["boundary_id", "period_index", "period"],
        ]
        .sort_values(
            ["boundary_id", "period_index"]
        )
        .drop_duplicates(
            "boundary_id"
        )
    )

    first_treated_counts = {
        str(period): int(count)
        for period, count in (
            first_treated["period"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    max_share = by_centre[
        "article4_share"
    ].max()

    return {
        "treated_25": {
            "always_treated_centres": always_treated,
            "never_treated_centres": never_treated,
            "switching_centres": switchers,
            "first_treated_period_counts": first_treated_counts,
        },
        "continuous_article4": {
            "centres_ever_positive": int(
                max_share.gt(0).sum()
            ),
            "centres_never_positive": int(
                max_share.eq(0).sum()
            ),
        },
    }


def build_summary(
    centres,
    periods,
    matched,
    panel,
    variation,
):
    """Record panel reconciliation and substantive variation."""

    positive_rows = int(
        panel["application_count"].gt(0).sum()
    )

    period_application_counts = {
        str(period): int(count)
        for period, count in (
            panel.groupby("period")[
                "application_count"
            ]
            .sum()
            .sort_index()
            .items()
        )
    }

    return {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "inputs": {
            "centre_count": len(centres),
            "observed_period_count": len(periods),
            "matched_application_count": len(matched),
        },
        "panel": {
            "centre_period_rows": len(panel),
            "unique_centre_period_keys": int(
                panel[
                    ["boundary_id", "period"]
                ]
                .drop_duplicates()
                .shape[0]
            ),
            "total_application_count": int(
                panel["application_count"].sum()
            ),
            "centre_periods_with_applications": positive_rows,
            "centre_periods_with_zero_applications": int(
                len(panel) - positive_rows
            ),
            "max_applications_in_centre_period": int(
                panel["application_count"].max()
            ),
            "application_counts_by_period": period_application_counts,
        },
        "article4_variation": variation,
        "ptal": {
            "centres_with_ptal": int(
                panel[
                    ["boundary_id", "ptal_mean_ai"]
                ]
                .drop_duplicates("boundary_id")[
                    "ptal_mean_ai"
                ]
                .notna()
                .sum()
            ),
            "mean_ptal_mean_ai": float(
                panel[
                    ["boundary_id", "ptal_mean_ai"]
                ]
                .drop_duplicates("boundary_id")[
                    "ptal_mean_ai"
                ]
                .mean()
            ),
        },
    }


def main():
    """Construct and save the balanced analytical panel."""

    ensure_outputs_do_not_exist()

    centre_summary = read_json(
        CENTRE_SUMMARY_PATH
    )
    spatial_summary = read_json(
        SPATIAL_SUMMARY_PATH
    )
    article4_summary = read_json(
        ARTICLE4_SUMMARY_PATH
    )
    ptal_summary = read_json(
        PTAL_SUMMARY_PATH
    )

    centres = read_centres(
        centre_summary
    )

    periods = build_period_table()

    skeleton = build_panel_skeleton(
        centres,
        periods,
    )

    application_counts, matched = (
        read_application_counts(
            spatial_summary,
            set(periods["period"]),
        )
    )

    article4 = read_article4(
        article4_summary
    )

    ptal = read_ptal(
        ptal_summary
    )

    panel = merge_components(
        skeleton,
        application_counts,
        article4,
        ptal,
    )

    expected_rows = len(centres) * len(periods)

    if len(panel) != expected_rows:
        raise ValueError(
            "Final panel row count does not reconcile: "
            f"{len(panel)} != {expected_rows}."
        )

    if panel[
        ["boundary_id", "period"]
    ].duplicated().any():
        raise ValueError(
            "Final panel contains duplicate centre-period keys."
        )

    variation = treatment_variation(
        panel
    )

    summary = build_summary(
        centres,
        periods,
        matched,
        panel,
        variation,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.to_csv(
        PANEL_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    with SUMMARY_OUTPUT_PATH.open(
        "x",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=2,
            ensure_ascii=False,
        )
        summary_file.write("\n")

    print(f"Centres: {len(centres)}")
    print(f"Observed periods: {len(periods)}")
    print(f"Matched applications: {len(matched)}")
    print(f"Panel rows: {len(panel)}")
    print(
        "Total panel application count: "
        f"{int(panel['application_count'].sum())}"
    )
    print(
        "Centre-periods with applications: "
        f"{int(panel['application_count'].gt(0).sum())}"
    )
    print(
        "treated_25 always / never / switchers: "
        f"{variation['treated_25']['always_treated_centres']} / "
        f"{variation['treated_25']['never_treated_centres']} / "
        f"{variation['treated_25']['switching_centres']}"
    )
    print(
        "Centres ever positive article4_share: "
        f"{variation['continuous_article4']['centres_ever_positive']}"
    )
    print(f"Panel output: {PANEL_OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
