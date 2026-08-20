"""
Apply the completed policy-level manual review to text-only Article 4 areas.

Structured 3MA records are included directly. 
Text-only records receive the manual decision of their policy identification group.
"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RELEVANT_INPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "article4"
    / "article4_relevant_areas.gpkg"
)
RELEVANT_INPUT_LAYER = "article4_relevant_areas"

MANUAL_REVIEW_PATH = (
    PROJECT_ROOT
    / "manual_inputs"
    / "article4_text_only_manual_review.csv"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "article4"

FINAL_GPKG_PATH = (
    OUTPUT_DIRECTORY / "article4_relevant_areas_final.gpkg"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "article4_relevance_final_summary.json"
)

FINAL_LAYER_NAME = "article4_relevant_areas_final"

REQUIRED_REVIEW_COLUMNS = [
    "policy_group_id",
    "manual_decision",
    "reason",
]


def ensure_outputs_do_not_exist():
    """Prevent the final reviewed snapshot from being silently replaced."""

    existing = []

    for path in [
        FINAL_GPKG_PATH,
        SUMMARY_OUTPUT_PATH,
    ]:
        if path.exists():
            existing.append(str(path))

    if existing:
        raise FileExistsError(
            "Final Article 4 relevance outputs already exist. "
            "Review and move them before rerunning:\n- "
            + "\n- ".join(existing)
        )


def make_policy_group_id(frame):
    """Create the same policy-group identifier used during targeted review."""

    organisation = (
        frame["organisation-entity"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    direction = (
        frame["article-4-direction"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    direction = direction.where(
        direction != "",
        "NO_DIRECTION_REF",
    )

    return organisation + "::" + direction


def read_inputs():
    """Read the pre-review relevant areas and completed manual review."""

    if not RELEVANT_INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Relevant Article 4 input not found: {RELEVANT_INPUT_PATH}"
        )

    if not MANUAL_REVIEW_PATH.is_file():
        raise FileNotFoundError(
            f"Article 4 manual review not found: {MANUAL_REVIEW_PATH}"
        )

    relevant = gpd.read_file(
        RELEVANT_INPUT_PATH,
        layer=RELEVANT_INPUT_LAYER,
    )

    review = pd.read_csv(
        MANUAL_REVIEW_PATH,
        dtype="string",
        low_memory=False,
    )

    missing_review_columns = [
        column
        for column in REQUIRED_REVIEW_COLUMNS
        if column not in review.columns
    ]

    if missing_review_columns:
        raise ValueError(
            "Article 4 manual review is missing required columns: "
            + ", ".join(missing_review_columns)
        )

    return relevant, review


def prepare_review(review):
    """Validate the completed policy-level manual decisions."""

    review = review.copy()

    review["policy_group_id"] = (
        review["policy_group_id"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    if review["policy_group_id"].duplicated().any():
        raise ValueError(
            "Article 4 manual review contains duplicate policy_group_id values."
        )

    review["manual_decision"] = (
        review["manual_decision"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )

    allowed = {"INCLUDE", "EXCLUDE", "UNRESOLVED"}

    invalid = ~review["manual_decision"].isin(allowed)

    if invalid.any():
        raise ValueError(
            "Article 4 manual review contains blank or invalid decisions."
        )

    return review


def apply_manual_decisions(relevant, review):
    """Propagate each policy level manual decision back to its area records."""

    relevant = relevant.copy()

    relevant["policy_group_id"] = make_policy_group_id(relevant)

    text_only = relevant["relevance_basis"] == "text_only"

    text_policy_groups = set(
        relevant.loc[
            text_only,
            "policy_group_id",
        ].astype(str)
    )

    reviewed_policy_groups = set(
        review["policy_group_id"].astype(str)
    )

    if text_policy_groups != reviewed_policy_groups:
        missing = sorted(
            text_policy_groups - reviewed_policy_groups
        )
        extra = sorted(
            reviewed_policy_groups - text_policy_groups
        )

        raise ValueError(
            "Manual review does not exactly cover the text-only policy groups. "
            f"Missing={missing[:10]}; extra={extra[:10]}."
        )

    review_small = review[
        [
            "policy_group_id",
            "manual_decision",
            "reason",
        ]
    ].rename(
        columns={
            "reason": "manual_review_reason",
        }
    )

    relevant = relevant.merge(
        review_small,
        on="policy_group_id",
        how="left",
    )

    relevant = gpd.GeoDataFrame(
        relevant,
        geometry="geometry",
        crs=relevant.crs,
    )

    text_only = relevant["relevance_basis"] == "text_only"

    relevant["final_relevance_decision"] = "INCLUDE"
    relevant["relevance_decision_source"] = "structured_3ma"

    relevant.loc[
        text_only,
        "final_relevance_decision",
    ] = relevant.loc[
        text_only,
        "manual_decision",
    ]

    relevant.loc[
        text_only,
        "relevance_decision_source",
    ] = "manual_text_review"

    return relevant


def build_summary(screening, final_relevant, review):
    """Record the completed Article 4 relevance decision flow."""

    text_only = (
        screening["relevance_basis"] == "text_only"
    )

    policy_decisions = (
        review["manual_decision"]
        .value_counts()
        .sort_index()
    )

    text_area_decisions = (
        screening.loc[
            text_only,
            "final_relevance_decision",
        ]
        .value_counts()
        .sort_index()
    )

    final_decisions = (
        screening["final_relevance_decision"]
        .value_counts()
        .sort_index()
    )

    return {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "inputs": {
            "pre_review_relevant_areas": len(screening),
            "manual_policy_groups": int(
                screening.loc[
                    text_only,
                    "policy_group_id",
                ].nunique()
            ),
            "text_only_area_rows": int(text_only.sum()),
        },
        "manual_policy_decisions": {
            str(decision): int(count)
            for decision, count in policy_decisions.items()
        },
        "text_only_area_decisions": {
            str(decision): int(count)
            for decision, count in text_area_decisions.items()
        },
        "final_area_decisions": {
            str(decision): int(count)
            for decision, count in final_decisions.items()
        },
        "final_relevant_area_rows": len(final_relevant),
        "final_missing_effective_start_date": int(
            final_relevant["effective_start_date"].isna().sum()
        ),
    }


def main():
    """Apply manual decisions and save the final relevant Article 4 dataset."""

    ensure_outputs_do_not_exist()

    relevant, review = read_inputs()

    review = prepare_review(review)

    screening = apply_manual_decisions(
        relevant,
        review,
    )

    final_relevant = screening[
        screening["final_relevance_decision"] == "INCLUDE"
    ].copy()

    summary = build_summary(
        screening,
        final_relevant,
        review,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )



    final_relevant.to_file(
        FINAL_GPKG_PATH,
        layer=FINAL_LAYER_NAME,
        driver="GPKG",
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

    print(
        "Reviewed text-only policy groups: "
        f"{summary['inputs']['manual_policy_groups']}"
    )
    print(
        "Text-only area records: "
        f"{summary['inputs']['text_only_area_rows']}"
    )
    print(
        "Final relevant Article 4 areas: "
        f"{summary['final_relevant_area_rows']}"
    )
    print(
        "Missing effective start date: "
        f"{summary['final_missing_effective_start_date']}"
    )
    print(f"Final relevant areas: {FINAL_GPKG_PATH}")
    print(f"Summary: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()