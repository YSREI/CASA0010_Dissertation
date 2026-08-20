"""
Prepare the relevant Article 4 direction-area dataset from the frozen official
Article 4 source files.

This stage is independent of centre geometry. 
It:
-  joins direction metadata to direction-area records using
    (article-4-direction, organisation-entity);
-  identifies Article 4 areas relevant to Class MA / commercial-to-residential
   permitted-development restrictions;
-  constructs effective start/end dates using area dates first and direction
   dates as fallback;
-  preserves all relevant records, including those with missing effective dates.

"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIRECTION_PATH = (
    PROJECT_ROOT
    / "data_raw"
    / "article4"
    / "article4_direction_official.csv"
)
AREA_PATH = (
    PROJECT_ROOT
    / "data_raw"
    / "article4"
    / "article4_direction_area_official.geojson"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "article4"
RELEVANT_GPKG_PATH = OUTPUT_DIRECTORY / "article4_relevant_areas.gpkg"
RELEVANT_CSV_PATH = OUTPUT_DIRECTORY / "article4_relevant_areas.csv"
TEXT_ONLY_CSV_PATH = OUTPUT_DIRECTORY / "article4_text_only_relevance.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "article4_relevance_summary.json"

OUTPUT_LAYER_NAME = "article4_relevant_areas"
TARGET_CRS = "EPSG:27700"

REQUIRED_DIRECTION_COLUMNS = [
    "reference",
    "organisation-entity",
    "start-date",
    "end-date",
]

REQUIRED_AREA_COLUMNS = [
    "article-4-direction",
    "organisation-entity",
    "permitted-development-rights",
    "start-date",
    "end-date",
]


def ensure_outputs_do_not_exist():
    """Prevent the reviewed Article 4 snapshot from being silently replaced."""

    existing_outputs = []

    for output_path in [
        RELEVANT_GPKG_PATH,
        RELEVANT_CSV_PATH,
        TEXT_ONLY_CSV_PATH,
        SUMMARY_PATH,
    ]:
        if output_path.exists():
            existing_outputs.append(str(output_path))

    if existing_outputs:
        raise FileExistsError(
            "Article 4 outputs already exist. Review and move them before a "
            "new run:\n- " + "\n- ".join(existing_outputs)
        )


def require_columns(frame, required_columns, source_name):
    """Require the fields used by the current methodological rules."""

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


def normalise_key(series):
    """Normalise source join keys without altering their substantive content."""

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def read_sources():
    """Read the two frozen official Article 4 source files."""

    if not DIRECTION_PATH.is_file():
        raise FileNotFoundError(
            f"Article 4 direction file not found: {DIRECTION_PATH}"
        )

    if not AREA_PATH.is_file():
        raise FileNotFoundError(
            f"Article 4 direction-area file not found: {AREA_PATH}"
        )

    directions = pd.read_csv(
        DIRECTION_PATH,
        low_memory=False,
    )

    try:
        areas = gpd.read_file(AREA_PATH)
    except Exception as error:
        raise RuntimeError(
            f"Could not read Article 4 direction-area file: {error}"
        ) from error

    require_columns(
        directions,
        REQUIRED_DIRECTION_COLUMNS,
        "Article 4 direction file",
    )
    require_columns(
        areas,
        REQUIRED_AREA_COLUMNS,
        "Article 4 direction-area file",
    )

    if areas.crs is None:
        raise ValueError("Article 4 direction-area file has no CRS.")

    return directions, areas


def join_direction_metadata(directions, areas):
    """
    Join direction metadata to each direction-area record.
    Direction references are local identifiers, so organisation-entity is part of the key. 
    Direction rows without a usable two-part key are retained in the raw source 
    but cannot provide fallback metadata to an area record.
    """

    directions = directions.copy()
    areas = areas.copy()

    directions["join_direction_reference"] = normalise_key(
        directions["reference"]
    )
    directions["join_organisation_entity"] = normalise_key(
        directions["organisation-entity"]
    )

    areas["join_direction_reference"] = normalise_key(
        areas["article-4-direction"]
    )
    areas["join_organisation_entity"] = normalise_key(
        areas["organisation-entity"]
    )

    usable_direction_key = (
        (directions["join_direction_reference"] != "")
        & (directions["join_organisation_entity"] != "")
    )

    directions_joinable = directions[
        usable_direction_key
    ].copy()

    duplicate_key = directions_joinable.duplicated(
        subset=[
            "join_direction_reference",
            "join_organisation_entity",
        ],
        keep=False,
    )

    if duplicate_key.any():
        duplicate_rows = directions_joinable.loc[
            duplicate_key,
            [
                "reference",
                "organisation-entity",
            ],
        ]

        raise ValueError(
            "Article 4 direction records are not unique by "
            "(reference, organisation-entity):\n"
            + duplicate_rows.head(20).to_string(index=False)
        )

    direction_columns = {
        "reference": "direction_reference",
        "name": "direction_name",
        "description": "direction_description",
        "notes": "direction_notes",
        "start-date": "direction_start_date",
        "end-date": "direction_end_date",
        "document-url": "direction_document_url",
        "documentation-url": "direction_documentation_url",
    }

    available_direction_columns = [
        column
        for column in direction_columns
        if column in directions_joinable.columns
    ]

    directions_joinable = directions_joinable[
        [
            "join_direction_reference",
            "join_organisation_entity",
        ]
        + available_direction_columns
    ].rename(columns=direction_columns)

    merged = areas.merge(
        directions_joinable,
        on=[
            "join_direction_reference",
            "join_organisation_entity",
        ],
        how="left",
    )

    merged = gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=areas.crs,
    )

    return merged


def classify_relevance(areas):
    """
    Identify Article 4 areas relevant to the frozen research construct.
    Structured 3MA is the strongest indicator. 
    Text rules capture directions that clearly describe Class MA, Class E-to-C3, 
    or commercial-to-residential restrictions where the structured code is absent.
    """

    areas = areas.copy()

    text_columns = [
        "name",
        "description",
        "notes",
        "permitted-development-rights",
        "direction_name",
        "direction_description",
        "direction_notes",
    ]
    text_columns = [
        column
        for column in text_columns
        if column in areas.columns
    ]

    combined_text = (
        areas[text_columns]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
    )

    pdr = (
        areas["permitted-development-rights"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    has_3ma = pdr.str.contains(
        r"\b3ma\b",
        regex=True,
        na=False,
    )

    has_class_ma_text = combined_text.str.contains(
        r"class\s*ma|part\s*3\s*class\s*ma|schedule\s*2.*class\s*ma",
        regex=True,
        na=False,
    )

    has_commercial_to_residential_text = combined_text.str.contains(
        r"class\s*e.*c3"
        r"|use\s*class\s*e.*use\s*class\s*c3"
        r"|commercial.*residential"
        r"|business.*service.*residential"
        r"|office.*residential"
        r"|retail.*residential"
        r"|shop.*residential",
        regex=True,
        na=False,
    )

    hmo_like = combined_text.str.contains(
        r"\bhmo\b"
        r"|house\s*in\s*multiple\s*occupation"
        r"|c3.*c4"
        r"|c4.*hmo",
        regex=True,
        na=False,
    )

    text_relevant = (
        has_class_ma_text
        | has_commercial_to_residential_text
    )

    areas["is_relevant_3ma"] = has_3ma
    areas["is_relevant_text"] = text_relevant
    areas["is_hmo_like"] = hmo_like

    areas["is_relevant_article4"] = (
        (has_3ma | text_relevant)
        & ~(hmo_like & ~has_3ma)
    )

    areas["relevance_basis"] = ""

    areas.loc[
        has_3ma,
        "relevance_basis",
    ] = "3MA"

    areas.loc[
        ~has_3ma & text_relevant,
        "relevance_basis",
    ] = "text_only"

    return areas


def add_effective_dates(relevant):
    """
    Construct effective dates using the frozen fallback rule.
    Area-specific dates are preferred. 
    Direction-level dates are used only where the area-specific value is missing.
    """

    relevant = relevant.copy()

    relevant["area_start_date"] = pd.to_datetime(
        relevant["start-date"],
        errors="coerce",
    )
    relevant["area_end_date"] = pd.to_datetime(
        relevant["end-date"],
        errors="coerce",
    )

    if "direction_start_date" in relevant.columns:
        relevant["direction_start_date_parsed"] = pd.to_datetime(
            relevant["direction_start_date"],
            errors="coerce",
        )
    else:
        relevant["direction_start_date_parsed"] = pd.NaT

    if "direction_end_date" in relevant.columns:
        relevant["direction_end_date_parsed"] = pd.to_datetime(
            relevant["direction_end_date"],
            errors="coerce",
        )
    else:
        relevant["direction_end_date_parsed"] = pd.NaT

    relevant["effective_start_date"] = (
        relevant["area_start_date"]
        .fillna(relevant["direction_start_date_parsed"])
    )

    relevant["effective_end_date"] = (
        relevant["area_end_date"]
        .fillna(relevant["direction_end_date_parsed"])
    )

    return relevant


def build_summary(directions, areas, relevant):
    """Record the Article 4 source, join, relevance and date counts."""

    direction_metadata_joined = relevant[
        "direction_reference"
    ].notna().sum()

    return {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "inputs": {
            "direction_csv": str(DIRECTION_PATH),
            "direction_rows": len(directions),
            "direction_area_geojson": str(AREA_PATH),
            "direction_area_rows": len(areas),
        },
        "relevance": {
            "relevant_area_rows": len(relevant),
            "relevant_with_3ma": int(
                relevant["is_relevant_3ma"].sum()
            ),
            "relevant_with_text_rule": int(
                relevant["is_relevant_text"].sum()
            ),
            "relevant_text_only": int(
                (relevant["relevance_basis"] == "text_only").sum()
            ),
            "relevant_hmo_like": int(
                relevant["is_hmo_like"].sum()
            ),
        },
        "direction_metadata": {
            "relevant_rows_with_direction_metadata": int(
                direction_metadata_joined
            ),
            "relevant_rows_without_direction_metadata": int(
                len(relevant) - direction_metadata_joined
            ),
        },
        "dates": {
            "relevant_missing_effective_start_date": int(
                relevant["effective_start_date"].isna().sum()
            ),
            "relevant_missing_effective_end_date": int(
                relevant["effective_end_date"].isna().sum()
            ),
        },
        "output_crs": TARGET_CRS,
    }


def main():
    """Create and save the relevant Article 4 direction-area snapshot."""

    ensure_outputs_do_not_exist()

    directions, areas = read_sources()

    merged = join_direction_metadata(
        directions,
        areas,
    )

    classified = classify_relevance(merged)

    relevant = classified[
        classified["is_relevant_article4"]
    ].copy()

    if relevant.empty:
        raise ValueError(
            "No Article 4 direction-area records satisfy the relevance rules."
        )

    relevant = add_effective_dates(relevant)

    relevant = relevant.to_crs(TARGET_CRS)

    summary = build_summary(
        directions,
        areas,
        relevant,
    )

    text_only = relevant[
        relevant["relevance_basis"] == "text_only"
    ].copy()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    relevant.to_file(
        RELEVANT_GPKG_PATH,
        layer=OUTPUT_LAYER_NAME,
        driver="GPKG",
    )

    relevant.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        RELEVANT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    text_only.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        TEXT_ONLY_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    with SUMMARY_PATH.open(
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

    print(f"Direction records: {len(directions)}")
    print(f"Direction-area records: {len(areas)}")
    print(f"Relevant Article 4 areas: {len(relevant)}")
    print(
        "Relevant with 3MA: "
        f"{summary['relevance']['relevant_with_3ma']}"
    )
    print(
        "Relevant text-only: "
        f"{summary['relevance']['relevant_text_only']}"
    )
    print(
        "Missing effective start date: "
        f"{summary['dates']['relevant_missing_effective_start_date']}"
    )
    print(f"Relevant areas: {RELEVANT_GPKG_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
