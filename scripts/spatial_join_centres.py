"""
Assign every eligible PLD application a centre-spatial status.

Every eligible application remains in the final application-level output.
Applications with unusable centroids are retained as invalid_centroid, 
and valid centroids that intersect no analytical centre are retained as unmatched.
"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPLICATION_INPUT_PATH = (
    PROJECT_ROOT / "data_processed" / "conversions" / "pld_clean.csv"
)
APPLICATION_SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "conversions" / "merge_summary.json"
)

CENTRE_INPUT_PATH = (
    PROJECT_ROOT / "data_processed" / "centres" / "centres.gpkg"
)
CENTRE_SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "centres" / "centre_summary.json"
)
CENTRE_LAYER_NAME = "centres"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "spatial"
SPATIAL_OUTPUT_PATH = OUTPUT_DIRECTORY / "pld_spatial.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIRECTORY / "spatial_summary.json"

TARGET_CRS = "EPSG:27700"

REQUIRED_APPLICATION_COLUMNS = [
    "es_id",
    "centroid_easting",
    "centroid_northing",
]

REQUIRED_CENTRE_COLUMNS = [
    "boundary_id",
    "centre_name",
    "centre_borough",
    "designation",
    "source_file",
    "source_layer",
    "tier",
    "priority_order",
    "duplicate_source_count",
    "duplicate_sources",
    "area_ha",
    "geometry",
]


def read_json(path):
    """Read a required upstream summary."""

    if not path.is_file():
        raise FileNotFoundError(f"Required upstream summary not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_outputs_do_not_exist():
    """Prevent the reconciled spatial snapshot from being silently replaced."""

    existing_outputs = []

    for output_path in [SPATIAL_OUTPUT_PATH, SUMMARY_OUTPUT_PATH]:
        if output_path.exists():
            existing_outputs.append(str(output_path))

    if existing_outputs:
        raise FileExistsError(
            "Spatial outputs already exist. Review and move them before a new "
            "run:\n- " + "\n- ".join(existing_outputs)
        )


def read_applications(application_summary):
    """
    Read the authoritative eligible application universe.

    The upstream merge summary supplies the expected eligible count. 
    Centroid validity is recalculated here because it belongs to the spatial stage.
    """

    if not APPLICATION_INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Eligible application input not found: {APPLICATION_INPUT_PATH}"
        )

    applications = pd.read_csv(
        APPLICATION_INPUT_PATH,
        dtype={"es_id": "string"},
        low_memory=False,
    )

    missing_columns = [
        column
        for column in REQUIRED_APPLICATION_COLUMNS
        if column not in applications.columns
    ]

    if missing_columns:
        raise ValueError(
            "Eligible application CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    try:
        expected_application_count = int(
            application_summary["final_classification"]["eligible_include"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "merge_summary.json does not contain a valid "
            "final_classification.eligible_include."
        ) from error

    if len(applications) != expected_application_count:
        raise ValueError(
            "Cross-stage application count mismatch: "
            f"merge_summary.json reports {expected_application_count}, "
            f"but pld_clean.csv contains {len(applications)} rows."
        )

    missing_es_id = applications["es_id"].isna() | (
        applications["es_id"].str.strip() == ""
    )

    if missing_es_id.any():
        raise ValueError("Eligible applications contain missing es_id values.")

    if not applications["es_id"].is_unique:
        raise ValueError("Eligible applications contain duplicate es_id values.")

    applications["centroid_easting"] = pd.to_numeric(
        applications["centroid_easting"],
        errors="coerce",
    )
    applications["centroid_northing"] = pd.to_numeric(
        applications["centroid_northing"],
        errors="coerce",
    )

    applications["has_valid_centroid"] = (
        applications["centroid_easting"].between(
            500000,
            570000,
            inclusive="both",
        )
        & applications["centroid_northing"].between(
            150000,
            210000,
            inclusive="both",
        )
    )

    return applications


def read_centres(centre_summary):
    """
    Read the frozen analytical centre universe.
    The centre summary supplies the expected centre count. 
    The CRS and the two fields used for overlap resolution are checked here because 
    they directly determine the spatial assignment.
    """

    if not CENTRE_INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Centre GeoPackage not found: {CENTRE_INPUT_PATH}"
        )

    try:
        centres = gpd.read_file(
            CENTRE_INPUT_PATH,
            layer=CENTRE_LAYER_NAME,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not read centre GeoPackage layer: {error}"
        ) from error

    missing_columns = [
        column
        for column in REQUIRED_CENTRE_COLUMNS
        if column not in centres.columns
    ]

    if missing_columns:
        raise ValueError(
            "Centre GeoPackage is missing required columns: "
            + ", ".join(missing_columns)
        )

    try:
        expected_centre_count = int(
            centre_summary["final_centres"]["final_centre_count"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "centre_summary.json does not contain a valid "
            "final_centres.final_centre_count."
        ) from error

    if len(centres) != expected_centre_count:
        raise ValueError(
            "Cross-stage centre count mismatch: "
            f"centre_summary.json reports {expected_centre_count}, "
            f"but centres.gpkg contains {len(centres)} features."
        )

    if centres.crs is None or centres.crs.to_epsg() != 27700:
        raise ValueError(
            f"Centre CRS is {centres.crs}; expected {TARGET_CRS}."
        )

    centres["priority_order"] = pd.to_numeric(
        centres["priority_order"],
        errors="coerce",
    )
    centres["area_ha"] = pd.to_numeric(
        centres["area_ha"],
        errors="coerce",
    )

    if centres["priority_order"].isna().any():
        raise ValueError("Centre GeoPackage contains missing priority_order.")

    if centres["area_ha"].isna().any():
        raise ValueError("Centre GeoPackage contains missing area_ha.")

    return centres


def choose_one_centre_per_matched_application(matched_rows):
    """
    Resolve applications intersecting more than one analytical centre.
    Methodological priority is applied first. 
    Polygon area is used only when more than one candidate remains at the same priority.
    """

    selected_rows = []
    resolved_by_priority = 0
    resolved_by_area = 0

    for es_id, group in matched_rows.groupby("es_id", sort=False):
        if len(group) == 1:
            selected_rows.append(group.iloc[0].copy())
            continue

        lowest_priority = group["centre_priority_order"].min()
        priority_candidates = group[
            group["centre_priority_order"] == lowest_priority
        ]

        if len(priority_candidates) == 1:
            selected_rows.append(priority_candidates.iloc[0].copy())
            resolved_by_priority += 1
            continue

        smallest_area = priority_candidates["centre_area_ha"].min()
        final_candidates = priority_candidates[
            priority_candidates["centre_area_ha"] == smallest_area
        ]

        if len(final_candidates) != 1:
            tied_boundaries = sorted(
                final_candidates["boundary_id"].astype(str).tolist()
            )
            raise ValueError(
                "Centre assignment remains tied after priority_order and "
                f"area_ha for es_id {es_id}: {tied_boundaries}"
            )

        selected_rows.append(final_candidates.iloc[0].copy())
        resolved_by_area += 1

    if not selected_rows:
        selected = pd.DataFrame(columns=matched_rows.columns)
    else:
        selected = pd.DataFrame(selected_rows)

    resolution_summary = {
        "resolved_by_priority": resolved_by_priority,
        "resolved_by_area_fallback": resolved_by_area,
    }

    return selected, resolution_summary


def build_final_output(applications, selected, raw_match_counts):
    """Attach at most one selected centre while preserving every application."""

    output = applications.copy()

    output["raw_centre_match_count"] = (
        output["es_id"].map(raw_match_counts).fillna(0).astype(int)
    )
    output["had_multiple_matches"] = (
        output["raw_centre_match_count"] > 1
    )

    output["spatial_status"] = "invalid_centroid"

    valid_mask = output["has_valid_centroid"]

    output.loc[
        valid_mask & (output["raw_centre_match_count"] == 0),
        "spatial_status",
    ] = "unmatched"

    output.loc[
        valid_mask & (output["raw_centre_match_count"] > 0),
        "spatial_status",
    ] = "matched"

    centre_columns = [
        "boundary_id",
        "centre_name",
        "centre_borough",
        "centre_designation",
        "centre_source_file",
        "centre_source_layer",
        "centre_tier",
        "centre_priority_order",
        "centre_area_ha",
        "centre_duplicate_source_count",
        "centre_duplicate_sources",
    ]

    for column in centre_columns:
        output[column] = pd.NA

    if not selected.empty:
        selected_indexed = selected.set_index("es_id")

        for column in centre_columns:
            output[column] = output["es_id"].map(
                selected_indexed[column]
            )

    return output


def build_summary(
    applications,
    centres,
    output,
    zero_match_count,
    exactly_one_match_count,
    multiple_match_count,
    resolution_summary,
):
    """Record the spatial-assignment counts and overlap resolution outcomes."""

    valid_count = int(applications["has_valid_centroid"].sum())
    invalid_count = len(applications) - valid_count

    matched = output["spatial_status"] == "matched"
    unmatched = output["spatial_status"] == "unmatched"

    matched_count = int(matched.sum())
    unmatched_count = int(unmatched.sum())

    represented_centre_count = int(
        output.loc[matched, "boundary_id"].nunique()
    )

    matched_tier_counts = (
        output.loc[matched, "centre_tier"]
        .value_counts()
        .sort_index()
    )

    return {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "inputs": {
            "application_csv": str(APPLICATION_INPUT_PATH),
            "application_summary": str(APPLICATION_SUMMARY_PATH),
            "centre_gpkg": str(CENTRE_INPUT_PATH),
            "centre_summary": str(CENTRE_SUMMARY_PATH),
            "input_eligible_applications": len(applications),
            "available_centre_count": len(centres),
            "crs": TARGET_CRS,
            "spatial_predicate": "intersects",
        },
        "centroid_counts": {
            "valid_centroid": valid_count,
            "invalid_centroid": invalid_count,
        },
        "raw_match_counts": {
            "zero_match_applications": zero_match_count,
            "exactly_one_match_applications": exactly_one_match_count,
            "multiple_match_applications": multiple_match_count,
        },
        "multiple_match_resolution": {
            "resolved_by_priority": resolution_summary[
                "resolved_by_priority"
            ],
            "resolved_by_area_fallback": resolution_summary[
                "resolved_by_area_fallback"
            ],
        },
        "final_assignment": {
            "matched_applications": matched_count,
            "unmatched_applications": unmatched_count,
            "represented_centre_count": represented_centre_count,
            "matched_tier_counts": {
                str(tier): int(count)
                for tier, count in matched_tier_counts.items()
            },
        },
    }


def main():
    """Create the final application level centre-spatial output."""

    ensure_outputs_do_not_exist()

    application_summary = read_json(APPLICATION_SUMMARY_PATH)
    centre_summary = read_json(CENTRE_SUMMARY_PATH)

    applications = read_applications(application_summary)
    centres = read_centres(centre_summary)

    valid = applications[
        applications["has_valid_centroid"]
    ].copy()

    points = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(
            valid["centroid_easting"],
            valid["centroid_northing"],
        ),
        crs=TARGET_CRS,
    )

    centres_for_join = centres[
        [
            "boundary_id",
            "centre_name",
            "centre_borough",
            "designation",
            "source_file",
            "source_layer",
            "tier",
            "priority_order",
            "duplicate_source_count",
            "duplicate_sources",
            "area_ha",
            "geometry",
        ]
    ].rename(
        columns={
            "designation": "centre_designation",
            "source_file": "centre_source_file",
            "source_layer": "centre_source_layer",
            "tier": "centre_tier",
            "priority_order": "centre_priority_order",
            "duplicate_source_count": "centre_duplicate_source_count",
            "duplicate_sources": "centre_duplicate_sources",
            "area_ha": "centre_area_ha",
        }
    )

    joined = gpd.sjoin(
        points,
        centres_for_join,
        how="left",
        predicate="intersects",
    )

    matched_rows = joined[
        joined["boundary_id"].notna()
    ].copy()

    raw_match_counts = matched_rows.groupby("es_id").size()

    valid_match_counts = (
        valid["es_id"].map(raw_match_counts).fillna(0).astype(int)
    )

    zero_match_count = int((valid_match_counts == 0).sum())
    exactly_one_match_count = int((valid_match_counts == 1).sum())
    multiple_match_count = int((valid_match_counts > 1).sum())

    if (
        zero_match_count
        + exactly_one_match_count
        + multiple_match_count
        != len(valid)
    ):
        raise ValueError(
            "Raw spatial-match groups do not reconcile to the valid-centroid "
            "application universe."
        )

    selected, resolution_summary = (
        choose_one_centre_per_matched_application(matched_rows)
    )

    matched_application_count = (
        exactly_one_match_count + multiple_match_count
    )

    if len(selected) != matched_application_count:
        raise ValueError(
            "Resolved centre assignments do not reconcile to applications "
            "with at least one raw centre match."
        )

    output = build_final_output(
        applications,
        selected,
        raw_match_counts,
    )

    matched_count = int(
        (output["spatial_status"] == "matched").sum()
    )
    unmatched_count = int(
        (output["spatial_status"] == "unmatched").sum()
    )
    invalid_count = int(
        (output["spatial_status"] == "invalid_centroid").sum()
    )

    if (
        matched_count + unmatched_count + invalid_count
        != len(applications)
    ):
        raise ValueError(
            "Final spatial statuses do not reconcile to the complete eligible "
            "application universe."
        )

    summary = build_summary(
        applications,
        centres,
        output,
        zero_match_count,
        exactly_one_match_count,
        multiple_match_count,
        resolution_summary,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        SPATIAL_OUTPUT_PATH,
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

    print(f"Eligible applications: {len(applications)}")
    print(
        "Valid centroids: "
        f"{summary['centroid_counts']['valid_centroid']}"
    )
    print(
        "Invalid centroids: "
        f"{summary['centroid_counts']['invalid_centroid']}"
    )
    print(
        "Matched applications: "
        f"{summary['final_assignment']['matched_applications']}"
    )
    print(
        "Unmatched applications: "
        f"{summary['final_assignment']['unmatched_applications']}"
    )
    print(
        "Represented centres: "
        f"{summary['final_assignment']['represented_centre_count']}"
    )
    print(f"Spatial output: {SPATIAL_OUTPUT_PATH}")
    print(f"Spatial summary: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
