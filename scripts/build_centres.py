"""
Build the analytical centre-boundary dataset from the freshly reviewed
Local Plan source-layer manifest.

35 active Local Plan GeoPackages
- complete centre_layer_inventory.csv
- fresh manual centre_layer_manifest.csv
- centres.gpkg + centre_summary.json

The inventory defines the complete source-layer universe presented for manual
review. The manifest records the researcher's reviewed methodological decision
about which layers represent analytical centres. 

This script checks that the
manifest covers the complete inventory, then implements those reviewed decisions.
"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import force_2d, make_valid, normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_PLAN_DIRECTORY = PROJECT_ROOT / "data_raw" / "local_plan"

INVENTORY_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "centres"
    / "centre_layer_inventory.csv"
)

MANIFEST_PATH = PROJECT_ROOT / "manual_inputs" / "centre_layer_manifest.csv"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "centres"
CENTRES_OUTPUT_PATH = OUTPUT_DIRECTORY / "centres.gpkg"
SUMMARY_OUTPUT_PATH = OUTPUT_DIRECTORY / "centre_summary.json"

OUTPUT_LAYER_NAME = "centres"
TARGET_CRS = "EPSG:27700"

EXPECTED_GPKG_FILES = [
    f"planning_local_plan_data_{number:02d}.gpkg"
    for number in range(1, 36)
]

REQUIRED_INVENTORY_COLUMNS = [
    "source_file",
    "source_layer",
]

REQUIRED_MANIFEST_COLUMNS = [
    "source_file",
    "source_layer",
    "include",
    "tier",
    "priority_order",
    "selection_reason",
]

NAME_COLUMN_CANDIDATES = [
    "sitename",
    "site_name",
    "name",
    "centre_name",
    "town_centre",
    "town_centre_name",
    "designation",
    "layer",
]

BOROUGH_COLUMN_CANDIDATES = [
    "borough",
    "local_authority",
    "planningauthority",
    "planning_authority",
    "lpa_name",
    "authority",
]

DESIGNATION_COLUMN_CANDIDATES = [
    "designation",
    "boroughdesignation",
    "type",
    "category",
    "class",
    "policy",
]


def value_is_missing(value):
    """Treat source nulls and blank text consistently."""

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False

    return str(value).strip() == ""


def parse_include(value):
    """Require an explicit Boolean like manifest decision."""

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    raise ValueError(f"Unrecognised manifest include value: {value}")


def first_nonempty_source_value(row, lower_column_map, candidates):
    """Return the first populated value from a short provenance field list."""

    for candidate in candidates:
        actual_column = lower_column_map.get(candidate.lower())

        if actual_column is None:
            continue

        value = row[actual_column]

        if not value_is_missing(value):
            return str(value).strip()

    return ""


def ensure_outputs_do_not_exist():
    """Prevent a reviewed centre snapshot from being silently replaced."""

    existing_outputs = []

    for output_path in [CENTRES_OUTPUT_PATH, SUMMARY_OUTPUT_PATH]:
        if output_path.exists():
            existing_outputs.append(str(output_path))

    if existing_outputs:
        raise FileExistsError(
            "Centre outputs already exist. Review and move them before a new "
            "run:\n- " + "\n- ".join(existing_outputs)
        )


def check_expected_source_files():
    """Require the complete and exact frozen set of 35 GeoPackages."""

    if not LOCAL_PLAN_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Local Plan directory not found: {LOCAL_PLAN_DIRECTORY}"
        )

    found_files = sorted(
        path.name for path in LOCAL_PLAN_DIRECTORY.glob("*.gpkg")
    )
    expected_files = sorted(EXPECTED_GPKG_FILES)

    if found_files != expected_files:
        missing_files = sorted(set(expected_files) - set(found_files))
        unexpected_files = sorted(set(found_files) - set(expected_files))

        raise ValueError(
            "The Local Plan folder does not contain exactly the expected 35 "
            "GeoPackages. "
            f"Missing: {missing_files}. Unexpected: {unexpected_files}."
        )

    return found_files


def read_inventory():
    """Read the complete source-layer inventory used for manual review."""

    if not INVENTORY_PATH.is_file():
        raise FileNotFoundError(
            f"Centre-layer inventory not found: {INVENTORY_PATH}"
        )

    inventory = pd.read_csv(
        INVENTORY_PATH,
        dtype="string",
        low_memory=False,
    )

    missing_columns = [
        column
        for column in REQUIRED_INVENTORY_COLUMNS
        if column not in inventory.columns
    ]

    if missing_columns:
        raise ValueError(
            "Centre-layer inventory is missing required columns: "
            + ", ".join(missing_columns)
        )

    if inventory.empty:
        raise ValueError("Centre-layer inventory is empty.")

    if inventory.duplicated(
        subset=["source_file", "source_layer"],
        keep=False,
    ).any():
        raise ValueError(
            "Centre-layer inventory contains duplicate source-file/layer keys."
        )

    return inventory


def read_manifest(inventory):
    """
    Read the fresh manual review and prove it covers the complete inventory
    Every inventoried source-file/layer key must have exactly one manual
    INCLUDE or EXCLUDE decision.
    """

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Centre manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype="string",
        low_memory=False,
    )

    missing_columns = [
        column
        for column in REQUIRED_MANIFEST_COLUMNS
        if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            "Centre manifest is missing required columns: "
            + ", ".join(missing_columns)
        )

    if manifest.empty:
        raise ValueError("Centre manifest is empty.")

    manifest["include"] = manifest["include"].apply(parse_include)
    manifest["priority_order"] = pd.to_numeric(
        manifest["priority_order"],
        errors="coerce",
    )

    if manifest["priority_order"].isna().any():
        raise ValueError("Centre manifest contains a missing priority_order.")

    manifest["priority_order"] = manifest["priority_order"].astype(int)

    if manifest.duplicated(
        subset=["source_file", "source_layer"],
        keep=False,
    ).any():
        raise ValueError(
            "Centre manifest contains duplicate source-file/layer decisions."
        )

    inventory_keys = set(
        inventory[["source_file", "source_layer"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    manifest_keys = set(
        manifest[["source_file", "source_layer"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )

    if manifest_keys != inventory_keys:
        missing_from_manifest = sorted(inventory_keys - manifest_keys)
        extra_in_manifest = sorted(manifest_keys - inventory_keys)

        raise ValueError(
            "Fresh manifest does not exactly cover the complete source-layer "
            "inventory. "
            f"Missing decisions={missing_from_manifest[:10]}; "
            f"extra decisions={extra_in_manifest[:10]}."
        )

    missing_reasons = manifest["selection_reason"].isna() | (
        manifest["selection_reason"].str.strip() == ""
    )

    if missing_reasons.any():
        raise ValueError("Every manifest row must have a selection_reason.")

    selected = manifest[manifest["include"]].copy()

    if selected.empty:
        raise ValueError("Centre manifest selects no source layers.")

    if (~selected["tier"].isin(["tier1", "tier2"])).any():
        raise ValueError("Included manifest rows must use tier1 or tier2.")

    wrong_tier1_priority = (
        (selected["tier"] == "tier1")
        & (selected["priority_order"] != 1)
    )
    wrong_tier2_priority = (
        (selected["tier"] == "tier2")
        & (selected["priority_order"] != 2)
    )

    if wrong_tier1_priority.any():
        raise ValueError("Tier-1 manifest rows must use priority_order=1.")

    if wrong_tier2_priority.any():
        raise ValueError("Tier-2 manifest rows must use priority_order=2.")

    return manifest, selected


def clean_polygon_geometry(geometry, source_record_id):
    """
    Normalise one geometry and return its cleaning outcome.
    Points and lines are never converted into polygons. 
    Invalid source polygons are repaired, 
    and any geometry that still cannot provide a valid Polygon or
    MultiPolygon is counted as dropped.
    """

    if geometry is None:
        return None, "null_geometry", False, False

    if geometry.is_empty:
        return None, "empty_geometry", False, False

    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        return None, "non_polygon_geometry", False, False

    had_z_coordinate = bool(geometry.has_z)

    try:
        cleaned_geometry = force_2d(geometry)
    except Exception as error:
        raise ValueError(
            f"Could not force geometry to 2D for {source_record_id}: {error}"
        ) from error

    was_invalid = not cleaned_geometry.is_valid

    if was_invalid:
        try:
            cleaned_geometry = make_valid(cleaned_geometry)
        except Exception as error:
            raise ValueError(
                f"Could not repair invalid geometry for {source_record_id}: "
                f"{error}"
            ) from error

        if (
            cleaned_geometry.geom_type not in {"Polygon", "MultiPolygon"}
            or not cleaned_geometry.is_valid
        ):
            try:
                cleaned_geometry = cleaned_geometry.buffer(0)
            except Exception as error:
                raise ValueError(
                    f"Fallback geometry repair failed for {source_record_id}: "
                    f"{error}"
                ) from error

    if (
        cleaned_geometry is None
        or cleaned_geometry.is_empty
        or cleaned_geometry.geom_type not in {"Polygon", "MultiPolygon"}
        or not cleaned_geometry.is_valid
    ):
        return None, "dropped_after_cleaning", was_invalid, had_z_coordinate

    try:
        cleaned_geometry = normalize(cleaned_geometry)
    except Exception as error:
        raise ValueError(
            f"Could not normalise geometry for {source_record_id}: {error}"
        ) from error

    return cleaned_geometry, "retained", was_invalid, had_z_coordinate


def read_and_clean_layer(manifest_row):
    """Read one approved source layer and retain auditable polygon records."""

    source_file = str(manifest_row["source_file"])
    source_layer = str(manifest_row["source_layer"])
    source_path = LOCAL_PLAN_DIRECTORY / source_file

    try:
        source = gpd.read_file(source_path, layer=source_layer)
    except Exception as error:
        raise RuntimeError(
            f"Failed to read selected layer {source_file} / {source_layer}: "
            f"{error}"
        ) from error

    if source.crs is None:
        raise ValueError(
            f"Selected layer has no CRS: {source_file} / {source_layer}"
        )

    source_crs = str(source.crs)
    raw_feature_count = len(source)

    try:
        source = source.to_crs(TARGET_CRS)
    except Exception as error:
        raise RuntimeError(
            f"Failed to reproject {source_file} / {source_layer} to "
            f"{TARGET_CRS}: {error}"
        ) from error

    lower_column_map = {
        str(column).lower(): column
        for column in source.columns
    }

    records = []
    null_geometry_count = 0
    empty_geometry_count = 0
    non_polygon_geometry_count = 0
    forced_to_2d_count = 0
    invalid_geometry_count = 0
    repaired_geometry_count = 0
    dropped_after_cleaning_count = 0

    for source_row in range(raw_feature_count):
        row = source.iloc[source_row]
        geometry = row.geometry
        source_record_id = (
            f"{Path(source_file).stem}__{source_layer}__{source_row:06d}"
        )

        original_geometry_type = ""
        if geometry is not None:
            original_geometry_type = geometry.geom_type

        (
            cleaned_geometry,
            outcome,
            was_invalid,
            had_z_coordinate,
        ) = clean_polygon_geometry(
            geometry,
            source_record_id,
        )

        if had_z_coordinate:
            forced_to_2d_count += 1

        if outcome == "null_geometry":
            null_geometry_count += 1
            continue

        if outcome == "empty_geometry":
            empty_geometry_count += 1
            continue

        if outcome == "non_polygon_geometry":
            non_polygon_geometry_count += 1
            continue

        if was_invalid:
            invalid_geometry_count += 1

        if outcome == "dropped_after_cleaning":
            dropped_after_cleaning_count += 1
            continue

        if was_invalid:
            repaired_geometry_count += 1

        centre_name = first_nonempty_source_value(
            row,
            lower_column_map,
            NAME_COLUMN_CANDIDATES,
        )
        borough = first_nonempty_source_value(
            row,
            lower_column_map,
            BOROUGH_COLUMN_CANDIDATES,
        )
        designation = first_nonempty_source_value(
            row,
            lower_column_map,
            DESIGNATION_COLUMN_CANDIDATES,
        )

        if not designation:
            designation = source_layer

        records.append(
            {
                "source_record_id": source_record_id,
                "source_file": source_file,
                "source_layer": source_layer,
                "source_row": source_row,
                "centre_name": centre_name,
                "centre_borough": borough,
                "designation": designation,
                "tier": str(manifest_row["tier"]),
                "priority_order": int(manifest_row["priority_order"]),
                "selection_reason": str(manifest_row["selection_reason"]),
                "geom_type_original": original_geometry_type,
                "geometry": cleaned_geometry,
            }
        )

    retained_feature_count = len(records)

    if retained_feature_count == 0:
        raise ValueError(
            "Selected layer retained no polygon features after cleaning: "
            f"{source_file} / {source_layer}"
        )

    partition_total = (
        null_geometry_count
        + empty_geometry_count
        + non_polygon_geometry_count
        + dropped_after_cleaning_count
        + retained_feature_count
    )

    if partition_total != raw_feature_count:
        raise ValueError(
            f"Geometry-cleaning counts do not reconcile for "
            f"{source_file} / {source_layer}: "
            f"{partition_total} of {raw_feature_count}."
        )

    layer_summary = {
        "source_file": source_file,
        "source_layer": source_layer,
        "source_crs": source_crs,
        "raw_feature_count": raw_feature_count,
        "null_geometry_count": null_geometry_count,
        "empty_geometry_count": empty_geometry_count,
        "non_polygon_geometry_count": non_polygon_geometry_count,
        "forced_to_2d_count": forced_to_2d_count,
        "invalid_geometry_count": invalid_geometry_count,
        "repaired_geometry_count": repaired_geometry_count,
        "dropped_after_cleaning_count": dropped_after_cleaning_count,
        "retained_feature_count": retained_feature_count,
    }

    frame = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=TARGET_CRS,
    )

    return frame, layer_summary


def select_canonical_exact_duplicates(combined):
    """
    Retain one canonical source record for each exact normalised geometry.
    Tier 1 is preferred to an exactly coincident Tier 2 fallback. 
    Source file, layer and source-row order are deterministic tie-breaks only within the
    same methodological priority.
    """

    combined = combined.copy()
    combined["wkt_temp"] = combined.geometry.to_wkt(
        rounding_precision=-1
    )

    combined = combined.sort_values(
        [
            "wkt_temp",
            "priority_order",
            "source_file",
            "source_layer",
            "source_row",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    canonical_rows = []
    exact_duplicate_group_count = 0
    records_in_duplicate_groups = 0
    records_removed_as_exact_duplicates = 0

    for _, group in combined.groupby("wkt_temp", sort=False):
        group_size = len(group)

        if group_size > 1:
            exact_duplicate_group_count += 1
            records_in_duplicate_groups += group_size
            records_removed_as_exact_duplicates += group_size - 1

        canonical = group.iloc[0].copy()
        canonical["duplicate_source_count"] = group_size
        canonical["duplicate_sources"] = " | ".join(
            group["source_record_id"].astype(str).tolist()
        )
        canonical_rows.append(canonical)

    centres = gpd.GeoDataFrame(
        pd.DataFrame(canonical_rows),
        geometry="geometry",
        crs=TARGET_CRS,
    )

    centres["boundary_id"] = centres["source_record_id"]
    centres["area_m2"] = centres.geometry.area
    centres["area_ha"] = centres["area_m2"] / 10000
    centres = centres.drop(columns="wkt_temp")

    centres = centres.sort_values(
        [
            "priority_order",
            "source_file",
            "source_layer",
            "source_row",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    duplicate_summary = {
        "exact_duplicate_group_count": exact_duplicate_group_count,
        "records_in_exact_duplicate_groups": records_in_duplicate_groups,
        "records_removed_as_exact_duplicates": (
            records_removed_as_exact_duplicates
        ),
    }

    return centres, duplicate_summary


def build_summary(
    found_files,
    inventory,
    manifest,
    selected,
    layer_summaries,
    duplicate_summary,
    centres,
):
    """Record the centre-build lineage and transformation counts."""

    raw_feature_count = sum(
        row["raw_feature_count"]
        for row in layer_summaries
    )
    null_geometry_count = sum(
        row["null_geometry_count"]
        for row in layer_summaries
    )
    empty_geometry_count = sum(
        row["empty_geometry_count"]
        for row in layer_summaries
    )
    non_polygon_geometry_count = sum(
        row["non_polygon_geometry_count"]
        for row in layer_summaries
    )
    forced_to_2d_count = sum(
        row["forced_to_2d_count"]
        for row in layer_summaries
    )
    invalid_geometry_count = sum(
        row["invalid_geometry_count"]
        for row in layer_summaries
    )
    repaired_geometry_count = sum(
        row["repaired_geometry_count"]
        for row in layer_summaries
    )
    dropped_after_cleaning_count = sum(
        row["dropped_after_cleaning_count"]
        for row in layer_summaries
    )
    retained_before_duplicates = sum(
        row["retained_feature_count"]
        for row in layer_summaries
    )

    selected_tier_counts = (
        selected["tier"].value_counts().sort_index()
    )
    final_tier_counts = (
        centres["tier"].value_counts().sort_index()
    )

    return {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "source_files": {
            "expected_gpkg_count": len(EXPECTED_GPKG_FILES),
            "found_gpkg_count": len(found_files),
        },
        "inventory_lineage": {
            "inventory_csv": str(INVENTORY_PATH),
            "inventory_rows": len(inventory),
            "manifest_csv": str(MANIFEST_PATH),
            "manifest_rows": len(manifest),
        },
        "manifest": {
            "selected_layer_count": len(selected),
            "excluded_layer_count": int((~manifest["include"]).sum()),
            "selected_layer_tier_counts": {
                str(tier): int(count)
                for tier, count in selected_tier_counts.items()
            },
        },
        "source_features": {
            "raw_selected_source_features": raw_feature_count,
            "null_geometry_count": null_geometry_count,
            "empty_geometry_count": empty_geometry_count,
            "non_polygon_geometry_count": non_polygon_geometry_count,
            "forced_to_2d_count": forced_to_2d_count,
            "invalid_geometry_count": invalid_geometry_count,
            "repaired_geometry_count": repaired_geometry_count,
            "dropped_after_cleaning_count": dropped_after_cleaning_count,
            "retained_before_exact_duplicates": retained_before_duplicates,
        },
        "per_layer_counts": layer_summaries,
        "exact_duplicates": duplicate_summary,
        "final_centres": {
            "final_centre_count": len(centres),
            "tier_counts": {
                str(tier): int(count)
                for tier, count in final_tier_counts.items()
            },
            "crs": TARGET_CRS,
        },
    }


def main():
    """Build, reconcile, and save the final centre outputs."""

    ensure_outputs_do_not_exist()

    found_files = check_expected_source_files()
    inventory = read_inventory()
    manifest, selected = read_manifest(inventory)

    frames = []
    layer_summaries = []

    selected = selected.sort_values(
        ["priority_order", "source_file", "source_layer"],
        kind="mergesort",
    )

    for _, manifest_row in selected.iterrows():
        frame, layer_summary = read_and_clean_layer(manifest_row)
        frames.append(frame)
        layer_summaries.append(layer_summary)

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=TARGET_CRS,
    )

    combined_count = len(combined)

    centres, duplicate_summary = select_canonical_exact_duplicates(
        combined
    )

    removed_duplicates = duplicate_summary[
        "records_removed_as_exact_duplicates"
    ]

    if combined_count != len(centres) + removed_duplicates:
        raise ValueError(
            "Exact-duplicate removal does not reconcile: "
            f"{combined_count} != {len(centres)} + {removed_duplicates}."
        )

    if not centres["boundary_id"].is_unique:
        raise ValueError("Final boundary_id values are not unique.")

    summary = build_summary(
        found_files,
        inventory,
        manifest,
        selected,
        layer_summaries,
        duplicate_summary,
        centres,
    )

    output_columns = [
        "boundary_id",
        "source_record_id",
        "centre_name",
        "centre_borough",
        "designation",
        "source_file",
        "source_layer",
        "source_row",
        "tier",
        "priority_order",
        "selection_reason",
        "geom_type_original",
        "duplicate_source_count",
        "duplicate_sources",
        "area_m2",
        "area_ha",
        "geometry",
    ]

    centres = centres[output_columns].copy()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    centres.to_file(
        CENTRES_OUTPUT_PATH,
        layer=OUTPUT_LAYER_NAME,
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

    print(f"Complete inventoried layers: {len(inventory)}")
    print(f"Fresh reviewed manifest rows: {len(manifest)}")
    print(f"Selected source layers: {len(selected)}")
    print(f"Saved analytical centre boundaries: {len(centres)}")
    print(f"Centre GeoPackage: {CENTRES_OUTPUT_PATH}")
    print(f"Centre summary: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()