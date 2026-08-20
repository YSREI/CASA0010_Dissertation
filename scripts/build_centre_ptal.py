"""
Construct centre-level public transport accessibility from the TfL 2015 PTAL 100m grid.

The 2015 source provides Access Index (AI) values for a London-wide 100m grid.
This script reconstructs the grid-square support represented by the published locations, 
intersects those cells with the frozen analytical centre boundaries,
and calculates an area-weighted mean Access Index for each centre-boundary feature.

This stage is independent of the centre-period panel.
"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CENTRE_INPUT_PATH = (
    PROJECT_ROOT / "data_processed" / "centres" / "centres.gpkg"
)
CENTRE_SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "centres" / "centre_summary.json"
)
CENTRE_LAYER_NAME = "centres"

PTAL_INPUT_PATH = (
    PROJECT_ROOT
    / "data_raw"
    / "ptal"
    / "2015  PTALs Grid Values"
    / "2015  PTALs Grid Values 280515.xlsx"
)
PTAL_SHEET_NAME = "Query1"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "ptal"
CENTRE_PTAL_OUTPUT_PATH = OUTPUT_DIRECTORY / "centre_ptal.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIRECTORY / "centre_ptal_summary.json"

TARGET_CRS = "EPSG:27700"

GRID_SIZE_M = 100
HALF_GRID_M = GRID_SIZE_M / 2
EXPECTED_PTAL_ROWS = 159451

REQUIRED_CENTRE_COLUMNS = [
    "boundary_id",
    "centre_name",
    "centre_borough",
    "tier",
    "area_m2",
    "geometry",
]

REQUIRED_PTAL_COLUMNS = [
    "ID",
    "X",
    "Y",
    "AI2015",
    "PTAL2015",
]


def read_json(path):
    """Read a required upstream summary."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required upstream summary not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_outputs_do_not_exist():
    """Prevent the PTAL snapshot from being silently replaced."""

    existing = []

    for path in [
        CENTRE_PTAL_OUTPUT_PATH,
        SUMMARY_OUTPUT_PATH,
    ]:
        if path.exists():
            existing.append(str(path))

    if existing:
        raise FileExistsError(
            "PTAL outputs already exist. "
            "Review and move them before rerunning:\n- "
            + "\n- ".join(existing)
        )


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


def ai_to_ptal(ai):
    """Convert Access Index to the official PTAL band."""

    if ai == 0:
        return "0"
    if ai <= 2.5:
        return "1a"
    if ai <= 5:
        return "1b"
    if ai <= 10:
        return "2"
    if ai <= 15:
        return "3"
    if ai <= 20:
        return "4"
    if ai <= 25:
        return "5"
    if ai <= 40:
        return "6a"
    return "6b"


def read_centres(centre_summary):
    """Read the frozen centre universe and reconcile its row count."""

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

    if centres.crs is None or centres.crs.to_epsg() != 27700:
        raise ValueError(
            f"Centre CRS is {centres.crs}; expected {TARGET_CRS}."
        )

    return centres


def read_ptal_grid():
    """Read and validate the exact TfL 2015 PTAL grid snapshot."""

    if not PTAL_INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Required PTAL grid not found: {PTAL_INPUT_PATH}"
        )

    ptal = pd.read_excel(
        PTAL_INPUT_PATH,
        sheet_name=PTAL_SHEET_NAME,
    )

    require_columns(
        ptal,
        REQUIRED_PTAL_COLUMNS,
        "TfL PTAL grid",
    )

    if len(ptal) != EXPECTED_PTAL_ROWS:
        raise ValueError(
            "TfL PTAL source row count does not match the frozen "
            f"2015 snapshot: {len(ptal)} != {EXPECTED_PTAL_ROWS}."
        )

    if ptal["ID"].isna().any() or ptal["ID"].duplicated().any():
        raise ValueError(
            "TfL PTAL grid requires a non-null unique ID."
        )

    for column in ["X", "Y", "AI2015"]:
        ptal[column] = pd.to_numeric(
            ptal[column],
            errors="raise",
        )

    if ptal[["X", "Y", "AI2015", "PTAL2015"]].isna().any().any():
        raise ValueError(
            "TfL PTAL grid contains missing X, Y, AI2015 or PTAL2015 values."
        )

    if (ptal["AI2015"] < 0).any():
        raise ValueError(
            "TfL PTAL grid contains a negative Access Index."
        )

    if ptal.duplicated(["X", "Y"]).any():
        raise ValueError(
            "TfL PTAL grid contains duplicate published X/Y coordinates."
        )

    ptal["ptal_source_category"] = (
        ptal["PTAL2015"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    ptal["ptal_category_from_ai"] = (
        ptal["AI2015"].apply(ai_to_ptal)
    )

    mismatch = (
        ptal["ptal_source_category"]
        != ptal["ptal_category_from_ai"]
    )

    if mismatch.any():
        raise ValueError(
            "AI2015 and PTAL2015 are inconsistent in "
            f"{int(mismatch.sum())} source rows."
        )

    return ptal


def reconstruct_grid_cells(ptal):
    """
    Reconstruct the 100m grid-square support.
    The modal coordinate residues identify the regular grid centres. 
    A half-open cell convention assigns published edge locations to the 
    corresponding 100m lattice cell deterministically.
    """

    ptal = ptal.copy()

    x_centre_residue = int(
        (ptal["X"] % GRID_SIZE_M).mode().iloc[0]
    )
    y_centre_residue = int(
        (ptal["Y"] % GRID_SIZE_M).mode().iloc[0]
    )

    x_boundary_residue = int(
        (x_centre_residue - HALF_GRID_M)
        % GRID_SIZE_M
    )
    y_boundary_residue = int(
        (y_centre_residue - HALF_GRID_M)
        % GRID_SIZE_M
    )

    ptal["grid_x"] = (
        (
            (ptal["X"] - x_boundary_residue)
            // GRID_SIZE_M
        )
        * GRID_SIZE_M
        + x_boundary_residue
        + HALF_GRID_M
    ).astype(int)

    ptal["grid_y"] = (
        (
            (ptal["Y"] - y_boundary_residue)
            // GRID_SIZE_M
        )
        * GRID_SIZE_M
        + y_boundary_residue
        + HALF_GRID_M
    ).astype(int)

    if (
        (ptal["X"] - ptal["grid_x"]).abs() > HALF_GRID_M
    ).any() or (
        (ptal["Y"] - ptal["grid_y"]).abs() > HALF_GRID_M
    ).any():
        raise ValueError(
            "A TfL source point falls outside its reconstructed 100m cell."
        )

    cell_counts = (
        ptal.groupby(["grid_x", "grid_y"])
        .size()
        .rename("source_rows")
        .reset_index()
    )

    duplicate_cells = cell_counts[
        cell_counts["source_rows"] > 1
    ].copy()

    single_cell_keys = cell_counts[
        cell_counts["source_rows"] == 1
    ][["grid_x", "grid_y"]]

    cells = ptal.merge(
        single_cell_keys,
        on=["grid_x", "grid_y"],
        how="inner",
    ).copy()

    cells = gpd.GeoDataFrame(
        cells[
            [
                "ID",
                "grid_x",
                "grid_y",
                "AI2015",
                "ptal_source_category",
            ]
        ].copy(),
        geometry=[
            box(
                x - HALF_GRID_M,
                y - HALF_GRID_M,
                x + HALF_GRID_M,
                y + HALF_GRID_M,
            )
            for x, y in zip(
                cells["grid_x"],
                cells["grid_y"],
            )
        ],
        crs=TARGET_CRS,
    )

    duplicate_cells = gpd.GeoDataFrame(
        duplicate_cells.copy(),
        geometry=[
            box(
                x - HALF_GRID_M,
                y - HALF_GRID_M,
                x + HALF_GRID_M,
                y + HALF_GRID_M,
            )
            for x, y in zip(
                duplicate_cells["grid_x"],
                duplicate_cells["grid_y"],
            )
        ],
        crs=TARGET_CRS,
    )

    reconstruction = {
        "grid_size_m": GRID_SIZE_M,
        "grid_x_centre_residue": x_centre_residue,
        "grid_y_centre_residue": y_centre_residue,
        "source_rows": len(ptal),
        "unique_reconstructed_cells": len(cell_counts),
        "single_source_cells": len(single_cell_keys),
        "duplicate_reconstructed_cell_groups": len(
            duplicate_cells
        ),
    }

    return cells, duplicate_cells, reconstruction


def check_duplicate_cells(centres, duplicate_cells):
    """
    Confirm duplicate reconstructed edge cells do not overlap study centres.
    A duplicate cell that overlaps a centre would require targeted inspection
    before any AI value could be assigned to that cell.
    """

    if duplicate_cells.empty:
        return

    candidates = gpd.sjoin(
        centres[["boundary_id", "geometry"]],
        duplicate_cells[
            ["grid_x", "grid_y", "geometry"]
        ],
        how="inner",
        predicate="intersects",
    )

    positive = []

    for centre_index, row in candidates.iterrows():
        cell_geometry = duplicate_cells.loc[
            row["index_right"],
            "geometry",
        ]

        overlap_area = (
            centres.loc[centre_index, "geometry"]
            .intersection(cell_geometry)
            .area
        )

        if overlap_area > 0:
            positive.append(
                {
                    "boundary_id": row["boundary_id"],
                    "grid_x": int(row["grid_x"]),
                    "grid_y": int(row["grid_y"]),
                    "overlap_area_m2": float(overlap_area),
                }
            )

    if positive:
        raise ValueError(
            "A reconstructed PTAL cell with multiple source rows "
            "overlaps an analytical centre:\n"
            + pd.DataFrame(positive)
            .head(20)
            .to_string(index=False)
        )


def build_centre_ptal(centres, cells):
    """Calculate area-weighted mean Access Index for each centre boundary."""

    centres_small = centres[
        [
            "boundary_id",
            "centre_name",
            "centre_borough",
            "tier",
            "area_m2",
            "geometry",
        ]
    ].copy()

    candidates = gpd.sjoin(
        centres_small,
        cells[
            [
                "ID",
                "AI2015",
                "geometry",
            ]
        ],
        how="left",
        predicate="intersects",
    )

    candidates = (
        candidates[
            candidates["ID"].notna()
        ]
        .reset_index()
        .rename(
            columns={
                "index": "centre_index",
                "index_right": "cell_index",
            }
        )
    )

    rows = []

    for _, row in candidates.iterrows():
        centre_geometry = centres.loc[
            row["centre_index"],
            "geometry",
        ]
        cell_geometry = cells.loc[
            int(row["cell_index"]),
            "geometry",
        ]

        intersection_area = (
            centre_geometry
            .intersection(cell_geometry)
            .area
        )

        if intersection_area <= 0:
            continue

        rows.append(
            {
                "boundary_id": row["boundary_id"],
                "AI2015": float(row["AI2015"]),
                "intersection_area_m2": float(
                    intersection_area
                ),
            }
        )

    intersections = pd.DataFrame(rows)

    if intersections.empty:
        raise ValueError(
            "No positive-area intersections were found between "
            "centre boundaries and reconstructed PTAL cells."
        )

    intersections["weighted_ai_component"] = (
        intersections["AI2015"]
        * intersections["intersection_area_m2"]
    )

    aggregated = (
        intersections.groupby("boundary_id")
        .agg(
            ptal_cell_count=(
                "AI2015",
                "size",
            ),
            ptal_covered_area_m2=(
                "intersection_area_m2",
                "sum",
            ),
            weighted_ai_total=(
                "weighted_ai_component",
                "sum",
            ),
        )
        .reset_index()
    )

    centre_ptal = centres_small.drop(
        columns="geometry"
    ).merge(
        aggregated,
        on="boundary_id",
        how="left",
    )

    centre_ptal["ptal_cell_count"] = (
        centre_ptal["ptal_cell_count"]
        .fillna(0)
        .astype(int)
    )

    centre_ptal["ptal_covered_area_m2"] = (
        centre_ptal["ptal_covered_area_m2"]
        .fillna(0.0)
    )

    centre_ptal["ptal_coverage_share"] = (
        centre_ptal["ptal_covered_area_m2"]
        / centre_ptal["area_m2"]
    )

    if (
        centre_ptal["ptal_coverage_share"] > 1.000001
    ).any():
        raise ValueError(
            "Reconstructed PTAL coverage exceeds centre area."
        )

    if (
        centre_ptal["ptal_covered_area_m2"] == 0
    ).any():
        raise ValueError(
            "At least one analytical centre has zero PTAL coverage."
        )

    centre_ptal["ptal_mean_ai"] = (
        centre_ptal["weighted_ai_total"]
        / centre_ptal["ptal_covered_area_m2"]
    )

    centre_ptal["ptal_mean_category"] = (
        centre_ptal["ptal_mean_ai"].apply(
            ai_to_ptal
        )
    )

    centre_ptal = centre_ptal.drop(
        columns="weighted_ai_total"
    )

    if len(centre_ptal) != len(centres):
        raise ValueError(
            "Centre-level PTAL output does not reconcile "
            f"to the centre universe: {len(centre_ptal)} != {len(centres)}."
        )

    return centre_ptal


def build_summary(
    centres,
    ptal,
    reconstruction,
    centre_ptal,
):
    """Record source provenance, coverage and centre-level accessibility."""

    coverage = centre_ptal[
        "ptal_coverage_share"
    ]
    mean_ai = centre_ptal[
        "ptal_mean_ai"
    ]

    partial = centre_ptal[
        coverage < 0.999999
    ][
        [
            "boundary_id",
            "centre_name",
            "centre_borough",
            "tier",
            "ptal_coverage_share",
        ]
    ].copy()

    return {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "inputs": {
            "centre_count": len(centres),
            "ptal_grid_source_row_count": len(ptal),
            "ptal_source_file": str(
                PTAL_INPUT_PATH.relative_to(PROJECT_ROOT)
            ),
            "ptal_sheet": PTAL_SHEET_NAME,
            "crs": TARGET_CRS,
        },
        "grid_reconstruction": reconstruction,
        "coverage": {
            "centres_with_positive_coverage": int(
                coverage.gt(0).sum()
            ),
            "centres_with_full_coverage": int(
                coverage.ge(0.999999).sum()
            ),
            "centres_with_partial_coverage": int(
                coverage.lt(0.999999).sum()
            ),
            "min_ptal_coverage_share": float(
                coverage.min()
            ),
            "partial_coverage_centres": partial.to_dict(
                orient="records"
            ),
        },
        "accessibility": {
            "mean_ptal_mean_ai": float(
                mean_ai.mean()
            ),
            "median_ptal_mean_ai": float(
                mean_ai.median()
            ),
            "min_ptal_mean_ai": float(
                mean_ai.min()
            ),
            "max_ptal_mean_ai": float(
                mean_ai.max()
            ),
            "ptal_mean_category_counts": {
                str(key): int(value)
                for key, value in (
                    centre_ptal[
                        "ptal_mean_category"
                    ]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
        },
    }


def main():
    """Build and save centre-level PTAL accessibility."""

    ensure_outputs_do_not_exist()

    centre_summary = read_json(
        CENTRE_SUMMARY_PATH
    )

    centres = read_centres(
        centre_summary
    )

    ptal = read_ptal_grid()

    cells, duplicate_cells, reconstruction = (
        reconstruct_grid_cells(ptal)
    )

    check_duplicate_cells(
        centres,
        duplicate_cells,
    )

    centre_ptal = build_centre_ptal(
        centres,
        cells,
    )

    summary = build_summary(
        centres,
        ptal,
        reconstruction,
        centre_ptal,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    centre_ptal.to_csv(
        CENTRE_PTAL_OUTPUT_PATH,
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
    print(f"TfL PTAL source rows: {len(ptal)}")
    print(
        "Reconstructed grid cells: "
        f"{reconstruction['unique_reconstructed_cells']}"
    )
    print(
        "Duplicate reconstructed cell groups: "
        f"{reconstruction['duplicate_reconstructed_cell_groups']}"
    )
    print(
        "Centres with full PTAL coverage: "
        f"{summary['coverage']['centres_with_full_coverage']}"
    )
    print(
        "Centres with partial PTAL coverage: "
        f"{summary['coverage']['centres_with_partial_coverage']}"
    )
    print(
        "Minimum PTAL coverage share: "
        f"{summary['coverage']['min_ptal_coverage_share']:.6f}"
    )
    print(
        "Mean centre Access Index: "
        f"{summary['accessibility']['mean_ptal_mean_ai']:.4f}"
    )
    print(
        "Median centre Access Index: "
        f"{summary['accessibility']['median_ptal_mean_ai']:.4f}"
    )
    print(f"PTAL output: {CENTRE_PTAL_OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
