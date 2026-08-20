"""
Construct Article 4 spatial exposure for every analytical centre and observed
study period.

For each observed period:
-  select relevant Article 4 areas active at the midpoint of the actually
   observed part of that quarter
-  intersect active Article 4 polygons with centre boundaries
-  dissolve overlapping Article 4 intersections within each centre to avoid
   double-counting
-  calculate article4_share = covered centre area / total centre area;
-  derive 10%, 25% and 50% binary exposure thresholds.

"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CENTRE_INPUT_PATH = (
    PROJECT_ROOT / "data_processed" / "centres" / "centres.gpkg"
)
CENTRE_SUMMARY_PATH = (
    PROJECT_ROOT / "data_processed" / "centres" / "centre_summary.json"
)
CENTRE_LAYER_NAME = "centres"

ARTICLE4_INPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "article4"
    / "article4_relevant_areas_final.gpkg"
)
ARTICLE4_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "article4"
    / "article4_relevance_final_summary.json"
)
ARTICLE4_LAYER_NAME = "article4_relevant_areas_final"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "article4"

EXPOSURE_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "article4_centre_period.csv"
)

MISSING_DATE_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "article4_missing_start_date_intersecting_centres.csv"
)
UNUSABLE_GEOMETRY_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "article4_unusable_geometry.csv"
)
SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "article4_centre_period_summary.json"
)

TARGET_CRS = "EPSG:27700"

STUDY_START = pd.Timestamp("2021-08-01")
STUDY_END = pd.Timestamp("2024-03-04")

REQUIRED_CENTRE_COLUMNS = [
    "boundary_id",
    "centre_name",
    "centre_borough",
    "tier",
    "area_m2",
    "geometry",
]

REQUIRED_ARTICLE4_COLUMNS = [
    "entity",
    "effective_start_date",
    "effective_end_date",
    "geometry",
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
    """Prevent the Article 4 exposure snapshot from being silently replaced."""

    existing = []

    for path in [
        EXPOSURE_OUTPUT_PATH,
        MISSING_DATE_OUTPUT_PATH,
        UNUSABLE_GEOMETRY_OUTPUT_PATH,
        SUMMARY_OUTPUT_PATH,
    ]:
        if path.exists():
            existing.append(str(path))

    if existing:
        raise FileExistsError(
            "Article 4 exposure outputs already exist. "
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


def read_article4(article4_summary):
    """Read the final reviewed Article 4 area universe."""

    article4 = gpd.read_file(
        ARTICLE4_INPUT_PATH,
        layer=ARTICLE4_LAYER_NAME,
    )

    require_columns(
        article4,
        REQUIRED_ARTICLE4_COLUMNS,
        "Final Article 4 GeoPackage",
    )

    expected_count = int(
        article4_summary["final_relevant_area_rows"]
    )

    if len(article4) != expected_count:
        raise ValueError(
            "Article 4 input does not match its final summary: "
            f"{len(article4)} != {expected_count}."
        )

    if article4.crs is None:
        raise ValueError("Final Article 4 GeoPackage has no CRS.")

    article4 = article4.to_crs(TARGET_CRS)

    return article4


def prepare_article4_geometry(article4):
    """
    Separate unusable non-polygon records and repair invalid polygons.
    Non-polygon Article 4 records are retained for QA but are not converted or
    buffered into artificial areal exposure.
    """

    article4 = article4.copy()

    polygon_mask = article4.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )

    unusable = article4[
        ~polygon_mask
    ].copy()

    usable = article4[
        polygon_mask
    ].copy()

    invalid = ~usable.geometry.is_valid
    invalid_count = int(invalid.sum())

    if invalid.any():
        usable.loc[
            invalid,
            "geometry",
        ] = usable.loc[
            invalid,
            "geometry",
        ].apply(make_valid)

    remaining_invalid = ~usable.geometry.is_valid

    if remaining_invalid.any():
        raise ValueError(
            "An Article 4 polygon remains invalid after make_valid()."
        )

    # make_valid() can produce a valid non-polygon geometry.
    # Article 4 exposure is area-based, so repaired records must remain polygon.
    repaired_non_polygon = ~usable.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )

    if repaired_non_polygon.any():
        affected = usable.loc[
            repaired_non_polygon,
            ["entity", "geometry"],
        ].copy()

        affected["geom_type"] = (
            affected.geometry.geom_type
        )

        raise ValueError(
            "make_valid() produced non-polygon Article 4 geometry:\n"
            + affected[
                ["entity", "geom_type"]
            ].to_string(index=False)
        )
    return usable, unusable, invalid_count


def build_period_table():
    """Create the 11 observed quarter labels and their observed midpoints."""

    rows = []

    periods = pd.period_range(
        STUDY_START,
        STUDY_END,
        freq="Q",
    )

    for period in periods:
        observed_start = max(
            period.start_time.normalize(),
            STUDY_START,
        )
        observed_end = min(
            period.end_time.normalize(),
            STUDY_END,
        )
        observed_midpoint = (
            observed_start
            + (observed_end - observed_start) / 2
        )

        rows.append(
            {
                "period": str(period),
                "observed_start": observed_start,
                "observed_midpoint": observed_midpoint,
                "observed_end": observed_end,
            }
        )

    return pd.DataFrame(rows)


def identify_missing_start_date_overlap(article4, centres):
    """
    Identify undated relevant Article 4 areas that spatially intersect centres.
    These records remain excluded from temporal activation because their start dates are unknown.
    """

    missing = article4[
        article4["effective_start_date"].isna()
    ].copy()

    if missing.empty:
        return missing

    centres_union = centres.geometry.union_all()

    missing["intersects_centres"] = (
        missing.geometry.intersects(centres_union)
    )

    return missing[
        missing["intersects_centres"]
    ].copy()


def construct_exposure(article4, centres, periods):
    """Construct the full centre-period Article 4 exposure table."""

    article4 = article4.copy()

    dated = article4[
        article4["effective_start_date"].notna()
    ].copy()

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

    centres_small = centres_small.rename(
        columns={
            "area_m2": "centre_area_m2",
        }
    )

    exposure_rows = []
    period_summaries = []

    for _, period_row in periods.iterrows():
        period = period_row["period"]
        midpoint = period_row["observed_midpoint"]

        active = dated[
            (dated["effective_start_date"] <= midpoint)
            & (
                dated["effective_end_date"].isna()
                | (dated["effective_end_date"] >= midpoint)
            )
        ].copy()

        if active.empty:
            period_exposure = centres_small.drop(
                columns="geometry"
            ).copy()

            period_exposure["period"] = period
            period_exposure["article4_area_m2"] = 0.0
            period_exposure["article4_share"] = 0.0
            period_exposure["active_article4_area_count"] = 0

        else:
            intersections = gpd.overlay(
                centres_small,
                active[
                    [
                        "entity",
                        "effective_start_date",
                        "effective_end_date",
                        "geometry",
                    ]
                ],
                how="intersection",
                keep_geom_type=False,
            )

            intersections["intersection_area_m2"] = (
                intersections.geometry.area
            )

            intersections = intersections[
                intersections["intersection_area_m2"] > 0
            ].copy()

            if intersections.empty:
                period_exposure = centres_small.drop(
                    columns="geometry"
                ).copy()

                period_exposure["period"] = period
                period_exposure["article4_area_m2"] = 0.0
                period_exposure["article4_share"] = 0.0
                period_exposure["active_article4_area_count"] = 0

            else:
                dissolved = intersections[
                    ["boundary_id", "geometry"]
                ].dissolve(
                    by="boundary_id"
                )

                dissolved["article4_area_m2"] = (
                    dissolved.geometry.area
                )

                dissolved = dissolved.reset_index()

                area_counts = (
                    intersections.groupby("boundary_id")
                    .agg(
                        active_article4_area_count=(
                            "entity",
                            "nunique",
                        )
                    )
                    .reset_index()
                )

                period_exposure = centres_small.drop(
                    columns="geometry"
                ).merge(
                    dissolved[
                        [
                            "boundary_id",
                            "article4_area_m2",
                        ]
                    ],
                    on="boundary_id",
                    how="left",
                )

                period_exposure = period_exposure.merge(
                    area_counts,
                    on="boundary_id",
                    how="left",
                )

                period_exposure["article4_area_m2"] = (
                    period_exposure["article4_area_m2"]
                    .fillna(0.0)
                )

                period_exposure[
                    "active_article4_area_count"
                ] = (
                    period_exposure[
                        "active_article4_area_count"
                    ]
                    .fillna(0)
                    .astype(int)
                )

                period_exposure["article4_share"] = (
                    period_exposure["article4_area_m2"]
                    / period_exposure["centre_area_m2"]
                )

                if (
                    period_exposure["article4_share"] > 1.000001
                ).any():
                    raise ValueError(
                        "Article 4 covered area exceeds centre area."
                    )

                period_exposure["article4_share"] = (
                    period_exposure["article4_share"]
                    .clip(lower=0, upper=1)
                )

                period_exposure["period"] = period

        period_exposure["has_article4"] = (
            period_exposure["article4_share"] > 0
        ).astype(int)

        period_exposure["treated_10"] = (
            period_exposure["article4_share"] >= 0.10
        ).astype(int)

        period_exposure["treated_25"] = (
            period_exposure["article4_share"] >= 0.25
        ).astype(int)

        period_exposure["treated_50"] = (
            period_exposure["article4_share"] >= 0.50
        ).astype(int)

        exposure_rows.append(period_exposure)

        period_summaries.append(
            {
                "period": period,
                "observed_start": str(
                    period_row["observed_start"].date()
                ),
                "observed_midpoint": str(
                    period_row["observed_midpoint"]
                ),
                "observed_end": str(
                    period_row["observed_end"].date()
                ),
                "active_article4_areas": len(active),
                "centres_with_any_article4": int(
                    period_exposure["has_article4"].sum()
                ),
                "centres_treated_10": int(
                    period_exposure["treated_10"].sum()
                ),
                "centres_treated_25": int(
                    period_exposure["treated_25"].sum()
                ),
                "centres_treated_50": int(
                    period_exposure["treated_50"].sum()
                ),
            }
        )

    exposure = pd.concat(
        exposure_rows,
        ignore_index=True,
    )

    expected_rows = len(centres) * len(periods)

    if len(exposure) != expected_rows:
        raise ValueError(
            "Centre-period exposure table does not reconcile: "
            f"{len(exposure)} != {len(centres)} * {len(periods)}."
        )

    return exposure, period_summaries

def build_summary(
    centres,
    article4,
    final_relevant_area_count,
    periods,
    exposure,
    period_summaries,
    repaired_geometry_count,
    unusable_geometry_count,
    missing_start_intersecting_count,
):
    """Record the final Article 4 centre-period exposure counts."""

    return {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "inputs": {
            "centre_count": len(centres),
            "final_relevant_article4_area_count": final_relevant_area_count,
            "spatially_usable_article4_polygon_count": len(article4),
            "observed_period_count": len(periods),
            "study_start": str(STUDY_START.date()),
            "study_end": str(STUDY_END.date()),
        },
        "article4_geometry": {
            "invalid_geometries_repaired": repaired_geometry_count,
            "non_polygon_records_withheld": unusable_geometry_count,
        },
        "article4_dates": {
            "missing_effective_start_date": int(
                article4["effective_start_date"].isna().sum()
            ),
            "missing_start_date_intersecting_centres": (
                missing_start_intersecting_count
            ),
        },
        "exposure": {
            "centre_period_rows": len(exposure),
            "centre_periods_with_any_article4": int(
                exposure["has_article4"].sum()
            ),
            "centre_periods_treated_10": int(
                exposure["treated_10"].sum()
            ),
            "centre_periods_treated_25": int(
                exposure["treated_25"].sum()
            ),
            "centre_periods_treated_50": int(
                exposure["treated_50"].sum()
            ),
            "mean_article4_share": float(
                exposure["article4_share"].mean()
            ),
            "max_article4_share": float(
                exposure["article4_share"].max()
            ),
        },
        "periods": period_summaries,
    }


def main():
    """Build and save the Article 4 centre-period exposure dataset."""

    ensure_outputs_do_not_exist()

    centre_summary = read_json(
        CENTRE_SUMMARY_PATH
    )
    article4_summary = read_json(
        ARTICLE4_SUMMARY_PATH
    )

    centres = read_centres(
        centre_summary
    )
    article4_all = read_article4(
        article4_summary
    )

    final_relevant_area_count = len(article4_all)

    article4, unusable_geometry, repaired_geometry_count = (
        prepare_article4_geometry(article4_all)
    )

    article4["effective_start_date"] = pd.to_datetime(
        article4["effective_start_date"],
        errors="coerce",
    )
    article4["effective_end_date"] = pd.to_datetime(
        article4["effective_end_date"],
        errors="coerce",
    )

    periods = build_period_table()

    missing_start_intersecting = (
        identify_missing_start_date_overlap(
            article4,
            centres,
        )
    )

    exposure, period_summaries = (
        construct_exposure(
            article4,
            centres,
            periods,
        )
    )

    

    summary = build_summary(
        centres,
        article4,
        final_relevant_area_count,
        periods,
        exposure,
        period_summaries,
        repaired_geometry_count,
        len(unusable_geometry),
        len(missing_start_intersecting),
    )

    

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    exposure.to_csv(
        EXPOSURE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    missing_start_intersecting.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        MISSING_DATE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    unusable_geometry.drop(
        columns="geometry",
        errors="ignore",
    ).to_csv(
        UNUSABLE_GEOMETRY_OUTPUT_PATH,
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
    print(f"Spatially usable Article 4 polygons: {len(article4)}")
    print(
        "Non-polygon Article 4 records withheld: "
        f"{len(unusable_geometry)}"
    )
    print(
        "Missing start date intersecting centres: "
        f"{len(missing_start_intersecting)}"
    )
    print(
        "Centre-period rows: "
        f"{len(exposure)}"
    )
    print(
        "Centre-periods treated_25: "
        f"{int(exposure['treated_25'].sum())}"
    )
    print(f"Exposure output: {EXPOSURE_OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_OUTPUT_PATH}")

if __name__ == "__main__":
    main()

