"""
Enumerate every layer in the 35 active Local Plan GeoPackages.

Purpose
- prove the complete source-layer universe before manual centre-layer review;
- record simple metadata for every layer;
- provide keyword-based review hints without using keywords as a filter.

"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyogrio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_PLAN_DIRECTORY = PROJECT_ROOT / "data_raw" / "local_plan"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "centres"
INVENTORY_OUTPUT_PATH = OUTPUT_DIRECTORY / "centre_layer_inventory.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIRECTORY / "centre_layer_inventory_summary.json"

EXPECTED_GPKG_FILES = [
    f"planning_local_plan_data_{number:02d}.gpkg"
    for number in range(1, 36)
]


# These words are review hints only.
CENTRE_HINT_KEYWORDS = [
    "town",
    "centre",
    "center",
    "district",
    "major",
    "metropolitan",
    "international",
    "local",
    "neighbourhood",
    "neighborhood",
    "area centre",
    "shopping",
    "retail",
    "primary",
    "principal",
    "high street",
    "frontage",
]

UNRELATED_HINT_KEYWORDS = [
    "conservation",
    "flood",
    "open space",
    "green belt",
    "heritage",
    "industrial",
    "employment",
    "road",
    "transport",
    "airport",
    "opportunity area",
    "safeguard",
]


def normalise_layer_name(value):
    """Make layer-name matching transparent and deterministic."""

    text = str(value).lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def matched_keywords(layer_name, keywords):
    """Return all review-hint keywords present in a layer name."""

    normalised = normalise_layer_name(layer_name)
    matches = []

    for keyword in keywords:
        if keyword in normalised:
            matches.append(keyword)

    return matches


def review_hint(layer_name, geometry_type):
    """
    Produce a non-binding review hint.

    Manual review remains authoritative.
    """

    centre_matches = matched_keywords(
        layer_name,
        CENTRE_HINT_KEYWORDS,
    )
    unrelated_matches = matched_keywords(
        layer_name,
        UNRELATED_HINT_KEYWORDS,
    )

    geometry_text = str(geometry_type or "").lower()
    polygon_layer = "polygon" in geometry_text

    if not polygon_layer:
        return "non_polygon_review"

    if centre_matches and not unrelated_matches:
        return "centre_name_signal"

    if unrelated_matches and not centre_matches:
        return "unrelated_name_signal"

    if centre_matches and unrelated_matches:
        return "mixed_name_signal"

    return "manual_check_no_keyword_signal"


def check_expected_source_files():
    """Require exactly the active 35 Local Plan GeoPackages."""

    if not LOCAL_PLAN_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Local Plan directory not found: {LOCAL_PLAN_DIRECTORY}"
        )

    found_files = sorted(
        path.name
        for path in LOCAL_PLAN_DIRECTORY.glob("*.gpkg")
    )

    expected_files = sorted(EXPECTED_GPKG_FILES)

    if found_files != expected_files:
        missing = sorted(set(expected_files) - set(found_files))
        unexpected = sorted(set(found_files) - set(expected_files))

        raise ValueError(
            "Local Plan source folder does not contain exactly the expected "
            f"35 GeoPackages. Missing={missing}; unexpected={unexpected}."
        )

    return found_files


def ensure_outputs_do_not_exist():
    """Prevent silent replacement of a reviewed source inventory."""

    existing = []

    for path in [INVENTORY_OUTPUT_PATH, SUMMARY_OUTPUT_PATH]:
        if path.exists():
            existing.append(str(path))

    if existing:
        raise FileExistsError(
            "Centre-layer inventory output already exists. Review/move it "
            "before rerunning:\n- "
            + "\n- ".join(existing)
        )


def inventory_one_gpkg(source_file):
    """Enumerate every layer in one GeoPackage and read its metadata."""

    source_path = LOCAL_PLAN_DIRECTORY / source_file

    try:
        listed_layers = pyogrio.list_layers(source_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not enumerate layers in {source_file}: {error}"
        ) from error

    if len(listed_layers) == 0:
        raise ValueError(
            f"GeoPackage contains no layers: {source_file}"
        )

    rows = []

    for listed_layer in listed_layers:
        source_layer = str(listed_layer[0])

        try:
            info = pyogrio.read_info(
                source_path,
                layer=source_layer,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not read layer metadata for "
                f"{source_file} / {source_layer}: {error}"
            ) from error

        geometry_type = info.get("geometry_type")
        feature_count = info.get("features")
        crs = info.get("crs")

        if isinstance(feature_count, bool):
            raise ValueError(
                f"Invalid feature count for {source_file} / {source_layer}."
            )

        try:
            feature_count = int(feature_count)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Missing/invalid feature count for "
                f"{source_file} / {source_layer}: {feature_count}"
            ) from error

        centre_matches = matched_keywords(
            source_layer,
            CENTRE_HINT_KEYWORDS,
        )
        unrelated_matches = matched_keywords(
            source_layer,
            UNRELATED_HINT_KEYWORDS,
        )

        geometry_text = str(geometry_type or "")
        is_polygon_layer = "polygon" in geometry_text.lower()

        rows.append(
            {
                "source_file": source_file,
                "source_layer": source_layer,
                "geometry_type": geometry_text,
                "feature_count": feature_count,
                "crs": "" if crs is None else str(crs),
                "is_polygon_layer": bool(is_polygon_layer),
                "centre_keyword_matches": "; ".join(centre_matches),
                "unrelated_keyword_matches": "; ".join(unrelated_matches),
                "review_hint": review_hint(
                    source_layer,
                    geometry_type,
                ),
            }
        )

    return rows


def build_summary(found_files, inventory):
    """Summarise the complete machine-generated source-layer inventory."""

    layer_count_by_file = (
        inventory.groupby("source_file")
        .size()
        .sort_index()
        .to_dict()
    )

    layer_count_by_file = {
        str(key): int(value)
        for key, value in layer_count_by_file.items()
    }

    hint_counts = (
        inventory["review_hint"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    hint_counts = {
        str(key): int(value)
        for key, value in hint_counts.items()
    }

    polygon_count = int(inventory["is_polygon_layer"].sum())
    non_polygon_count = len(inventory) - polygon_count

    checks = {
        "exactly_35_source_files": len(found_files) == 35,
        "source_file_names_match_expected": (
            sorted(found_files) == sorted(EXPECTED_GPKG_FILES)
        ),
        "every_source_file_has_at_least_one_layer": (
            set(layer_count_by_file) == set(found_files)
            and all(count > 0 for count in layer_count_by_file.values())
        ),
        "source_file_layer_keys_unique": bool(
            ~inventory.duplicated(
                subset=["source_file", "source_layer"],
                keep=False,
            ).any()
        ),
        "all_feature_counts_non_negative": bool(
            (inventory["feature_count"] >= 0).all()
        ),
        "polygon_nonpolygon_partition_reconciled": (
            polygon_count + non_polygon_count == len(inventory)
        ),
    }

    for check_name, passed in checks.items():
        if passed is not True:
            raise ValueError(
                f"Centre-layer inventory reconciliation failed: {check_name}"
            )

    return {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "source_directory": str(LOCAL_PLAN_DIRECTORY),
        "source_files": {
            "expected_gpkg_count": 35,
            "found_gpkg_count": len(found_files),
        },
        "layers": {
            "total_layer_count": int(len(inventory)),
            "polygon_layer_count": polygon_count,
            "non_polygon_layer_count": non_polygon_count,
            "layer_count_by_file": layer_count_by_file,
            "review_hint_counts": hint_counts,
        },
        "methodological_note": (
            "Every layer in every active GeoPackage is inventoried. "
            "Keyword matches and review_hint are non-binding review aids only."
        ),
        "reconciliation_checks": checks,
    }


def main():
    """Enumerate all active Local Plan layers and freeze the inventory."""

    ensure_outputs_do_not_exist()

    found_files = check_expected_source_files()

    rows = []

    for source_file in found_files:
        rows.extend(
            inventory_one_gpkg(source_file)
        )

    inventory = pd.DataFrame(rows)

    if inventory.empty:
        raise ValueError(
            "No Local Plan layers were inventoried."
        )

    inventory = inventory.sort_values(
        ["source_file", "source_layer"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = build_summary(
        found_files,
        inventory,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        INVENTORY_OUTPUT_PATH,
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

    print(f"Active Local Plan GeoPackages: {len(found_files)}")
    print(f"All inventoried layers: {len(inventory)}")
    print(
        "Polygon layers: "
        f"{summary['layers']['polygon_layer_count']}"
    )
    print(
        "Non-polygon layers: "
        f"{summary['layers']['non_polygon_layer_count']}"
    )
    print(f"Inventory CSV: {INVENTORY_OUTPUT_PATH}")
    print(f"Inventory summary: {SUMMARY_OUTPUT_PATH}")
    print("All centre-layer inventory reconciliation checks passed.")


if __name__ == "__main__":
    main()
