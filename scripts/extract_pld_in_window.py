"""
Extract the study-period PLD candidate universe from the frozen snapshot.

"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_JSONL = PROJECT_ROOT / "data_raw" / "pld" / "pld_full_query_hits_2026-08-09.jsonl"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "conversions"

IN_WINDOW_OUTPUT_PATH = OUTPUT_DIRECTORY / "pld_in_window.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIRECTORY / "pld_extraction_summary.json"

"""
Study Period
"""
START_DATE = pd.Timestamp("2021-08-01")
END_DATE = pd.Timestamp("2024-03-04")

"""
PLD fields
"""
SOURCE_FIELDS = [
    "id",
    "lpa_name",
    "borough",
    "lpa_app_no",
    "application_type",
    "application_type_full",
    "development_type",
    "description",
    "application_details",
    "decision",
    "decision_date",
    "valid_date",
    "centroid_easting",
    "centroid_northing",
    "postcode",
    "site_name",
    "street_name",
    "url_planning_app",
]

"""
Candidate classification rules
These rules are retained only as diagnostic information
"""

HIGH_PRECISION_PATTERNS = [
    r"class\s*ma",
    r"class\s*e\s*(to|2)\s*c3",
    r"class\s*e\s*to\s*class\s*c3",
    r"commercial\s*to\s*residential",
    r"office\s*to\s*residential",
    r"shop\s*to\s*residential",
    r"retail\s*to\s*residential",
    r"commercial\s*unit\s*.*residential",
    r"change\s*of\s*use\s*.*class\s*e\s*.*c3",
    r"change\s*of\s*use\s*.*commercial\s*.*residential",
]

CONTEXT_PATTERNS = [
    r"prior\s*approval",
    r"change\s*of\s*use",
    r"dwellinghouse",
    r"dwellinghouses",
    r"residential",
    r"use\s*class\s*c3",
    r"class\s*c3",
]

EXCLUSION_PATTERNS = [
    r"\bhmo\b",
    r"house\s*in\s*multiple\s*occupation",
    r"student\s*accommodation",
    r"\bhotel\b",
    r"\bhostel\b",
    r"care\s*home",
    r"\bc2\b",
    r"telecom",
    r"telecommunications",
]

REVIEW_PATTERNS = [
    r"roof\s*extension",
    r"single\s*storey",
    r"householder",
    r"demolition",
    r"redevelopment",
    r"extension",
]

#RETAINED_LABELS = [
#    "include_candidate",
#    "review_candidate",
#    "review_exclusion",
#]

ALL_LABELS = [
    "include_candidate",
    "review_candidate",
    "review_exclusion",
    "drop_low_confidence"
]


def metadata_path_for_snapshot(input_path):
    """
    Derive the metadata file name with the raw snapshot
    """
    prefix = "pld_full_query_hits_"

    if not input_path.name.startswith(prefix):
        raise ValueError(
            "Input filename does not match the frozen PLD shapshot names."
        )

    data_part = input_path.name[len(prefix):].removesuffix(".jsonl")

    return input_path.with_name(f"pld_full_query_metadata_{data_part}.json")

def read_metadata(metadata_path):
    """
    Read and validate the retrieval metadata.
    """

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"PLD retrieval metadata not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

    required_fields = [
         "api_reported_total_hits",
         "actually_retrieved_hits",
         "exact_reconciliation_passed",
    ]

    for field in required_fields:
        if field not in metadata:
            raise ValueError(
                f"PLD metadata file is missing required field: {field}"
            )
    if metadata["exact_reconciliation_passed"] is not True:
        raise ValueError(
            "PLD metadata indicates that the retrieval was incomplete."
        )
    
    if metadata["api_reported_total_hits"] != metadata["actually_retrieved_hits"]:
        raise ValueError(
            "PLD metadata totals do not reconcile."
        )

    return metadata

def read_raw_hits(input_path):
    """
    Read and flatten the preserved Elasticsearch hits.
    """

    if not input_path.is_file():
        raise FileNotFoundError(f"Frozen PLD JSONL file not found: {input_path}")

    records = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank line found at JSONL line {line_number}."
                )

            try:
                hit = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at PLD JSONL line {line_number}:{error}"
                    ) from error
            

            if not isinstance(hit, dict):
                raise ValueError(
                    f"PLD JSONL line {line_number} is nota JSON object."
                )
            
            source = hit.get("_source")

            if not isinstance(source, dict):
                raise ValueError(
                    f"PLD JSONL line {line_number} has no usable _source."
                )

            application_details = source.get("application_details")

            if  isinstance(application_details,(dict, list)):
                application_details = json.dumps(
                    application_details, 
                    ensure_ascii=False
                )

            record = {
                field: source.get(field) 
                for field in SOURCE_FIELDS
            }

            record["application_details"] = application_details
            record["es_id"] = hit.get("es_id") or hit.get("_id")
            
            records.append(record)

    if not records:
        raise ValueError("The frozen PLD snapshot contains no records.")

    return pd.DataFrame(records)


def raw_value_is_missing(value):
    """
    treat null, empty and empty values as missing.
    """

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
        
    except (TypeError, ValueError):
        return False

    return str(value).strip() == ""


def normalise_text(value):
    if raw_value_is_missing(value):
        return ""

    return (
        str(value)
        .replace("&nbsp;", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .lower()
    )


def score_record(row):
    """
    Create the old candidate score for diagnostics only.
    No label produced here is allowed to remove an in-window record.
    """
    text_parts = [
        row.get("description", ""),
        row.get("application_type", ""),
        row.get("application_type_full", ""),
        row.get("development_type", ""),
        row.get("application_details", ""),
    ]

    text = normalise_text(
        " ".join(
                str(value)
                for value in text_parts
                if not raw_value_is_missing(value)
            )
        )

    high_matches = [] # such as "class ma"
    context_matches= [] # such as "prior approval"
    exclusion_matches = [] # such as "hotel"
    review_matches = [] # such as "roof extension"

    for pattern in HIGH_PRECISION_PATTERNS:
        if re.search(pattern, text):
            high_matches.append(pattern)

    for pattern in CONTEXT_PATTERNS:
        if re.search(pattern, text):
            context_matches.append(pattern)

    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, text):
            exclusion_matches.append(pattern)

    for pattern in REVIEW_PATTERNS:
        if re.search(pattern, text):
            review_matches.append(pattern)

    keyword_score = (
        len(high_matches) * 10
        + len(context_matches) * 2
        - len(exclusion_matches) * 8
    )

    if (len(high_matches) >= 1
        and len(exclusion_matches) == 0
        and len(review_matches) == 0
    ):
        candidate_label = "include_candidate"

    elif (len(high_matches) >= 1 and len(exclusion_matches) > 0):
        candidate_label = "review_exclusion"

    elif keyword_score >= 8:
        candidate_label = "review_candidate"

    else:
        candidate_label = "drop_low_confidence"

    return pd.Series(
        {
            "keyword_score": keyword_score,
            "high_precision_matches": "; ".join(high_matches),
            "context_matches": ";".join(context_matches),
            "exclusion_matches": ";".join(exclusion_matches),
            "review_matches": ";".join(review_matches),
            "candidate_label" : candidate_label,
        }
    )

def add_date_fields(data):
    """
    Preserve raw dates and create separate parsed fields.
    """
    data["valid_date_parsed"] = pd.to_datetime(
        data["valid_date"],
        dayfirst=True,
        errors="coerce",
    )
    data["decision_date_parsed"] = pd.to_datetime(
        data["decision_date"],
        dayfirst=True,
        errors="coerce",
    )

    data["quarter"] = (
        data["valid_date_parsed"]
        .dt.to_period("Q")
        .astype(str)
    )

    data["in_main_window"] =(
        (data["valid_date_parsed"] >= START_DATE)
        & (data["valid_date_parsed"] <= END_DATE)
    )

    return data

def add_coordinate_quality(data):
    """
    Flag whether each PLD centroid is spatially usable in London boundary
    """
    data["centroid_easting"] = pd.to_numeric(
        data["centroid_easting"],
        errors="coerce",
    )
    data["centroid_northing"] = pd.to_numeric(
        data["centroid_northing"],
        errors="coerce",
    )

    # Coordinates are only flagged here because final spatial eligibility 
    # is a later analytical decision. 
    # Removing them now would shrink the
    # candidate universe before that decision can be documented and reviewed.
    data["has_valid_centroid"] = (
        data["centroid_easting"].notna()
        & data["centroid_northing"].notna()
        & data["centroid_easting"].between(
            500000, 
            570000, 
            inclusive="both",
        )
        & data["centroid_northing"].between(
            150000,
            210000,
            inclusive="both",
        )
    )

    return data

def check_raw_integrity(data, metadata):
    """
    Reconcile the actual JSONL rows against the frozen metadata.
    """
    raw_rows = len(data)

    total_hits = metadata["api_reported_total_hits"]
    retrieved_total = metadata["actually_retrieved_hits"]

    if raw_rows != retrieved_total:
        raise ValueError(
            f"Raw row count ({raw_rows}) does not match retrieved total ({retrieved_total})."
        )

    if total_hits != retrieved_total:
        raise ValueError(
            f"API reported total ({total_hits}) does not match retrieved total ({retrieved_total})."
        )

    missing_es_id = int (
        data["es_id"]
        .apply(raw_value_is_missing)
        .sum()
    )

    if missing_es_id != 0:
        raise ValueError(
            f"Raw PLD snapshot contains {missing_es_id} missing es_id value(s)."
        )

    duplicate_es_id = int(data["es_id"].astype(str).duplicated().sum()
    )

    if duplicate_es_id != 0:
        raise ValueError(
            f"Raw PLD snapshot contains {duplicate_es_id} duplicate es_id value(s)."
        )
    
def make_summary(data,in_window_output, input_path, metadata_path, metadata):
    """
    Build aset of integrity counts and arithmetic checks.
    """
    raw_missing_date = data["valid_date"].apply(raw_value_is_missing)
    parseable_date = data["valid_date_parsed"].notna()
    nonmissing_unparseable_date = (~raw_missing_date) & (~parseable_date)
    before_window = parseable_date & (data["valid_date_parsed"] < START_DATE)
    within_window = parseable_date & data["in_main_window"]
    after_window = parseable_date & (data["valid_date_parsed"] > END_DATE)

    """
    Date partition reconciliation
    """
    date_partition_total = (
        int(raw_missing_date.sum())
        + int(nonmissing_unparseable_date.sum())
        + int(before_window.sum())
        + int(within_window.sum())
        + int(after_window.sum())
    )

    if date_partition_total != len(data):
        raise ValueError(
            "Date status counts do not reconcile to raw PLD rows."
        )

    """
    Candidate labels with study window

    """
    in_window_data = data.loc[within_window]

    label_counts = {}

    for label in ALL_LABELS:
        label_counts[label] = int(
            (in_window_data["candidate_label"] == label).sum()
        )

    if sum(label_counts.values()) != len(in_window_data):
        raise ValueError(
            "In-window candidate-label counts do not reconcile to study window records."
        )

    valid_centroid_count = int(
        in_window_output["has_valid_centroid"].sum()
    )
    invalid_centroid_count = (
        len(in_window_output) - valid_centroid_count
    )


    checks = {
        "raw_rows_reconciled_to_retrieval_metadata": (
            len(data) == metadata["actually_retrieved_hits"]
            == metadata["api_reported_total_hits"]
        ),
        "date_partition_reconciled": date_partition_total == len(data),
        "study_window_output_equals_within_window_count": (
            len(in_window_output) == int(within_window.sum())
        ),
        "study_window_labels_reconciled": (
            sum(label_counts.values()) == len(in_window_output)
        ),
        "study_window_es_id_non_null": bool(
            in_window_output["es_id"].notna().all()
        ),
        "study_window_es_id_unique": bool(
            in_window_output["es_id"].is_unique
        ),
        "centroid_partition_reconciled": (
            len(in_window_output)
            == valid_centroid_count + invalid_centroid_count
        ),
    }

    for check_name, passed in checks.items():
        if passed is not True:
            raise ValueError(
                f"PLD extraction reconciliation check failed: {check_name}"
            )

    return {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "input_jsonl": str(input_path),
        "input_metadata": str(metadata_path),
        "study_period": {
            "date_field": "valid_date",
            "start": START_DATE.strftime("%Y-%m-%d"),
            "end": END_DATE.strftime("%Y-%m-%d"),
            "both_boundaries_inclusive": True,
        },
        "raw_input": {
            "total_raw_hits": int(len(data)),
            "unique_es_id": int(data["es_id"].astype(str).nunique()),
        },
        "date_integrity": {
            "missing_null_or_empty_raw_valid_date": int(
                raw_missing_date.sum()
            ),
            "nonmissing_but_unparseable_valid_date": int(
                nonmissing_unparseable_date.sum()
            ),
            "parseable_before_study_window": int(before_window.sum()),
            "parseable_within_study_window": int(within_window.sum()),
            "parseable_after_study_window": int(after_window.sum()),
        },
        "diagnostic_candidate_labels_within_study_window": label_counts,
        "study_window_output": {
            "output_rows": int(len(in_window_output)),
            "records_with_valid_centroid": valid_centroid_count,
            "records_with_invalid_or_missing_centroid": invalid_centroid_count,
        },
        "methodological_note": (
            "candidate_label is diagnostic only; every record inside the "
            "study window proceeds to the substantive cleaner."
        ),
        "reconciliation_checks": checks,
    }


def main():

    input_path = INPUT_JSONL.resolve()

    """
    Match the frozen snapshot to its metadata
    """

    metadata_path = metadata_path_for_snapshot(input_path)

    metadata = read_metadata(metadata_path)

    """
    Read raw PLD records
    """
    data = read_raw_hits(input_path)

    #verify the raw snapshot before any candidate processing
    check_raw_integrity(data, metadata)

    """
    Candidate scoring
    """
    scored = data.apply(score_record, axis=1)
    data = pd.concat([data, scored], axis=1)

    """
    Date and coordinate variables
    """
    data = add_date_fields(data)
    data = add_coordinate_quality(data)

    """
    Every record inside the study window is reserved
    """
    in_window_output = data.loc[
            data["in_main_window"]
        ].copy()

    in_window_output = in_window_output.sort_values(
        ["valid_date_parsed", "es_id"],
        kind="mergesort"
    ).reset_index(drop=True)

    summary = make_summary(
        data, 
        in_window_output, 
        input_path,
        metadata_path,
        metadata
    )

    OUTPUT_DIRECTORY.mkdir(parents=True,exist_ok=True)

    #These outputs represent one frozen candidate extraction.
    # Refusing to overwrite them prevents accidental silent replacement.
    if IN_WINDOW_OUTPUT_PATH.exists():
        raise FileExistsError(
                f"Study_window already exists:{IN_WINDOW_OUTPUT_PATH}"
            )
    
    
    if SUMMARY_OUTPUT_PATH.exists():
            raise FileExistsError(
                f"Extraction summary already exists:{SUMMARY_OUTPUT_PATH}"
            )

    in_window_output.to_csv(
        IN_WINDOW_OUTPUT_PATH,
        index = False,
        encoding="utf-8-sig",
        mode="x",
    )
    
    with SUMMARY_OUTPUT_PATH.open(
            "x",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                summary,
                output_file,
                indent=2,
                ensure_ascii=False,
            )
            output_file.write("\n")
    
    
    print(f"Raw PLD hits: {len(data)}")
    print(
        "Study-window PLD records: "
        f"{summary['study_window_output']['output_rows']}"
    )
    print(
        "Diagnostic candidate labels: "
        f"{summary['diagnostic_candidate_labels_within_study_window']}"
    )
    print(
        "Valid-centroid study-window records: "
        f"{summary['study_window_output']['records_with_valid_centroid']}"
    )
    print(
        "Invalid/missing-centroid study-window records: "
        f"{summary['study_window_output']['records_with_invalid_or_missing_centroid']}"
    )
    print(f"Study-window CSV: {IN_WINDOW_OUTPUT_PATH}")
    print(f"Extraction summary: {SUMMARY_OUTPUT_PATH}")
    print("All PLD extraction reconciliation checks passed.")
    
    
if __name__ == "__main__":
    main()