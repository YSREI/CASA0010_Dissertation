"""
This script makes conservative automated decisions.
"""

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRECTORY = PROJECT_ROOT / "data_processed" / "conversions"
INPUT_IN_WINDOW_PATH = INPUT_DIRECTORY / "pld_in_window.csv"
EXTRACTION_SUMMARY_PATH = INPUT_DIRECTORY / "pld_extraction_summary.json"

ALL_DECISIONS_PATH = INPUT_DIRECTORY / "conversions_all_decisions.csv"
REVIEW_PATH = INPUT_DIRECTORY / "conversions_needs_review.csv"
CLEANING_SUMMARY_PATH = INPUT_DIRECTORY / "cleaning_summary.json"

OUTPUT_PATHS = [
    ALL_DECISIONS_PATH,
    REVIEW_PATH,
    CLEANING_SUMMARY_PATH,
]

REQUIRED_COLUMNS = [
    "es_id",
    "id",
    "lpa_name",
    "borough",
    "lpa_app_no",
    "description",
    "application_details",
    "application_type",
    "application_type_full",
    "development_type",
    "decision",
    "candidate_label",
    "has_valid_centroid",
    "valid_date",
    "valid_date_parsed",
    "quarter",
    "in_main_window",
]

ADDED_COLUMNS = [
    "automated_cleaning_decision",
    "automated_cleaning_reason",
    "review_flags",
    "secondary_record_flag",
    "reverse_direction_flag",
    "redevelopment_only_flag",
    "clear_conversion_flag",
]


# These expressions describe the origin and destination concepts used below.
# They are deliberately limited to recognisable wording rather than attempting
# to create a complete legal ontology of every possible planning use.
COMMERCIAL_ORIGIN = (
    r"(?:"
    r"use\s*class\s*e(?:\s*\([^)]+\))?"
    r"|class\s*e(?:\s*\([^)]+\))?"
    r"|commercial(?:\s+(?:unit|premises|space|floorspace|building|use))?"
    r"|office(?:s|\s+use|\s+accommodation)?"
    r"|shop(?:s|\s+unit|\s+premises)?"
    r"|retail(?:\s+unit|\s+use|\s+premises)?"
    r"|business(?:\s+and\s+service)?(?:\s+use)?"
    r"|service\s+use"
    r"|bank"
    r"|restaurant"
    r"|caf[eé]"
    r"|public\s+house"
    r"|pub"
    r"|drinking\s+establishment"
    r"|hot\s+food\s+takeaway"
    r"|takeaway"
    r"|betting\s+(?:office|shop)"
    r"|launderette"
    r"|cinema"
    r"|nightclub"
    r"|theatre"
    r"|hotel"
    r")"
)

C3_DESTINATION = (
    r"(?:"
    r"use\s*class\s*c3(?:\s*\([a-z]\))?"
    r"|class\s*c3(?:\s*\([a-z]\))?"
    r"|\bc3\b"
    r"|ordinary\s+residential\s+use"
    r"|residential\s+(?:unit|units|flat|flats|dwelling|dwellings)"
    r"|self[-\s]*contained\s+(?:flat|flats|unit|units)"
    r"|dwellinghouse(?:s)?"
    r"|dwelling(?:s)?"
    r"|flat(?:s)?"
    r")"
)

NONCOMMERCIAL_RESIDENTIAL_ORIGIN = (
    r"(?:"
    r"use\s*class\s*c4"
    r"|class\s*c4"
    r"|\bc4\b"
    r"|\bhmo\b"
    r"|house\s+in\s+multiple\s+occupation"
    r"|use\s*class\s*c3"
    r"|class\s*c3"
    r"|residential\s+(?:use|unit|units|flat|flats|dwelling|dwellings)"
    r"|dwellinghouse(?:s)?"
    r"|dwelling(?:s)?"
    r"|flat(?:s)?"
    r")"
)


# A positive result requires a directional relationship, not merely commercial
# and residential words appearing somewhere in the same record.
FORWARD_CONVERSION_PATTERNS = [
    rf"\bfrom\s+{COMMERCIAL_ORIGIN}.{{0,140}}\b(?:to|into)\s+{C3_DESTINATION}",
    rf"{COMMERCIAL_ORIGIN}.{{0,100}}\b(?:to|into)\s+{C3_DESTINATION}",
    rf"\b(?:change\s+of\s+use|conversion)\b.{{0,120}}"
    rf"{COMMERCIAL_ORIGIN}.{{0,140}}{C3_DESTINATION}",
]

REVERSE_DIRECTION_PATTERNS = [
    rf"\bfrom\s+{C3_DESTINATION}.{{0,140}}\b(?:to|into)\s+"
    rf"{COMMERCIAL_ORIGIN}",
    rf"{C3_DESTINATION}.{{0,100}}\b(?:to|into)\s+{COMMERCIAL_ORIGIN}",
    rf"\b(?:change\s+of\s+use|conversion)\b.{{0,120}}"
    rf"{C3_DESTINATION}.{{0,140}}{COMMERCIAL_ORIGIN}",
]

NONCOMMERCIAL_ORIGIN_TO_C3_PATTERNS = [
    rf"\bfrom\s+{NONCOMMERCIAL_RESIDENTIAL_ORIGIN}.{{0,140}}"
    rf"\b(?:to|into)\s+{C3_DESTINATION}",
    rf"\b(?:change\s+of\s+use|conversion)\b.{{0,120}}"
    rf"{NONCOMMERCIAL_RESIDENTIAL_ORIGIN}.{{0,140}}{C3_DESTINATION}",
]

CONVERSION_LANGUAGE_PATTERNS = [
    r"\bchange\s+of\s+use\b",
    r"\bconversion\b",
    r"\bconvert(?:ed|ing)?\b",
]


# Administrative applications are not independent substantive applications.
# These patterns are evaluated before any positive conversion wording because
# a secondary record often quotes the description of its parent permission.
SECONDARY_CONDITION_PATTERNS = [
    r"\bdischarge\s+of\s+conditions?\b",
    r"\bapproval\s+of\s+details\s+reserved\s+by\s+(?:a\s+)?conditions?\b",
    r"\bdetails\s+pursuant\s+to\s+conditions?\b",
    r"\bdetails\s+required\s+by\s+conditions?\b",
    r"\bsubmission\s+of\s+details\s+pursuant\s+to\s+conditions?\b",
    r"\bsubmission\s+of\s+details\b.{0,80}\bconditions?\b",
    r"\bpursuant\s+to\s+conditions?\s+\d+\b",
]

SECONDARY_CERTIFICATE_PATTERNS = [
    r"\bcertificate\s+of\s+lawfulness\b",
    r"\blawful\s+development\s+certificate\b",
    r"\bcertificate\b.{0,40}\blawful\s+development\b",
    r"\blawful\s+development\s*:\s*(?:proposed|existing)\s+use\b",
    r"\bcleud\b",
    r"\bclopud\b",
]

SECONDARY_AMENDMENT_PATTERNS = [
    r"\bnon[-\s]*material\s+amendment\b",
    r"\bminor\s+material\s+amendment\b",
    r"\bvariation\s+of\s+conditions?\b",
    r"\bremoval\s+of\s+conditions?\b",
    r"\bremoval\s*/\s*variation\s+of\s+(?:a\s+)?conditions?\b",
    r"\bvary\s+conditions?\b",
    r"\bsection\s*73\b",
    r"\bs73\b",
]

# Pre-application advice is not a substantive planning application.
# The description check is anchored at the beginning so that a normal
# application merely referring to earlier pre-application advice is not excluded.
SECONDARY_PRE_APPLICATION_PATTERNS = [
    r"^\s*pre[-\s]*application\b",
    r"^\s*pre[-\s]*app\b",
]


# These records reproduce another authority's application for consultation
# purposes and should not be counted as a second substantive application.
SECONDARY_CONSULTATION_PATTERNS = [
    r"^\s*consultation\s+from\s+(?:a\s+)?neighbou?ring\s+authority\b",
    r"^\s*consultation\s+by\s+london\s+borough\b",
    r"^\s*adjoining\s+authority\s+consultation\b",
]


# Discharge of Section 106 / planning obligations is a follow up administrative
# record referring to an earlier substantive planning permission.
SECONDARY_S106_PATTERNS = [
    r"^\s*discharge\s+of\s+developer'?s?\s+obligation\b",
    r"^\s*discharge\s+of\s+planning\s+obligation\b",
    r"^\s*discharge\b.{0,80}\b(?:s106|section\s*106)\b",
]


REDEVELOPMENT_NEW_BUILD_PATTERNS = [
    r"\bdemolition\s+of\s+all\s+existing\b.{0,180}"
    r"\b(?:erection|construction|redevelopment)\b",
    r"\bdemolish(?:ed|ing)?\b.{0,180}\b(?:erect|erection|construct|"
    r"construction|new[-\s]*build|replacement)\b",
    r"\bsite\s+clearance\b.{0,180}\b(?:erection|construction|"
    r"redevelopment|new[-\s]*build)\b",
    r"\bphased\s+mixed[-\s]*use\s+redevelopment\b",
]

NON_C3_DESTINATION_PATTERNS = {
    "destination_hmo": [
        rf"{COMMERCIAL_ORIGIN}.{{0,140}}\b(?:to|into|as)\b.{{0,40}}"
        r"(?:use\s*class\s*c4|class\s*c4|\bc4\b|\bhmo\b|"
        r"house\s+in\s+multiple\s+occupation)",
    ],
    "destination_student": [
        rf"{COMMERCIAL_ORIGIN}.{{0,140}}\b(?:to|into|as|for)\b.{{0,40}}"
        r"(?:purpose[-\s]*built\s+)?student\s+accommodation",
    ],
    "destination_hotel_hostel": [
        rf"{COMMERCIAL_ORIGIN}.{{0,140}}\b(?:to|into|as)\b.{{0,40}}"
        r"(?:hotel|hostel)",
    ],
    "destination_c2": [
        rf"{COMMERCIAL_ORIGIN}.{{0,140}}\b(?:to|into|as)\b.{{0,40}}"
        r"(?:use\s*class\s*c2|class\s*c2|\bc2\b|care\s+home)",
    ],
}


def value_is_missing(value):
    """Treat nulls and blank strings consistently before text matching."""

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False

    return str(value).strip() == ""


def normalise_text(value):
    """Apply simple, transparent normalisation without adding NLP methods."""

    if value_is_missing(value):
        return ""

    text = str(value)
    text = text.replace("&nbsp;", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.lower()

    # Collapsing repeated whitespace prevents formatting differences from
    # changing a substantive regex decision.
    return re.sub(r"\s+", " ", text).strip()


def join_text_values(values):
    """Join already selected fields without searching any extra columns."""

    parts = []

    for value in values:
        text = normalise_text(value)
        if text:
            parts.append(text)

    return " ".join(parts)


def has_match(text, patterns):
    """Return True when at least one listed expression matches the text."""

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def read_extraction_summary(summary_path):
    """Read the upstream count that study window input must reconcile against."""

    if not summary_path.is_file():
        raise FileNotFoundError(f"Upstream extraction summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    try:
        expected_rows = summary[
            "study_window_output"
        ]["output_rows"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Upstream extraction summary is missing study_window_output.output_rows."
        ) from error

    if not isinstance(expected_rows, int) or expected_rows < 1:
        raise ValueError(
            "Upstream extraction study_window_output.output_rows must be a positive integer."
        )

    return expected_rows


def read_and_validate_in_window_records(input_path, expected_rows):
    """Read upstream extraction records and fail before cleaning if integrity is lost."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Upstream extraction study window CSV not found: {input_path}")

    records = pd.read_csv(input_path, low_memory=False)

    missing_columns = []
    for column in REQUIRED_COLUMNS:
        if column not in records.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(
            "Study-window CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    conflicting_columns = []
    for column in ADDED_COLUMNS:
        if column in records.columns:
            conflicting_columns.append(column)

    if conflicting_columns:
        raise ValueError(
            "Study-window CSV already contains automated cleaning output columns: "
            + ", ".join(conflicting_columns)
        )

    if len(records) != expected_rows:
        raise ValueError(
            "Upstream extraction records count mismatch: "
            f"CSV contains {len(records)} rows but the upstream extraction summary "
            f"reports {expected_rows}."
        )

    missing_es_id = records["es_id"].isna()
    missing_es_id = missing_es_id | (
        records["es_id"].astype("string").str.strip() == ""
    )

    if missing_es_id.any():
        raise ValueError(
            f"Study-window CSV contains {int(missing_es_id.sum())} missing es_id values."
        )

    duplicate_es_id = records["es_id"].duplicated(keep=False)

    if duplicate_es_id.any():
        duplicate_count = int(duplicate_es_id.sum())
        raise ValueError(
            "Study-window CSV contains non-unique es_id values across "
            f"{duplicate_count} rows."
        )

    in_main_window = records["in_main_window"].apply(centroid_is_valid)

    if not in_main_window.all():
        outside_count = int((~in_main_window).sum())
        raise ValueError(
            "Study-window CSV contains "
            f"{outside_count} record(s) not marked in_main_window=True."
        )

    return records


def ensure_outputs_do_not_exist():
    """Protect reviewed outputs from being silently overwritten."""

    existing_outputs = []

    for output_path in OUTPUT_PATHS:
        if output_path.exists():
            existing_outputs.append(str(output_path))

    if existing_outputs:
        raise FileExistsError(
            "Study window input output files already exist. Move or remove them only after "
            "reviewing the previous run:\n- "
            + "\n- ".join(existing_outputs)
        )


def secondary_exclusion_reason(primary_text, application_type_full_text):
    """
    Identify clear administrative/secondary records.

    These rules have the highest exclusion priority because secondary records
    often repeat the description of the original commercial-to-C3 permission.
    """

    if has_match(primary_text, SECONDARY_CONDITION_PATTERNS):
        return "secondary_conditions"

    if has_match(primary_text, SECONDARY_CERTIFICATE_PATTERNS):
        return "secondary_certificate"

    if has_match(primary_text, SECONDARY_AMENDMENT_PATTERNS):
        return "secondary_amendment"

    if has_match(primary_text, SECONDARY_PRE_APPLICATION_PATTERNS):
        return "secondary_pre_application"

    if has_match(primary_text, SECONDARY_CONSULTATION_PATTERNS):
        return "secondary_authority_consultation"

    if has_match(primary_text, SECONDARY_S106_PATTERNS):
        return "secondary_s106_obligation"

    # A formal Listed Building Consent record is not counted independently
    # from the substantive planning application.
    if re.fullmatch(
        r"(?:application\s+for\s+)?listed\s+building\s+consent(?:\s+only)?",
        application_type_full_text.strip(" ."),
    ):
        return "secondary_lbc_only"

    return ""


def non_c3_destination_reason(substantive_text):
    """Return a reason only for a directional commercial-to-non-C3 proposal."""

    for reason, patterns in NON_C3_DESTINATION_PATTERNS.items():
        if has_match(substantive_text, patterns):
            return reason

    return ""


def make_review_flags(primary_text, supporting_text):
    """Record complexity indicators separately from the final decision."""

    flags = []

    if re.search(r"\bmixed[-\s]*use\b|\bmixed\s+scheme\b", primary_text):
        flags.append("mixed_scheme")

    if re.search(r"\bdemol(?:ition|ish|ished|ishing)\b|\bredevelopment\b", primary_text):
        flags.append("demolition_or_redevelopment")

    if re.search(r"\bextension\b|\bextensions\b|\bextend(?:ed|ing)?\b", primary_text):
        flags.append("extension")

    if re.search(r"\blisted\s+building\b", primary_text):
        flags.append("listed_building")

    if re.search(
        r"\bhmo\b|house\s+in\s+multiple\s+occupation|\bclass\s*c4\b",
        primary_text,
    ):
        flags.append("hmo_language")

    if re.search(
        r"student\s+accommodation|\bhotel\b|\bhostel\b|"
        r"\bclass\s*c2\b|care\s+home",
        primary_text,
    ):
        flags.append("student_hotel_c2_language")

    if re.search(r"\bsui\s+generis\b", primary_text):
        flags.append("sui_generis_origin")

    # Supporting application_details can reveal that a case needs attention,
    # but it cannot independently cause inclusion or exclusion.
    supporting_has_direction = has_match(
        supporting_text,
        FORWARD_CONVERSION_PATTERNS + REVERSE_DIRECTION_PATTERNS,
    )
    primary_has_direction = has_match(
        primary_text,
        FORWARD_CONVERSION_PATTERNS + REVERSE_DIRECTION_PATTERNS,
    )

    if supporting_has_direction and not primary_has_direction:
        flags.append("complex_or_conflicting_text")

    return flags


def result_series(
    decision,
    reason,
    review_flags,
    secondary_flag,
    reverse_flag,
    redevelopment_only_flag,
    clear_conversion_flag,
):
    """Return the automated cleaning decision columns."""

    return pd.Series(
        {
            "automated_cleaning_decision": decision,
            "automated_cleaning_reason": reason,
            "review_flags": "; ".join(sorted(set(review_flags))),
            "secondary_record_flag": bool(secondary_flag),
            "reverse_direction_flag": bool(reverse_flag),
            "redevelopment_only_flag": bool(redevelopment_only_flag),
            "clear_conversion_flag": bool(clear_conversion_flag),
        }
    )


def classify_record(row):
    """Apply the frozen substantive cleaning hierarchy."""

    description_text = normalise_text(row.get("description"))
    application_type_full_text = normalise_text(
        row.get("application_type_full")
    )

    type_text = join_text_values(
        [
            row.get("application_type"),
            row.get("application_type_full"),
            row.get("development_type"),
        ]
    )

    primary_text = join_text_values([description_text, type_text])
    supporting_text = normalise_text(row.get("application_details"))

    # The description is the best account of the substantive proposal. Formal
    # type fields are used as the evidence text only where no description was
    # supplied; a generic type can therefore never rescue a contradictory
    # substantive description.
    if description_text:
        substantive_text = description_text
    else:
        substantive_text = type_text

    review_flags = make_review_flags(primary_text, supporting_text)

    secondary_reason = secondary_exclusion_reason(
        primary_text,
        application_type_full_text,
    )

    # Some PLD records describe themselves as Listed Building Consent in the
    # description even when the formal application type is incomplete or missing.
    # These are not automatically excluded because some schemes may combine
    # planning permission and listed-building consent. 
    # Manual review needed.
    lbc_description_flag = bool(
        re.match(
            r"^\s*(?:application\s+for\s+)?listed\s+building\s+consent\b",
            description_text,
        )
    )

    forward_flag = has_match(
        substantive_text,
        FORWARD_CONVERSION_PATTERNS,
    )
    reverse_flag = has_match(
        substantive_text,
        REVERSE_DIRECTION_PATTERNS,
    )
    conversion_language_flag = has_match(
        substantive_text,
        CONVERSION_LANGUAGE_PATTERNS,
    )
    commercial_origin_flag = bool(
        re.search(COMMERCIAL_ORIGIN, substantive_text)
    )
    c3_destination_flag = bool(
        re.search(C3_DESTINATION, substantive_text)
    )
    noncommercial_origin_flag = has_match(
        substantive_text,
        NONCOMMERCIAL_ORIGIN_TO_C3_PATTERNS,
    )

    redevelopment_pattern_flag = has_match(
        substantive_text,
        REDEVELOPMENT_NEW_BUILD_PATTERNS,
    )
    redevelopment_only_flag = bool(
        redevelopment_pattern_flag
        and commercial_origin_flag
        and c3_destination_flag
        and not forward_flag
        and not conversion_language_flag
    )

    non_c3_reason = non_c3_destination_reason(substantive_text)

    # Rule 1: clear administrative records have the highest priority. Quoted
    # parent wording cannot convert one of these into a substantive record.
    if secondary_reason:
        return result_series(
            "exclude",
            secondary_reason,
            review_flags,
            True,
            reverse_flag,
            redevelopment_only_flag,
            False,
        )

    # If the description itself presents the record as Listed Building Consent
    # but the formal application type did not prove that it is LBC-only,
    # send it to human review rather than counting or excluding it automatically.
    if lbc_description_flag:
        review_flags.append("listed_building")
        review_flags.append("complex_or_conflicting_text")

        return result_series(
            "review",
            "possible_lbc_companion",
            review_flags,
            False,
            reverse_flag,
            redevelopment_only_flag,
            False,
        )

    # Rule 2: where both directions are explicit, the text is too complex for
    # a safe automated decision. A one-way C3-to-commercial proposal is clear.
    if reverse_flag and forward_flag:
        review_flags.append("complex_or_conflicting_text")
        return result_series(
            "review",
            "conflicting_conversion_directions",
            review_flags,
            False,
            True,
            redevelopment_only_flag,
            True,
        )

    if reverse_flag:
        return result_series(
            "exclude",
            "reverse_residential_to_commercial",
            review_flags,
            False,
            True,
            redevelopment_only_flag,
            False,
        )

    # Rule 3: demolition itself is not disqualifying. Exclusion requires clear
    # replacement development and no identifiable retained-space conversion.
    if redevelopment_only_flag:
        return result_series(
            "exclude",
            "redevelopment_new_build_only",
            review_flags,
            False,
            False,
            True,
            False,
        )

    # Rule 4: incidental HMO, student, hotel, hostel or C2 wording does not
    # exclude a separate, explicitly identified commercial-to-C3 component.
    if non_c3_reason and not forward_flag:
        return result_series(
            "exclude",
            non_c3_reason,
            review_flags,
            False,
            False,
            False,
            False,
        )

    if non_c3_reason and forward_flag:
        review_flags.append("mixed_scheme")

    # Rule 5: a clearly residential/HMO origin is outside the construct unless
    # a separate commercial-to-C3 component has also been identified.
    if noncommercial_origin_flag and not forward_flag:
        return result_series(
            "exclude",
            "origin_noncommercial",
            review_flags,
            False,
            False,
            False,
            False,
        )

    # Rule 6: generic residential wording is not enough unless it identifies a
    # recognisable C3, dwelling, flat or residential-unit destination.
    if not c3_destination_flag:
        review_flags.append("destination_unclear")
        return result_series(
            "review",
            "destination_unclear",
            review_flags,
            False,
            False,
            False,
            False,
        )

    # Rules 7 and 8: explicit direction permits inclusion even in a mixed
    # scheme, because the eligible commercial-to-C3 component is identifiable.
    if forward_flag:
        return result_series(
            "include",
            "clear_commercial_to_c3",
            review_flags,
            False,
            False,
            False,
            True,
        )

    # Rule 9: Sui Generis is neither automatically commercial nor automatically
    # ineligible. Without an explicit qualifying direction it remains review.
    if re.search(r"\bsui\s+generis\b", substantive_text) and not forward_flag:
        review_flags.append("sui_generis_origin")
        review_flags.append("complex_or_conflicting_text")
        return result_series(
            "review",
            "sui_generis_origin_unclear",
            review_flags,
            False,
            False,
            False,
            False,
        )

    if not commercial_origin_flag:
        review_flags.append("origin_unclear")
        return result_series(
            "review",
            "origin_unclear",
            review_flags,
            False,
            False,
            False,
            False,
        )
    

    # Rule 10: both concepts may be present without proving that existing
    # commercial floorspace is actually converted. Review is safer than using
    # co-occurring keywords as a false positive.
    if conversion_language_flag:
        review_flags.append("complex_or_conflicting_text")
        reason = "commercial_to_c3_direction_unclear"
    else:
        review_flags.append("complex_or_conflicting_text")
        reason = "conversion_component_unclear"

    return result_series(
        "review",
        reason,
        review_flags,
        False,
        False,
        False,
        False,
    )


def centroid_is_valid(value):
    """Interpret the upstream extraction centroid flag without using it for eligibility."""

    if value_is_missing(value):
        return False

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes"}


def count_review_flags(series):
    """Count semicolon-separated flags for the concise audit summary."""

    counts = {}

    for value in series:
        if value_is_missing(value):
            continue

        for flag in str(value).split(";"):
            flag = flag.strip()
            if flag:
                counts[flag] = counts.get(flag, 0) + 1

    return dict(sorted(counts.items()))


def reason_counts_by_decision(output):
    """Show why each include, exclude and review decision was reached."""

    counts = {}

    for decision in ["include", "exclude", "review"]:
        group = output[
            output["automated_cleaning_decision"] == decision
        ]
        reason_counts = group["automated_cleaning_reason"].value_counts()
        counts[decision] = {
            str(reason): int(count)
            for reason, count in reason_counts.sort_index().items()
        }

    return counts


def valid_centroid_counts(output):
    """Report spatial-data quality without changing cleaning eligibility."""

    counts = {}
    valid_centroid = output["has_valid_centroid"].apply(centroid_is_valid)

    for decision in ["include", "exclude", "review"]:
        group_mask = output["automated_cleaning_decision"] == decision
        counts[decision] = int((group_mask & valid_centroid).sum())

    return counts


def candidate_label_decision_counts(output):
    """Cross-tab diagnostic retrieval labels against substantive decisions."""

    counts = {}
    labels = output["candidate_label"].fillna("(missing)").astype(str)

    for label in sorted(labels.unique()):
        label_mask = labels == label
        counts[label] = {
            "automated_include": int(
                (
                    label_mask
                    & (output["automated_cleaning_decision"] == "include")
                ).sum()
            ),
            "automated_exclude": int(
                (
                    label_mask
                    & (output["automated_cleaning_decision"] == "exclude")
                ).sum()
            ),
            "automated_review": int(
                (
                    label_mask
                    & (output["automated_cleaning_decision"] == "review")
                ).sum()
            ),
        }

    return counts


def build_summary(output, expected_rows):
    """Build only the reconciliation information needed to audit automated cleaning."""

    decision_counts = output["automated_cleaning_decision"].value_counts()

    include_count = int(decision_counts.get("include", 0))
    exclude_count = int(decision_counts.get("exclude", 0))
    review_count = int(decision_counts.get("review", 0))
    partition_total = include_count + exclude_count + review_count
    label_decision_counts = candidate_label_decision_counts(output)
    label_decision_total = 0

    for label_counts in label_decision_counts.values():
        label_decision_total += sum(label_counts.values())

    return {
        "inputs": {
            "input_csv": str(INPUT_IN_WINDOW_PATH),
            "extraction_summary": str(EXTRACTION_SUMMARY_PATH),
            "expected_input_rows": expected_rows,
            "input_rows": int(len(output)),
            "unique_es_id": int(output["es_id"].nunique()),
        },
        "automated_decisions": {
            "include": include_count,
            "exclude": exclude_count,
            "review": review_count,
        },
        "candidate_label_automated_decision_crosstab": label_decision_counts,
        "decision_reason_counts": reason_counts_by_decision(output),
        "review_flag_counts": count_review_flags(output["review_flags"]),
        "valid_centroid_count_by_decision": valid_centroid_counts(output),
        "reconciliation_checks": {
            "input_rows_equal_extraction_summary": len(output) == expected_rows,
            "all_es_id_non_null": bool(output["es_id"].notna().all()),
            "all_es_id_unique": bool(output["es_id"].is_unique),
            "classified_rows_equal_input_rows": len(output) == expected_rows,
            "output_partitions_equal_input_rows": partition_total == len(output),
            "candidate_label_crosstab_equals_input_rows": (
                label_decision_total == len(output)
            ),
        },
    }


def main():
    """Read, classify, reconcile, and write the three specified output files."""

    ensure_outputs_do_not_exist()

    expected_rows = read_extraction_summary(EXTRACTION_SUMMARY_PATH)
    records = read_and_validate_in_window_records(
        INPUT_IN_WINDOW_PATH,
        expected_rows,
    )

    input_row_count = len(records)
    # candidate_label is retained for diagnostics only. 
    # Every in-window source row reaches the same substantive cleaning rules below.
    classified_columns = records.apply(classify_record, axis=1)

    if len(classified_columns) != input_row_count:
        raise ValueError(
            "Classification changed the number of study window rows: "
            f"before={input_row_count}, after={len(classified_columns)}."
        )

    output = pd.concat(
        [records.reset_index(drop=True), classified_columns.reset_index(drop=True)],
        axis=1,
    )

    if len(output) != input_row_count:
        raise ValueError(
            "study window input row multiplication detected after classification: "
            f"before={input_row_count}, after={len(output)}."
        )

    if output["es_id"].isna().any() or not output["es_id"].is_unique:
        raise ValueError(
            "es_id integrity changed during classification."
        )

    allowed_decisions = {"include", "exclude", "review"}
    observed_decisions = set(output["automated_cleaning_decision"].unique())

    if not observed_decisions.issubset(allowed_decisions):
        raise ValueError(
            "Unexpected study window input decision values: "
            f"observed={sorted(observed_decisions)}."
        )

    clean = output[
        output["automated_cleaning_decision"] == "include"
    ].copy()
    excluded = output[
        output["automated_cleaning_decision"] == "exclude"
    ].copy()
    review = output[
        output["automated_cleaning_decision"] == "review"
    ].copy()

    partition_total = len(clean) + len(excluded) + len(review)

    if partition_total != input_row_count:
        raise ValueError(
            "study window input partition mismatch: "
            f"include + exclude + review = {partition_total}, "
            f"but input rows = {input_row_count}."
        )

    summary = build_summary(output, expected_rows)

    for check_name, passed in summary["reconciliation_checks"].items():
        if passed is not True:
            raise ValueError(
                f"Study window input reconciliation check failed: {check_name}"
            )

    INPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        ALL_DECISIONS_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    review.to_csv(
        REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    with CLEANING_SUMMARY_PATH.open("x", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print(f"Study window input records: {input_row_count}")
    print(f"Automated include: {len(clean)}")
    print(f"Automated exclude: {len(excluded)}")
    print(f"Automated review: {len(review)}")
    print("All study window input reconciliation checks passed.")


if __name__ == "__main__":
    main()