"""
Merge manual PLD adjudications into the automated screening results.
Inputs:
- data_processed/conversions/conversions_all_decisions.csv
- data_processed/conversions/cleaning_summary.json
- manual_inputs/manual_review_pld.csv

Outputs:
- data_processed/conversions/pld_screening_final.csv
- data_processed/conversions/pld_clean.csv
- data_processed/conversions/merge_summary.json
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


csv.field_size_limit(50_000_000)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERSION_DIRECTORY = PROJECT_ROOT / "data_processed" / "conversions"

ALL_DECISIONS_PATH = CONVERSION_DIRECTORY / "conversions_all_decisions.csv"
CLEANING_SUMMARY_PATH = CONVERSION_DIRECTORY / "cleaning_summary.json"
MANUAL_REVIEW_PATH = PROJECT_ROOT / "manual_inputs" / "manual_review_pld.csv"

FINAL_MASTER_PATH = CONVERSION_DIRECTORY / "pld_screening_final.csv"
FINAL_CLEAN_PATH = CONVERSION_DIRECTORY / "pld_clean.csv"
MERGE_SUMMARY_PATH = CONVERSION_DIRECTORY / "merge_summary.json"

ALLOWED_MANUAL_DECISIONS = {
    "include",
    "exclude",
    "unresolved",
}


def read_csv(path):
    """Read one required CSV file."""

    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path):
    """Read one required JSON file."""

    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_text(value):
    """Return stripped text, treating missing CSV values as blank."""

    if value is None:
        return ""

    return str(value).strip()


def normalise_reference(value):
    """
    Conservatively normalise LPA/application-reference text for duplicate QA.
    Punctuation is preserved so distinct application numbers are not collapsed.
    """

    text = clean_text(value).upper()
    return re.sub(r"\s+", " ", text)


def ensure_outputs_do_not_exist():
    """Prevent silent replacement of the final reconciled outputs."""

    existing = []

    for path in [
        FINAL_MASTER_PATH,
        FINAL_CLEAN_PATH,
        MERGE_SUMMARY_PATH,
    ]:
        if path.exists():
            existing.append(str(path))

    if existing:
        raise FileExistsError(
            "Final screening outputs already exist. Move them before rerunning:\n- "
            + "\n- ".join(existing)
        )


def validate_automated_input(all_rows, cleaning_summary):
    """
    Reconcile conversions_all_decisions.csv with cleaning_summary.json.
    """

    try:
        expected_rows = int(
            cleaning_summary["inputs"]["input_rows"]
        )
        expected_counts = {
            "include": int(
                cleaning_summary["automated_decisions"]["include"]
            ),
            "exclude": int(
                cleaning_summary["automated_decisions"]["exclude"]
            ),
            "review": int(
                cleaning_summary["automated_decisions"]["review"]
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "cleaning_summary.json is missing valid input/decision counts."
        ) from error

    if len(all_rows) != expected_rows:
        raise ValueError(
            "Automated-cleaning row count mismatch: "
            f"CSV={len(all_rows)}, summary={expected_rows}."
        )

    es_ids = [
        clean_text(row.get("es_id"))
        for row in all_rows
    ]

    if any(not es_id for es_id in es_ids):
        raise ValueError(
            "Automated-cleaning input contains missing es_id values."
        )

    if len(es_ids) != len(set(es_ids)):
        raise ValueError(
            "Automated-cleaning input contains duplicate es_id values."
        )

    observed_counts = Counter(
        clean_text(
            row.get("automated_cleaning_decision")
        ).lower()
        for row in all_rows
    )

    if observed_counts != Counter(expected_counts):
        raise ValueError(
            "Automated decision counts do not match cleaning_summary.json: "
            f"observed={dict(observed_counts)}, "
            f"expected={expected_counts}."
        )

    return observed_counts


def validate_manual_review(manual_rows, review_ids):
    """
    Require exactly one valid manual decision for every automated-review es_id.
    """

    if not manual_rows:
        raise ValueError("Manual review file is empty.")

    required_columns = {"es_id", "final_decision"}
    missing_columns = required_columns - set(manual_rows[0].keys())

    if missing_columns:
        raise ValueError(
            "Manual review file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    manual_by_id = {}

    for row in manual_rows:
        es_id = clean_text(row.get("es_id"))
        decision = clean_text(
            row.get("final_decision")
        ).lower()

        if not es_id:
            raise ValueError(
                "Manual review file contains a missing es_id."
            )

        if es_id in manual_by_id:
            raise ValueError(
                f"Duplicate manual review es_id: {es_id}"
            )

        if decision not in ALLOWED_MANUAL_DECISIONS:
            raise ValueError(
                f"Invalid manual decision for {es_id}: {decision!r}"
            )

        manual_by_id[es_id] = row

    manual_ids = set(manual_by_id)

    if manual_ids != review_ids:
        raise ValueError(
            "Manual review IDs do not exactly match the current automated-review "
            f"population. Missing={len(review_ids - manual_ids)}, "
            f"extra={len(manual_ids - review_ids)}."
        )

    manual_counts = Counter(
        clean_text(
            row.get("final_decision")
        ).lower()
        for row in manual_rows
    )

    return manual_by_id, manual_counts


def merge_final_decisions(all_rows, manual_by_id):
    """
    Keep automated include/exclude decisions and replace only review decisions.
    """

    output_rows = []

    for row in all_rows:
        new_row = dict(row)

        es_id = clean_text(row.get("es_id"))
        automated_decision = clean_text(
            row.get("automated_cleaning_decision")
        ).lower()

        if automated_decision == "review":
            manual = manual_by_id[es_id]

            new_row["decision_source"] = "manual_review"
            new_row["final_decision"] = clean_text(
                manual.get("final_decision")
            ).lower()

            # Keep manual provenance where available.
            new_row["manual_note"] = clean_text(
                manual.get("manual_note")
            )
            new_row["manual_review_batch"] = clean_text(
                manual.get("review_batch")
            )
            new_row["manual_evidence_basis"] = clean_text(
                manual.get("evidence_basis")
            )
            new_row["manual_evidence_source_url"] = clean_text(
                manual.get("evidence_source_url")
            )

        else:
            new_row["decision_source"] = "automated"
            new_row["final_decision"] = automated_decision
            new_row["manual_note"] = ""
            new_row["manual_review_batch"] = ""
            new_row["manual_evidence_basis"] = ""
            new_row["manual_evidence_source_url"] = ""

        output_rows.append(new_row)

    input_ids = [
        clean_text(row.get("es_id"))
        for row in all_rows
    ]
    output_ids = [
        clean_text(row.get("es_id"))
        for row in output_rows
    ]

    if output_ids != input_ids:
        raise ValueError(
            "Merge changed the application row/ID sequence."
        )

    return output_rows


def check_final_counts(output_rows, automated_counts, manual_counts):
    """Prove that the final classification is exactly auto + manual decisions."""

    final_counts = Counter(
        clean_text(
            row.get("final_decision")
        ).lower()
        for row in output_rows
    )

    expected_counts = Counter({
        "include": (
            automated_counts["include"]
            + manual_counts.get("include", 0)
        ),
        "exclude": (
            automated_counts["exclude"]
            + manual_counts.get("exclude", 0)
        ),
        "unresolved": manual_counts.get(
            "unresolved",
            0,
        ),
    })

    if final_counts != expected_counts:
        raise ValueError(
            "Final classification does not reconcile: "
            f"observed={dict(final_counts)}, "
            f"expected={dict(expected_counts)}."
        )

    if sum(final_counts.values()) != len(output_rows):
        raise ValueError(
            "Final classification counts do not equal total input rows."
        )

    return final_counts


def check_application_reference_duplicates(clean_rows):
    """
    Check final eligible applications for repeated LPA/application references.

    Missing references are counted for transparency.
    Duplicate groups stop the script
    """

    reference_groups = defaultdict(list)
    missing_reference_count = 0

    for row in clean_rows:
        lpa_name = normalise_reference(
            row.get("lpa_name")
        )
        app_no = normalise_reference(
            row.get("lpa_app_no")
        )

        if not lpa_name or not app_no:
            missing_reference_count += 1
            continue

        reference_groups[
            (lpa_name, app_no)
        ].append(
            clean_text(row.get("es_id"))
        )

    duplicates = {
        f"{lpa} | {app_no}": es_ids
        for (lpa, app_no), es_ids in reference_groups.items()
        if len(es_ids) > 1
    }

    if duplicates:
        examples = list(duplicates.items())[:10]

        raise ValueError(
            "Final eligible sample contains duplicate "
            "(lpa_name, lpa_app_no) groups. "
            f"Groups={len(duplicates)}. Examples={examples}"
        )

    return missing_reference_count


def write_csv(path, rows):
    """Write one final CSV output."""

    if not rows:
        raise ValueError(
            f"Cannot write empty output: {path}"
        )

    headers = list(rows[0].keys())

    with path.open(
        "x",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Merge current automated and manual decisions and save final outputs."""

    ensure_outputs_do_not_exist()

    all_rows = read_csv(
        ALL_DECISIONS_PATH
    )
    cleaning_summary = read_json(
        CLEANING_SUMMARY_PATH
    )
    manual_rows = read_csv(
        MANUAL_REVIEW_PATH
    )

    automated_counts = validate_automated_input(
        all_rows,
        cleaning_summary,
    )

    review_ids = {
        clean_text(row.get("es_id"))
        for row in all_rows
        if clean_text(
            row.get("automated_cleaning_decision")
        ).lower() == "review"
    }

    manual_by_id, manual_counts = validate_manual_review(
        manual_rows,
        review_ids,
    )

    output_rows = merge_final_decisions(
        all_rows,
        manual_by_id,
    )

    final_counts = check_final_counts(
        output_rows,
        automated_counts,
        manual_counts,
    )

    clean_rows = [
        row
        for row in output_rows
        if clean_text(
            row.get("final_decision")
        ).lower() == "include"
    ]

    missing_reference_count = (
        check_application_reference_duplicates(
            clean_rows
        )
    )

    summary = {
        "input_rows": len(all_rows),
        "automated_triage": {
            "include": automated_counts["include"],
            "exclude": automated_counts["exclude"],
            "review": automated_counts["review"],
        },
        "manual_review": {
            "include": manual_counts.get("include", 0),
            "exclude": manual_counts.get("exclude", 0),
            "unresolved": manual_counts.get("unresolved", 0),
        },
        "final_classification": {
            "eligible_include": final_counts.get("include", 0),
            "ineligible_exclude": final_counts.get("exclude", 0),
            "unresolved_withheld": final_counts.get("unresolved", 0),
            "total": len(output_rows),
        },
        "duplicate_qa": {
            "missing_lpa_name_or_lpa_app_no": missing_reference_count,
        },
        "reconciliation_passed": True,
    }

    write_csv(
        FINAL_MASTER_PATH,
        output_rows,
    )
    write_csv(
        FINAL_CLEAN_PATH,
        clean_rows,
    )

    with MERGE_SUMMARY_PATH.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print("Manual review merge passed.")
    print(f"Input rows: {len(all_rows)}")
    print(f"Manual review rows: {len(manual_rows)}")
    print(
        f"Final include: "
        f"{final_counts.get('include', 0)}"
    )
    print(
        f"Final exclude: "
        f"{final_counts.get('exclude', 0)}"
    )
    print(
        f"Final unresolved: "
        f"{final_counts.get('unresolved', 0)}"
    )
    print(
        "Duplicate application-reference groups: 0"
    )
    print(
        f"Missing LPA/application reference: "
        f"{missing_reference_count}"
    )
    print("All final screening reconciliation checks passed.")


if __name__ == "__main__":
    main()
