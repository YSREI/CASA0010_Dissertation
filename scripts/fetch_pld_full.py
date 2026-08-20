"""
Fetch and freeze the complete broad-query PLD result set.

This file doing fetch only. Candidate classification and
study-period filtering belong in ``extract_pld_in_window.py``.
"""

import json
from datetime import datetime
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data_raw" / "pld"

SOURCE_ENDPOINT = (
    "https://planningdata.london.gov.uk/api-guest/applications/_search"
)

HEADERS = {
    "X-API-AllowRequest": "be2rmRnt&",
    "Content-Type": "application/json",
}

REQUEST_TIMEOUT_SECONDS = 120

# This snapshot contain every all-date match.
BROAD_QUERY = (
    '"Class MA" OR '
    '"Class E to C3" OR '
    '"Class E to Class C3" OR '
    '"Use Class E to C3" OR '
    '"Use Class E to Use Class C3" OR '
    '"commercial to residential" OR '
    '"office to residential" OR '
    '"shop to residential" OR '
    '"retail to residential" OR '
    '("Commercial, Business and Service" AND "Class C3") OR '
    '("Class E" AND "Class C3")'
)

SEARCHED_FIELDS = [
    "description",
    "application_details",
    "application_type",
    "application_type_full",
    "development_type",
]

REQUESTED_SOURCE_FIELDS = [
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

def make_query_body(size):
    """
    Build the PLD search request.
    The broad text query is identical for both requests:
    1.size=0: ask how many records exist.
    2.size=total_hits: total number of hits.

    """

    return {
        "size": size,
        "track_total_hits": True,
        "_source": REQUESTED_SOURCE_FIELDS,
        "query": {
            "query_string": {
                "query": BROAD_QUERY,
                "fields": SEARCHED_FIELDS,
            }
        },
    }


def request_pld(headers, request_body):
    """
    Send one PLD request and return the decoded JSON response.
    A failed HTTP request will stop the script.
    """

    response = requests.post(
        SOURCE_ENDPOINT,
        headers=headers,
        json=request_body,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    response_data = response.json()

    if "hits" not in response_data:
        raise ValueError("The PLD response does not contain a 'hits' section.")

    if "hits" not in response_data["hits"]:
        raise ValueError("The PLD response does not contain the expected hits list.")

    return response_data


def read_exact_total(response_data):
    """
    Read the exact number of records matching the broad query.
    track_total_hits=True should make relation='eq', which demonstrates the reported total is exact.
    """

    total = response_data["hits"].get("total")

    if isinstance(total, dict):
        if total.get("relation") != "eq":
            raise ValueError(
                "The PLD API did not report an exact hit total "
                f"Reported relation: {total.get('relation')}"
            )
        total_value = total.get("value")

    else:
        #Older Elasticsearch versions may return the total directly.
        total_value = total

    if isinstance(total_value, bool) or not isinstance(total_value, int):
        raise ValueError("The PLD API total is missing or invalid.")

    if total_value < 0:
        raise ValueError("The PLD API reported a negative hit total.")

    return total_value

def check_retrieval_integrity(retrieved_records, total_hits):
    """
    Prove that the raw retrieval is complete and contains unique documents
    """
    if len(retrieved_records) != total_hits:
        raise ValueError(
            f"Incomplete PLD retrieval: retrieved {len(retrieved_records)} "
            f"of {total_hits} records."
        )

    es_ids = []

    for hit in retrieved_records:
        es_id = hit.get("_id")

        if es_id is None or str(es_id).strip() == "":
            raise ValueError("At least one retrieved PLD record has no _id/es_id.")

        if not isinstance(hit.get("_source"), dict):
            raise ValueError(
                f"PLD record {es_id!r} has no usable _source object."
            )

        es_ids.append(str(es_id))

    if len(es_ids) != len(set(es_ids)):
        duplicate_count = len(es_ids) - len(set(es_ids))
        raise ValueError(
            "Duplicate es_id values were introduced during PLD pagination: "
            f"{duplicate_count} duplicate records."
        )

def write_jsonl(retrieved_records, output_path):
    """
    Write one preserved Elasticsearch hit per line, adding a clear es_id.
    """
    with output_path.open("x", encoding="utf-8") as output_file:
        for hit in retrieved_records:
            saved_hit = dict(hit)
            saved_hit["es_id"] = hit["_id"]

            output_file.write(
                json.dumps(saved_hit, ensure_ascii=False) + "\n"
            )


def main():


    """
    Ask the Api for the exact population size
    """
    count_response = request_pld(
        HEADERS,
        make_query_body(size=0),
    )

    total_hits = read_exact_total(count_response)

    print(f"PLD broad-query total: {total_hits}")


    if total_hits == 0:
        raise ValueError(
            "PLD broad query returned zero records. "
            "The extraction has been stopped for investigation."
        )

    """
    Request the exact population reported by PLD.
    If PLD could not return an exact size request, the script fails for investigation.
    """
    try:
        full_response = request_pld(
            HEADERS,
            make_query_body(size=total_hits),
        )

    except requests.RequestException as error:
        raise RuntimeError(
            "PLD did not allow the complete result set to be retrieved."
            "No dataset have been saved, a validated pagination method is required."
        ) from error

    second_total = read_exact_total(full_response)


    # The PLD index is live. 
    # If the total changes between the count request and the retrieval request,
    # rerun the extraction
    if second_total != total_hits:
        raise ValueError(
            "PLD total changed between the count request and full retrieval: "
            f"{total_hits} vs {second_total}. "
            "Run the extraction again."
        )

    retrieved_records = full_response["hits"]["hits"]

    """
    Uniqueness check
    """
    check_retrieval_integrity(
        retrieved_records,
        total_hits
    )

    """
    Create frozen raw snapshot path
    """
    retrieval_time = datetime.now().astimezone()
    run_date = retrieval_time.strftime("%Y-%m-%d")

    OUTPUT_DIRECTORY.mkdir(
        parents=True, 
        exist_ok=True
        )

    raw_output_path = (
        OUTPUT_DIRECTORY / f"pld_full_query_hits_{run_date}.jsonl"
    )

    metadata_output_path = (
        OUTPUT_DIRECTORY / f"pld_full_query_metadata_{run_date}.json"
    )

    # Never overwrite an existing snapshot
    if raw_output_path.exists():
        raise FileExistsError(
            f"raw PLD snapshot already exists: {raw_output_path}. "
        )

    if metadata_output_path.exists():
        raise FileExistsError(
            f"PLD metadata already exists: {metadata_output_path}. "
        )
    """
    Save the raw data after the uniquess check
    """
    write_jsonl(retrieved_records, raw_output_path)

    metadata = {
        "retrieval_timestamp": retrieval_time.isoformat(timespec="seconds"),
        "source_endpoint": SOURCE_ENDPOINT,
        "broad_query_text": BROAD_QUERY,
        "searched_fields": SEARCHED_FIELDS,
        "requested_source_fields": REQUESTED_SOURCE_FIELDS,

        # The requested size came from PLDs own reported population
        "retrieval_method": "single_request",
        "requested_result_size": total_hits,
        "api_reported_total_hits": total_hits,
        "actually_retrieved_hits": len(retrieved_records),
        "exact_reconciliation_passed": len(retrieved_records) == total_hits,
    }

    with metadata_output_path.open("x", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print(f"Saved {len(retrieved_records)} complete PLD hits to {raw_output_path}")
    print(f"Saved retrieval metadata to {metadata_output_path}")


if __name__ == "__main__":
    main()