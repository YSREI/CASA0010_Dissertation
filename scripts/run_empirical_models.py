"""
Estimate the final exploratory panel specifications.

Main outcome: 
application_count

Primary exposure:
article4_share

Accessibility moderator:
ptal_mean_ai, centred at the mean across the 1,122 analytical centre-boundary features.

Model structure:
M1: OLS, local planning-area + period fixed effects
    article4_share * centred PTAL
M2: OLS, centre-boundary + period fixed effects
    article4_share + article4_share * centred PTAL
    (primary within-centre specification)
M3: Poisson, local planning-area + period fixed effects
    article4_share * centred PTAL
    (contextual comparison)

Predefined sensitivity:
    - treated_10 / treated_25 / treated_50 in the M2 fixed-effects structure
    - Tier-1-only M2
    - nine-complete-quarter M2

Standard errors are clustered by analytical centre-boundary feature.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_INPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "panel"
    / "centre_period_panel.csv"
)
PANEL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "panel"
    / "centre_period_panel_summary.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data_processed"
    / "analysis"
    / "models"
)

KEY_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "model_key_results.csv"
)
SENSITIVITY_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "model_sensitivity_results.csv"
)
MARGINAL_EFFECTS_PATH = (
    OUTPUT_DIRECTORY
    / "article4_ptal_marginal_effects.csv"
)
SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "model_summary.json"
)

REQUIRED_COLUMNS = [
    "boundary_id",
    "centre_borough",
    "tier",
    "period",
    "article4_share",
    "treated_10",
    "treated_25",
    "treated_50",
    "ptal_mean_ai",
    "application_count",
]

COMPLETE_PERIODS = [
    "2021Q4",
    "2022Q1",
    "2022Q2",
    "2022Q3",
    "2022Q4",
    "2023Q1",
    "2023Q2",
    "2023Q3",
    "2023Q4",
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
    """Prevent model outputs from being silently replaced."""

    existing = []

    for path in [
        KEY_RESULTS_PATH,
        SENSITIVITY_RESULTS_PATH,
        MARGINAL_EFFECTS_PATH,
        SUMMARY_OUTPUT_PATH,
    ]:
        if path.exists():
            existing.append(str(path))

    if existing:
        raise FileExistsError(
            "Model outputs already exist. "
            "Review and move them before rerunning:\n- "
            + "\n- ".join(existing)
        )


def require_columns(frame):
    """Require only fields used directly by the model stage."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Balanced panel is missing required columns: "
            + ", ".join(missing)
        )


def read_panel(panel_summary):
    """Read the frozen balanced panel and reconcile its row count."""

    panel = pd.read_csv(
        PANEL_INPUT_PATH,
        dtype={
            "boundary_id": "string",
            "centre_borough": "string",
            "tier": "string",
            "period": "string",
        },
        low_memory=False,
    )

    require_columns(panel)

    expected_rows = int(
        panel_summary["panel"]["centre_period_rows"]
    )

    if len(panel) != expected_rows:
        raise ValueError(
            "Panel does not match centre_period_panel_summary.json: "
            f"{len(panel)} != {expected_rows}."
        )

    return panel


def prepare_model_data(panel):
    """Create the centred accessibility moderator without altering the panel."""

    model_data = panel.copy()

    centre_ptal = (
        model_data[
            ["boundary_id", "ptal_mean_ai"]
        ]
        .drop_duplicates("boundary_id")
    )

    ptal_mean = float(
        centre_ptal["ptal_mean_ai"].mean()
    )

    model_data["ptal_c"] = (
        model_data["ptal_mean_ai"]
        - ptal_mean
    )

    return model_data, ptal_mean


def fit_ols_context(model_data):
    """
    Estimate the contextual OLS specification.
    Local planning-area fixed effects retain the time-invariant PTAL main
    effect while absorbing broad local-area differences.
    """

    formula = (
        "application_count ~ "
        "article4_share * ptal_c "
        "+ C(centre_borough) "
        "+ C(period)"
    )

    return smf.ols(
        formula,
        data=model_data,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": model_data["boundary_id"],
        },
    )


def fit_ols_centre_fe(model_data):
    """
    Estimate the primary within-centre OLS specification.
    PTAL is time-invariant and is absorbed by centre fixed effects. 
    Its interaction with time varying Article 4 exposure remains estimable.
    """

    formula = (
        "application_count ~ "
        "article4_share "
        "+ article4_share:ptal_c "
        "+ C(boundary_id) "
        "+ C(period)"
    )

    return smf.ols(
        formula,
        data=model_data,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": model_data["boundary_id"],
        },
    )


def fit_poisson_context(model_data):
    """
    Estimate the Poisson functional-form check on the full panel.
    The same contextual fixed-effect structure as M1 is retained so centres
    with zero applications throughout remain in the analysis.
    """

    formula = (
        "application_count ~ "
        "article4_share * ptal_c "
        "+ C(centre_borough) "
        "+ C(period)"
    )

    return smf.glm(
        formula,
        data=model_data,
        family=sm.families.Poisson(),
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": model_data["boundary_id"],
        },
        maxiter=200,
    )


def fit_threshold_sensitivity(model_data, treatment):
    """Estimate a binary Article 4 threshold in the centre-FE structure."""

    formula = (
        f"application_count ~ "
        f"{treatment} "
        f"+ {treatment}:ptal_c "
        "+ C(boundary_id) "
        "+ C(period)"
    )

    return smf.ols(
        formula,
        data=model_data,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": model_data["boundary_id"],
        },
    )


def fit_continuous_sensitivity(model_data):
    """Estimate the primary continuous centre-FE structure on a subset."""

    formula = (
        "application_count ~ "
        "article4_share "
        "+ article4_share:ptal_c "
        "+ C(boundary_id) "
        "+ C(period)"
    )

    return smf.ols(
        formula,
        data=model_data,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": model_data["boundary_id"],
        },
    )


def extract_terms(
    result,
    model_name,
    model_label,
    terms,
):
    """Extract the substantive coefficients needed for reporting."""

    rows = []

    conf = result.conf_int()

    for term, term_label in terms:
        if term not in result.params.index:
            continue

        rows.append(
            {
                "model": model_name,
                "model_label": model_label,
                "term": term,
                "term_label": term_label,
                "coef": float(
                    result.params[term]
                ),
                "std_error": float(
                    result.bse[term]
                ),
                "p_value": float(
                    result.pvalues[term]
                ),
                "conf_low": float(
                    conf.loc[term, 0]
                ),
                "conf_high": float(
                    conf.loc[term, 1]
                ),
                "nobs": int(
                    result.nobs
                ),
            }
        )

    return rows


def calculate_marginal_effects(
    primary_result,
    panel,
    ptal_mean,
):
    """
    Calculate the M2 Article 4 slope across the observed PTAL distribution.
    Reported effects are for a 10 percentage-point increase in article4_share.
    """

    centre_ptal = (
        panel[
            ["boundary_id", "ptal_mean_ai"]
        ]
        .drop_duplicates("boundary_id")
    )

    low = float(
        centre_ptal["ptal_mean_ai"]
        .quantile(0.05)
    )
    high = float(
        centre_ptal["ptal_mean_ai"]
        .quantile(0.95)
    )

    values = np.linspace(
        low,
        high,
        100,
    )

    beta_a4 = float(
        primary_result.params[
            "article4_share"
        ]
    )
    beta_int = float(
        primary_result.params[
            "article4_share:ptal_c"
        ]
    )

    covariance = primary_result.cov_params()

    var_a4 = float(
        covariance.loc[
            "article4_share",
            "article4_share",
        ]
    )
    var_int = float(
        covariance.loc[
            "article4_share:ptal_c",
            "article4_share:ptal_c",
        ]
    )
    cov_a4_int = float(
        covariance.loc[
            "article4_share",
            "article4_share:ptal_c",
        ]
    )

    rows = []

    for ptal_value in values:
        ptal_c = (
            ptal_value
            - ptal_mean
        )

        slope_full_share = (
            beta_a4
            + beta_int * ptal_c
        )

        variance = (
            var_a4
            + (ptal_c ** 2) * var_int
            + 2 * ptal_c * cov_a4_int
        )

        se_full_share = float(
            np.sqrt(
                max(variance, 0)
            )
        )

        # Convert to the expected-count difference associated with
        # 10 percentage-point increase in Article 4 coverage.
        effect_10pp = (
            slope_full_share * 0.10
        )
        se_10pp = (
            se_full_share * 0.10
        )

        rows.append(
            {
                "ptal_mean_ai": float(
                    ptal_value
                ),
                "article4_effect_per_10pp": float(
                    effect_10pp
                ),
                "std_error": float(
                    se_10pp
                ),
                "conf_low": float(
                    effect_10pp
                    - 1.96 * se_10pp
                ),
                "conf_high": float(
                    effect_10pp
                    + 1.96 * se_10pp
                ),
            }
        )

    return pd.DataFrame(rows)


def treatment_support(panel):
    """Record treatment support for the three predefined binary thresholds."""

    rows = []

    periods = int(
        panel["period"].nunique()
    )

    for treatment in [
        "treated_10",
        "treated_25",
        "treated_50",
    ]:
        treatment_sum = (
            panel.groupby("boundary_id")[
                treatment
            ]
            .sum()
        )

        rows.append(
            {
                "treatment": treatment,
                "treated_centre_periods": int(
                    panel[treatment].sum()
                ),
                "always_treated_centres": int(
                    treatment_sum.eq(
                        periods
                    ).sum()
                ),
                "never_treated_centres": int(
                    treatment_sum.eq(0).sum()
                ),
                "switching_centres": int(
                    (
                        treatment_sum.gt(0)
                        & treatment_sum.lt(periods)
                    ).sum()
                ),
            }
        )

    return rows


def main():
    """Estimate final models and save only dissertation-relevant outputs."""

    ensure_outputs_do_not_exist()

    panel_summary = read_json(
        PANEL_SUMMARY_PATH
    )

    panel = read_panel(
        panel_summary
    )

    model_data, ptal_mean = (
        prepare_model_data(
            panel
        )
    )

    # Main model family
    m1 = fit_ols_context(
        model_data
    )

    m2 = fit_ols_centre_fe(
        model_data
    )

    m3 = fit_poisson_context(
        model_data
    )

    key_rows = []

    key_rows.extend(
        extract_terms(
            m1,
            "M1",
            "OLS: local planning-area + period FE",
            [
                (
                    "article4_share",
                    "Article 4 share",
                ),
                (
                    "ptal_c",
                    "PTAL Access Index (centred)",
                ),
                (
                    "article4_share:ptal_c",
                    "Article 4 share × PTAL",
                ),
            ],
        )
    )

    key_rows.extend(
        extract_terms(
            m2,
            "M2",
            "OLS: centre-boundary + period FE",
            [
                (
                    "article4_share",
                    "Article 4 share",
                ),
                (
                    "article4_share:ptal_c",
                    "Article 4 share × PTAL",
                ),
            ],
        )
    )

    key_rows.extend(
        extract_terms(
            m3,
            "M3",
            "Poisson: local planning-area + period FE",
            [
                (
                    "article4_share",
                    "Article 4 share",
                ),
                (
                    "ptal_c",
                    "PTAL Access Index (centred)",
                ),
                (
                    "article4_share:ptal_c",
                    "Article 4 share × PTAL",
                ),
            ],
        )
    )

    key_results = pd.DataFrame(
        key_rows
    )

    # Predefined threshold sensitivity
    sensitivity_rows = []

    for treatment in [
        "treated_10",
        "treated_25",
        "treated_50",
    ]:
        result = fit_threshold_sensitivity(
            model_data,
            treatment,
        )

        sensitivity_rows.extend(
            extract_terms(
                result,
                treatment,
                (
                    "OLS centre-boundary + period FE: "
                    + treatment
                ),
                [
                    (
                        treatment,
                        treatment,
                    ),
                    (
                        f"{treatment}:ptal_c",
                        treatment + " × PTAL",
                    ),
                ],
            )
        )

    # Tier-1-only sensitivity
    tier1 = model_data[
        model_data["tier"] == "tier1"
    ].copy()

    tier1_result = fit_continuous_sensitivity(
        tier1
    )

    sensitivity_rows.extend(
        extract_terms(
            tier1_result,
            "tier1_only",
            "OLS centre-boundary + period FE: Tier 1 only",
            [
                (
                    "article4_share",
                    "Article 4 share",
                ),
                (
                    "article4_share:ptal_c",
                    "Article 4 share × PTAL",
                ),
            ],
        )
    )

    # Nine-complete-quarter sensitivity
    complete_quarters = model_data[
        model_data["period"].isin(
            COMPLETE_PERIODS
        )
    ].copy()

    complete_result = fit_continuous_sensitivity(
        complete_quarters
    )

    sensitivity_rows.extend(
        extract_terms(
            complete_result,
            "complete_quarters",
            "OLS centre-boundary + period FE: nine complete quarters",
            [
                (
                    "article4_share",
                    "Article 4 share",
                ),
                (
                    "article4_share:ptal_c",
                    "Article 4 share × PTAL",
                ),
            ],
        )
    )

    sensitivity_results = pd.DataFrame(
        sensitivity_rows
    )

    marginal_effects = (
        calculate_marginal_effects(
            m2,
            panel,
            ptal_mean,
        )
    )

    support = treatment_support(
        panel
    )

    summary = {
        "created_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "input": {
            "panel_rows": len(panel),
            "centre_count": int(
                panel["boundary_id"].nunique()
            ),
            "period_count": int(
                panel["period"].nunique()
            ),
            "application_count": int(
                panel["application_count"].sum()
            ),
            "centre_periods_with_applications": int(
                panel["application_count"]
                .gt(0)
                .sum()
            ),
            "all_zero_centres": int(
                panel.groupby("boundary_id")[
                    "application_count"
                ]
                .sum()
                .eq(0)
                .sum()
            ),
        },
        "variables": {
            "dependent_variable": "application_count",
            "primary_article4_measure": "article4_share",
            "accessibility_measure": "ptal_mean_ai",
            "ptal_centre_mean": ptal_mean,
            "interaction": "article4_share * centred ptal_mean_ai",
        },
        "models": {
            "M1": {
                "estimator": "OLS",
                "fixed_effects": [
                    "centre_borough",
                    "period",
                ],
                "cluster": "boundary_id",
                "role": "contextual comparison",
            },
            "M2": {
                "estimator": "OLS",
                "fixed_effects": [
                    "boundary_id",
                    "period",
                ],
                "cluster": "boundary_id",
                "role": "primary within-centre specification",
            },
            "M3": {
                "estimator": "Poisson GLM",
                "fixed_effects": [
                    "centre_borough",
                    "period",
                ],
                "cluster": "boundary_id",
                "role": "Poisson contextual comparison",
            },
        },
        "treatment_support": support,
        "sensitivity": {
            "thresholds": [
                "treated_10",
                "treated_25",
                "treated_50",
            ],
            "tier1_only_rows": len(
                tier1
            ),
            "complete_quarter_rows": len(
                complete_quarters
            ),
        },
    }

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    key_results.to_csv(
        KEY_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    sensitivity_results.to_csv(
        SENSITIVITY_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    marginal_effects.to_csv(
        MARGINAL_EFFECTS_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    with SUMMARY_OUTPUT_PATH.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    print(
        "Primary outcome: application_count"
    )
    print(
        "Primary exposure: article4_share"
    )
    print(
        "Accessibility moderator: ptal_mean_ai"
    )
    print(
        f"PTAL mean used for centring: {ptal_mean:.6f}"
    )
    print(
        f"M1 observations: {int(m1.nobs)}"
    )
    print(
        f"M2 observations: {int(m2.nobs)}"
    )
    print(
        f"M3 observations: {int(m3.nobs)}"
    )

    print("\nKey model terms:")
    print(
        key_results[
            [
                "model",
                "term",
                "coef",
                "std_error",
                "p_value",
            ]
        ].to_string(index=False)
    )

    print("\nSensitivity terms:")
    print(
        sensitivity_results[
            [
                "model",
                "term",
                "coef",
                "std_error",
                "p_value",
            ]
        ].to_string(index=False)
    )

    print(f"\nKey results: {KEY_RESULTS_PATH}")
    print(
        f"Sensitivity results: {SENSITIVITY_RESULTS_PATH}"
    )
    print(
        f"Marginal effects: {MARGINAL_EFFECTS_PATH}"
    )
    print(
        f"Model summary: {SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
