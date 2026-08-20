"""
Generate the two tables retained for the dissertation.

Outputs:
- Table_1_descriptive_statistics.csv
- Table_2_main_models.csv

sensitivity is presented in Appendix Figure A1.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_PATH = (
    PROJECT_ROOT / "data_processed" / "panel" / "centre_period_panel.csv"
)

KEY_RESULTS_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "analysis"
    / "models"
    / "model_key_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

TABLE1_PATH = OUTPUT_DIR / "Table_1_descriptive_statistics.csv"
TABLE2_PATH = OUTPUT_DIR / "Table_2_main_models.csv"


def ensure_outputs_do_not_exist():
    existing = [
        str(path)
        for path in [TABLE1_PATH, TABLE2_PATH]
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Table outputs already exist. Review and move them before rerunning:\n- "
            + "\n- ".join(existing)
        )


def build_table_1(panel):
    centre_ptal = (
        panel[["boundary_id", "ptal_mean_ai"]]
        .drop_duplicates("boundary_id")
    )

    centre_article4 = (
        panel.groupby("boundary_id")["article4_share"]
        .max()
    )

    sample = [
        ("Analytical centre-boundary features", panel["boundary_id"].nunique()),
        ("Observed periods", panel["period"].nunique()),
        ("Centre-period observations", len(panel)),
        ("Matched applications", int(panel["application_count"].sum())),
        ("Centre-periods with applications", int(panel["application_count"].gt(0).sum())),
        (
            "Centres with zero applications across all periods",
            int(
                panel.groupby("boundary_id")["application_count"]
                .sum().eq(0).sum()
            ),
        ),
        ("Centres ever positively exposed to Article 4",
            int(centre_article4.gt(0).sum())
        ),
        ("Centres never positively exposed to Article 4",
            int(centre_article4.eq(0).sum())
        ),
    ]

    rows = [
        {
            "Measure": "Panel A. Sample structure",
            "N / Value": "",
            "Mean": "",
            "SD": "",
            "Median": "",
            "Min": "",
            "Max": "",
        }
    ]

    rows.extend(
        [
            {
                "Measure": measure,
                "N / Value": value,
                "Mean": "",
                "SD": "",
                "Median": "",
                "Min": "",
                "Max": "",
            }
            for measure, value in sample
        ]
    )

    rows.append(
        {
            "Measure": "Panel B. Variable distribution",
            "N / Value": "",
            "Mean": "",
            "SD": "",
            "Median": "",
            "Min": "",
            "Max": "",
        }
    )

    def stats(label, values):
        values = pd.Series(values).dropna().astype(float)
        return {
            "Measure": label,
            "N / Value": len(values),
            "Mean": round(float(values.mean()), 4),
            "SD": round(float(values.std(ddof=1)), 4),
            "Median": round(float(values.median()), 4),
            "Min": round(float(values.min()), 4),
            "Max": round(float(values.max()), 4),
        }

    rows.extend(
        [
            stats("Application count (centre-period)", panel["application_count"]),
            stats("Article 4 share (centre-period)", panel["article4_share"]),
            stats("PTAL mean Access Index (centre)", centre_ptal["ptal_mean_ai"]),
        ]
    )

    return pd.DataFrame(rows)

def model_nobs(key, model):
    values = (
        key.loc[key["model"] == model, "nobs"]
        .dropna()
        .astype(int)
        .unique()
    )
    if len(values) != 1:
        raise ValueError(f"Inconsistent nobs for {model}: {values}")
    return f"{values[0]:,}"

def build_table_2(key):
    def result(model, term):
        row = key.loc[
            (key["model"] == model)
            & (key["term"] == term)
        ]
        return None if row.empty else row.iloc[0]

    columns = {
        "M1 Contextual OLS": "M1",
        "M2 Primary within-centre OLS": "M2",
        "M3 Poisson contextual comparison": "M3",
    }

    rows = []

    for label, term in [
        ("Article 4 share", "article4_share"),
        ("PTAL Access Index (centred)", "ptal_c"),
        ("Article 4 share × PTAL", "article4_share:ptal_c"),
    ]:
        estimate = {"Statistic": label}
        p_value_row = {"Statistic": "  p-value"}

        for heading, model in columns.items():
            row = result(model, term)

            if row is None:
                estimate[heading] = (
                    "Absorbed by centre FE"
                    if term == "ptal_c"
                    else "-"
                )
                p_value_row[heading] = ""
            else:
                estimate[heading] = (
                    f"{row['coef']:.4f} "
                    f"({row['std_error']:.4f})"
                )

                p_value = float(row["p_value"])

                p_value_row[heading] = (
                    "<0.001"
                    if p_value < 0.001
                    else f"{p_value:.3f}"
                )

        rows.extend([estimate, p_value_row])

    rows.extend(
        [
            {
                "Statistic": "Estimator",
                "M1 Contextual OLS": "OLS",
                "M2 Primary within-centre OLS": "OLS",
                "M3 Poisson contextual comparison": "Poisson GLM",
            },
            {
                "Statistic": "Fixed effects",
                "M1 Contextual OLS": "Borough + period",
                "M2 Primary within-centre OLS": "Centre-boundary + period",
                "M3 Poisson contextual comparison": "Borough + period",
            },
            {
                "Statistic": "Observations",
                "M1 Contextual OLS": model_nobs(key, "M1"),
                "M2 Primary within-centre OLS": model_nobs(key, "M2"),
                "M3 Poisson contextual comparison": model_nobs(key, "M3"),
            },
        ]
    )

    return pd.DataFrame(rows)


def main():
    ensure_outputs_do_not_exist()

    panel = pd.read_csv(PANEL_PATH)
    key = pd.read_csv(KEY_RESULTS_PATH)

    table1 = build_table_1(panel)
    table2 = build_table_2(key)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table1.to_csv(
        TABLE1_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    table2.to_csv(
        TABLE2_PATH,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    print(f"Table 1: {TABLE1_PATH}")
    print(f"Table 2: {TABLE2_PATH}")


if __name__ == "__main__":
    main()