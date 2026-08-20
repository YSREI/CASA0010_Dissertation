# CASA0010_Dissertation

# CASA0010 Dissertation

## Article 4 Exposure and Commercial-to-Residential Planning Applications across London Centres: The Role of Public Transport Accessibility

This repository contains the reproducibility materials for my CASA0010 MSc Urban Spatial Science dissertation at UCL.

The dissertation examines how commercial-to-residential planning application activity is associated with Article 4 exposure across London centre-boundary features, and whether this relationship varies with public transport accessibility.

## Final analytical dataset

The final analysis uses:

* 1,122 analytical centre-boundary features;
* 11 observed periods from August 2021 to March 2024;
* 12,342 centre-period observations;
* 699 validated and spatially matched commercial-to-residential applications.

The primary variables are:

* `application_count` — matched eligible commercial-to-residential applications per centre-period;
* `article4_share` — share of each centre boundary covered by an active relevant Article 4 direction;
* `ptal_mean_ai` — area-weighted mean TfL Access Index.

## Repository structure

```text
scripts/          Data processing, spatial analysis and modelling scripts
manual_inputs/    Manual classification and validation decisions
data_processed/   Processed analytical datasets and final model outputs
outputs/figures/  Figures used in the dissertation
outputs/tables/   Tables used in the dissertation
```

Historical and superseded working files are not included.

## Main data sources

The analysis uses publicly available data from:

* Planning London Datahub — planning applications;
* Planning Local Plan Data — Local Plan centre geographies;
* Planning Data — Article 4 directions and direction areas;
* Transport for London — 2015 PTAL grid and Access Index.

Raw source datasets are not redistributed through this repository. They can be obtained from the original providers and reconstructed using the scripts supplied here.

## Analytical workflow

The main workflow is:

```text
PLD retrieval
→ temporal filtering
→ substantive eligibility screening
→ manual application review
→ centre-layer inventory and manual selection
→ centre construction
→ application spatial assignment
→ Article 4 relevance identification
→ Article 4 centre-period exposure
→ PTAL construction
→ balanced panel
→ empirical models
→ dissertation tables and figures
```

The primary empirical specification is an OLS model with centre-boundary and period fixed effects and standard errors clustered by `boundary_id`.

## Script order

A simplified execution order is:

```text
fetch_pld_full.py
extract_pld_in_window.py
clean_conversion_dataset.py
merge_manual_review.py

inventory_centre_layers.py
build_centres.py
spatial_join_centres.py

prepare_article4_relevant_areas.py
merge_article4_manual_review.py
build_article4_exposure.py

build_centre_ptal.py
build_balanced_panel.py
run_empirical_models.py

make_dissertation_tables.py
make_dissertation_figures.py
make_figure_1_spatial_context.py
```

Some scripts depend on source files downloaded from the original data providers and on the manual decision files included in `manual_inputs/`.

## Key final outputs

The principal analytical outputs include:

```text
data_processed/panel/centre_period_panel.csv
data_processed/article4/article4_centre_period.csv
data_processed/ptal/centre_ptal.csv
```

Model outputs include the main model estimates, sensitivity analyses and Article 4 × PTAL marginal associations.

Final dissertation figures and tables are stored under:

```text
outputs/figures/
outputs/tables/
```

## Reproducibility notes

The project uses explicit manual decision files for application eligibility, Local Plan centre-layer selection and Article 4 relevance review. These files are retained in the repository to preserve decision provenance.

Raw datasets are excluded because they are available from the original providers and may be subject to redistribution or file-size constraints.

The repository reflects the final analytical state used for the submitted dissertation. Historical and superseded versions are excluded to avoid ambiguity over the authoritative pipeline.

## Software

The analysis was conducted in Python. Required Python packages are listed in `requirements.txt`.

## Author

Yu Shi
MSc Urban Spatial Science
Centre for Advanced Spatial Analysis
University College London
2026
