# From Probability of Default to Expected Credit Loss

Start with [`notebooks/02_modeling_data_handoff.ipynb`](notebooks/02_modeling_data_handoff.ipynb). It inspects the complete selected vintages and demonstrates loading predictors, labels, chronological partitions and separate loss outcomes. The repository now also contains a reproducible PD, LGD, ECL and stress-testing layer in `src/credit_model.py`.

## Data and scope

The selected data are **all loans in the Freddie Mac Standard full annual archives for 2008, 2018 and 2023**, with all four quarters per year. Release 48 provides performance through June 2026. The three original ZIPs total **2,222,760,646 bytes (2.223 GB)**. Raw text stays compressed; cleaning streams nested quarterly ZIPs into Parquet. The 10 GB limit applies to all project data, including the earlier sample investigation, references, cleaned outputs, modeling outputs and temporary processing files.

**Completed:** 3,450,330 mortgages, 145,530,992 monthly records and 3,089,278 eligible labeled examples. All project data occupy approximately **6.03 GB**; the portable modeling folder is approximately **108.5 MB**. All 12 quarters passed uniqueness/key/linkage checks. **69 tests passed**, and the handoff notebook executed successfully.

| Ready partition | Loans | 12-month events |
|---|---:|---:|
| Train | 1,062,566 | 16,933 |
| Validation | 486,876 | 1,034 |
| Calibration | 668,644 | 1,871 |
| Test | 871,192 | 5,549 |

The loss table has **7,803 eligible resolved observations**. The audits retain 416 loans with 2,844 internal missing months, 7 field-level value violations, and one source-cohort mismatch (one loan / 209 monthly records in the 2008 Q4 archive with a 2009 Q1 identifier). The mismatch stays in full cleaned data and is excluded from ready partitions by the identifier-year rule. The code-update audit is in `data/reference/ingestion_code_history/`.


2008 supplies crisis-era outcomes, 2018 supplies later development/calibration data, and 2023 supplies an out-of-time test. These deliberately selected vintages are not representative of the entire US mortgage market or all Freddie Mac originations over time. The earlier 250,000-loan annual samples are retained separately under `data/raw/` and `data/processed/freddie_inspection/`; do not append them to these full-vintage tables because overlapping loans would be duplicated.

## Teammate quick start

Create an environment from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter lab notebooks/02_modeling_data_handoff.ipynb
```

Python 3.12 was used. On the current machine the existing environment is one folder above the repository: `../.venv/bin/python`.

The portable handoff in `~/Downloads/data_files.zip` has been extracted into
`data/modeling/` and the processed summary files. Run the complete modeling
pipeline from the repository root:

```bash
.venv/bin/python -m src.credit_model all --force
.venv/bin/python -m pytest -q
```

The command audits the source sample, trains logistic and XGBoost PD models,
fits calibration only on the calibration partition, fits LGD only on development
resolutions, evaluates the 2023 test set, and writes:

| Output | Contents |
|---|---|
| `reports/tables/pd_metrics.csv` | Raw and calibrated AUC, KS, Brier, log loss and calibration ratios |
| `reports/tables/stress_scenarios.csv` | Baseline, adverse and severe ECL |
| `reports/tables/sensitivity_grid_bps.csv` | PD multiplier and recovery haircut grid |
| `reports/tables/modeling_report.md` | Concise results and limitations |
| `reports/figures/` | Default rate, calibration, ECL bands, scenario and sensitivity figures |
| `models/pd/`, `models/lgd/` | Frozen estimators used for the test and ECL calculations |

Current 2023 out-of-time results use XGBoost as the selected ECL PD model:

| Metric | XGBoost | Logistic |
|---|---:|---:|
| AUC | 0.722 | 0.723 |
| KS | 0.333 | 0.321 |
| Brier | 0.00672 | 0.00670 |
| Mean PD / observed rate | 1.91x | 2.74x |

The 2023 event rate is 0.637%; calibrated XGBoost mean PD is 1.219%. The
material overprediction and negative static current-LTV LGD coefficient are
reported explicitly rather than hidden. At the 2023-06 portfolio date, the
selected baseline 12-month ECL rate is 0.351% of UPB, rising to 0.628% under the
adverse assumption and 1.212% under the severe assumption.

For model development, transfer the repository and **`data/modeling/`**, retaining their relative paths. The data are excluded from Git, so cloning the code alone does not include the training tables. The full `data/processed/full_population/` directory is needed to run every inspection cell or study monthly histories. Add `data/raw_full/` for complete reproduction. Source documentation and integrity manifests are in `data/reference/`. Follow Freddie Mac's access and usage terms when arranging access for collaborators; no upload or external sharing has been performed here.

```python
import json
from pathlib import Path
import pandas as pd

root = Path('.')  # repository root
folder = root / 'data/modeling'
contract = json.loads((folder / 'modeling_contract.json').read_text())
features, target = contract['feature_columns'], contract['target']
train = pd.read_parquet(folder / 'pd/train.parquet')
X_train, y_train = train[features], train[target].astype('int8')
validation = pd.read_parquet(folder / 'pd/validation.parquet')
X_validation, y_validation = validation[features], validation[target]
# calibration.parquet is reserved for fitting probability calibration.
# test.parquet is held out until model/preprocessing/calibration are frozen.
```

Use the explicit feature list: do not select every column except the target. Loan ID, date, cohort and refinance linkage are metadata. Future loss, termination, follow-up and event-date fields must never be predictors. Fit imputation, scaling, encoding, feature selection and any resampling only on training data. Missing values and original class prevalence are preserved. String codes, including postal prefixes, are categorical; some fields can be constant or entirely missing in a vintage. Choose an appropriate training-only policy for these fields.

## Target, exclusions and partitions

The prediction point is the **first observed performance reporting month**, which can be later than origination. This supports an acquisition-time assessment, not an application-time decision. Entry requires current delinquency status, positive UPB, no termination and no detected value-quality issue in the origination or baseline record.

`serious_delinquency_12m` is 1 for numeric delinquency of at least three months or REO acquisition (`RA`) during calendar months t+1 through t+12, up to the first observed termination. It is 0 only if all 12 months are observed, all delinquency statuses are known, and no event occurs. It is a research default proxy; a 90-day delinquency can cure.

Early payoff, sale/other exit, incomplete history, unknown status and administrative censoring retain a NULL label. Ineligible entry records also retain NULL labels. All source loans remain in `features/` and `outcomes/`; eligible labeled examples alone enter `pd/`. Excluding censored loans can bias estimated PD. Compare exclusion rates by cohort and consider survival/competing-risk modeling before applying results to the entire portfolio. Payoff is not silently treated as nondefault.

| Partition | Rule | Intended use |
|---|---|---|
| Train | Identifier vintage 2008; horizon ends before 2018 | Fit model/preprocessing |
| Validation | Identifier vintage 2018; first observed month before July 2018 | Choose settings |
| Calibration | Remaining 2018 entries; horizon ends before 2023 | Fit probability calibration |
| Test | Identifier vintage 2023; complete administrative horizon by June 2026 | Final out-of-time evaluation |

The 2018 partitions are retrospective development subsets; their outcome windows can overlap. All development label horizons precede test entry. This is not a deployment claim for 2018. Later loans with a disclosed HARP predecessor in another split are excluded from ready modeling tables. Undisclosed repeat borrowers cannot be identified. The final contract and invariant results are in `data/modeling/modeling_contract.json`.

## LGD, EAD and economic stress

`data/modeling/lgd_ead_outcomes.parquet` contains separate future outcomes for loans ever observed seriously delinquent. `use_for_lgd` selects eligible 12-month events with a resolved credit disposition (codes 02/03/09), disclosed actual loss and positive observed EAD, respecting resolution-date cutoffs for development partitions. Inspect the availability summary before fitting: this subset is much smaller and selected by observable resolution.

Observed EAD is positive UPB in the first serious-delinquency month, otherwise an explicitly marked immediately preceding calendar-month proxy. The observed loss fraction is signed `actual_loss / observed_ead`. Negative fractions and values above 1 remain intact. Missing loss/recovery is not zero, and unresolved defaults do not have measured zero LGD. The first delinquency can cure before a later disposition, so these outcomes do not isolate a single default episode. Full histories are available to refine episode definitions. Recovery information from a retrospectively corrected release is not proof that it was available in real time.

Expected loss later requires a calibrated PD and consistent LGD/EAD estimates or explicit assumptions. Original/current balance is not automatically future EAD. Recession scenarios, macroeconomic joins, recovery sensitivity, discounting and accounting ECL adjustments are modeling work and are not estimated by this cleaning pipeline. Keep macro series aligned to their historical release dates if they become predictors.

## Quality, provenance and reproduction

The July 2026 schema has 31 origination and 35 monthly fields. Every physical raw line is checked for width. Field-specific missing codes are converted without treating real zero as missing. Delinquency 99 is valid; XX is unavailable. Dates use actual calendar months because reported loan age can reset after modification. Original loss/recovery signs are preserved.

The pipeline retains all rows, requires unique origination and loan-month keys, and checks linkage both ways. Value violations are flagged and reported instead of deleting whole loans. Source row numbers and member names link cleaned data to the original archives. A source-quarter/identifier-quarter disagreement is reported separately; partition assignment uses the identifier vintage, with unmatched years excluded from ready tables.

Download access requires free registration/sign-in at [Freddie Mac Clarity](https://claritydownload.fmapps.freddiemac.com/CRT/). Select **Standard full data** for 2008, 2018 and 2023, preserving filenames `historical_data_YYYY.zip` in `data/raw_full/`. Do not unzip. Use the same release for every year. The downloaded archives' SHA-256 hashes, sizes, source and date are in `data/reference/full_download_manifest.json`.

```bash
.venv/bin/python -m src.full_population --root . --years 2008 2018 2023 --workers 2
.venv/bin/python -m src.modeling_data --root .
.venv/bin/python scripts/verify_handoff.py
.venv/bin/python -m pytest -q
```

Full ingestion resumes completed quarters only after matching source/code identity and output file sizes. It uses two bounded workers, compressed shards and bounded DuckDB scratch space. Modeling refuses to overwrite an existing completed output; preserve or deliberately remove that output before an intentional rebuild. A new release or code change must not silently reuse an older cleaned dataset. Integrity verification compares delivered bytes and SHA-256 checksums; it will intentionally report changes after a different rebuild. `scripts/verify_handoff.py --modeling-only` checks only the portable modeling folder. `--create` records a new manifest after a deliberate, validated rebuild.

| Path | Contents |
|---|---|
| `notebooks/02_modeling_data_handoff.ipynb` | Executed full-data inspection and loading examples |
| `notebooks/01_data_cleaning_inspection.ipynb` | Earlier sample cleaning/inspection, retained as exploratory history |
| `src/freddie_cleaning.py` | Schema and missing-value/range/category rules |
| `src/full_population.py` | Nested-ZIP ingestion, storage bounds and per-quarter audits |
| `src/modeling_data.py` | Calendar labels, partitions, future loss outcomes and contract |
| `data/raw_full/` | Three original full annual ZIPs |
| `data/processed/full_population/YYYYQn/` | Typed origination/monthly Parquet, coverage, missingness, audit and receipt |
| `data/modeling/pd/` | Train, validation, calibration and test Parquet |
| `data/modeling/features/`, `outcomes/` | Every loan's baseline features and separate label/censoring information |
| `data/modeling/lgd_ead_outcomes.parquet` | Future loss/exposure observations, eligibility and explicit proxies |
| `data/modeling/summaries/` | Partition counts, censoring, loss availability and inspection figure |
| `data/reference/` | Official documentation, archive provenance and handoff integrity |

Official sources: [Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset), [General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf), [File Layout](https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx), [File Headers](https://www.freddiemac.com/fmac-resources/research/docs/file_headers_july_2026.zip), [Release Notes](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf).
