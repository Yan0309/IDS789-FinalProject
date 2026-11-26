"""Create a lightweight notebook over the completed, full-population outputs."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# Mortgage credit risk: modeling data handoff

**Start here for modeling.** This notebook inspects all loans in the Freddie Mac Standard **2008, 2018 and 2023** origination vintages, after cleaning and label construction. It reads the completed Parquet outputs; it does not repeat the large download or ingestion.

The first notebook (`01_data_cleaning_inspection.ipynb`) is the earlier 250,000-loan sample investigation. Those samples are kept separately and are **not pooled into these full-vintage tables**.

The prediction point is the **first observed performance month**, with a 12-month serious-delinquency outcome. This is an acquisition-time research dataset, not an application-time model, accounting ECL, or a fitted model.

Source: [Freddie Mac dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset), [user guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf). Release 48; performance through June 2026. Local source copies and checksums are in `data/reference/`.''')
code('''from pathlib import Path
import sys, os, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/modeling_data.py').exists())
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import pandas as pd
import matplotlib.pyplot as plt
import duckdb
from IPython.display import display
from src.full_population import data_bytes

DATA = ROOT/'data/modeling'
FULL = ROOT/'data/processed/full_population'
contract = json.loads((DATA/'modeling_contract.json').read_text())
summary = json.loads((FULL/'summary.json').read_text())
assert summary['complete'] and all(contract['invariants'].values())
print(f"Full vintages: {summary['origination_rows']:,} loans; {summary['performance_rows']:,} monthly records")
print(f"Ready labeled examples: {contract['pd_ready_rows']:,}")
print(f"All project data: {data_bytes(ROOT)/1e9:.3f} GB / 10 GB")
display(pd.read_json(ROOT/'data/reference/full_download_manifest.json')[['year','bytes','release','downloaded_date']])''')
md('''## 1. Cleaning and structural checks

The pipeline validates every raw row against the July 2026 31/35-field layout, preserves source row numbers, applies field-specific missing codes, and writes typed Parquet. Dates use calendar months; categorical IDs and ZIP prefixes stay strings. Missing recoveries remain missing. Each quarter must pass unique origination keys, unique loan-month keys, valid keys and complete origination/performance linkage before publication.

Original signed loss/recovery amounts and extreme valid values remain intact. Value violations are flagged separately. An internal history gap does not automatically invalidate a loan; the target builder checks its actual future window.''')
code('''quality = pd.read_csv(FULL/'quality_by_quarter.csv')
display(quality)
assert quality.structural_pass.all()
print('Internal gap loans:', f'{quality.loans_with_internal_gaps.sum():,}')
print('Internal missing months:', f'{quality.internal_gap_months.sum():,}')
print('Flagged value violations:', f'{quality.value_violation_count.sum():,}')
audits = pd.concat([pd.read_csv(p).assign(cohort=p.parent.name) for p in FULL.glob('*/cleaning_audit.csv')])
violations = audits[audits.issue.isin(['malformed_count','range_violation_count','code_violation_count']) & audits['count'].gt(0)]
display(violations if len(violations) else pd.DataFrame({'result':['No detected value violations']}))''')
md('''## 2. Target and chronological partitions

Entry requires a currently performing loan, positive balance, no termination and no flagged origination/baseline value violations. The target is **1** if delinquency reaches at least three months or REO (`RA`) in calendar months **t+1 through t+12**, up to the first observed termination. It is **0** only with all 12 months observed and all statuses known, without an event.

Early payoff, other termination, missing history and unknown status retain a **NULL** target. All loans remain in `features/` and `outcomes/`; only eligible labeled rows enter `pd/`. This complete-outcome sample can be selected: compare censoring by cohort and consider survival/competing-risk methods before generalizing PD to every loan.

- **Train:** 2008 vintage, outcomes mature before 2018.
- **Validation:** 2018 entries before July; use for hyperparameters/feature choices.
- **Calibration:** remaining 2018 entries, outcomes mature before 2023; reserve for calibration.
- **Test:** 2023 vintage, mature 12-month window; keep untouched during development.

The two 2018 subsets are retrospective development partitions and their outcome windows can overlap. All development outcome windows precede test entry. Disclosed cross-split HARP refinancing links are excluded from later ready splits. Other repeat borrowers cannot be identified in these public data.''')
code('''counts = pd.read_csv(DATA/'summaries/pd_split_counts.csv')
order = ['train','validation','calibration','test']
counts = counts.set_index('split').loc[order].reset_index()
counts['observed_event_rate'] = counts.positive/counts.rows
display(counts)
statuses = pd.read_csv(DATA/'summaries/outcome_status_counts.csv')
display(statuses.pivot(index='outcome_status', columns='split', values='loans').fillna(0).astype(int))
print('Excluded known cross-split refinance loans:', contract['excluded_known_cross_split_refinances'])
display(pd.Series(contract['invariants'], name='passed'))''')
code('''fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].bar(counts.split, counts.rows/1e6, color='#246a91')
axes[0].set(ylabel='Labeled loans (millions)', title='Data available for each role')
axes[1].bar(counts.split, 100*counts.observed_event_rate, color='#d17b32')
axes[1].set(ylabel='Observed 12-month events (%)', title='Event rate among eligible labeled loans')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(DATA/'summaries/split_overview.png', dpi=140, bbox_inches='tight')
plt.show()''')
md('''## 3. Load predictors and target safely

Use the **explicit feature list** in `modeling_contract.json`. Never take every non-target column as a predictor: IDs, cohort, date and refinance linkage are metadata. Future histories, exit status and realized losses live in separate tables.

Missing values are intentionally preserved. Fit imputation, scaling, encoding and feature selection on training data only, using an sklearn pipeline or an equivalent model-specific process. Postal codes and all string codes are categorical. Some fields are constant or entirely absent in a vintage; handle this using training data without inspecting test performance. The selected historical release contains retrospective corrections, so true real-time availability is not established.''')
code('''features = contract['feature_columns']
target = contract['target']
train = pd.read_parquet(DATA/'pd/train.parquet')
X_train = train.loc[:, features]
y_train = train[target].astype('int8')
assert len(train) == int(counts.loc[counts.split.eq('train'), 'rows'].iloc[0])
assert y_train.isin([0,1]).all() and train.loan_identifier.is_unique
assert not set(features).intersection({target,'loan_identifier','actual_loss','observed_ead','resolution_date'})
print('X_train:', X_train.shape, 'y_train:', y_train.shape)
display(pd.DataFrame({'dtype':X_train.dtypes.astype(str), 'missing_fraction':X_train.isna().mean()}).sort_values('missing_fraction', ascending=False))
display(X_train.head())''')
md('''Use the same feature order for other partitions; fit no preprocessing on them:
```python
validation = pd.read_parquet(DATA/'pd/validation.parquet')
calibration = pd.read_parquet(DATA/'pd/calibration.parquet')
test = pd.read_parquet(DATA/'pd/test.parquet')
X_validation, y_validation = validation[features], validation[target]
# Train model on X_train/y_train; choose settings with validation.
# Fit probability calibration on calibration[features]/calibration[target].
# Evaluate the frozen pipeline once on test[features]/test[target].
```
Report PR-AUC, ROC-AUC, Brier score and calibration by cohort; handle class imbalance without changing the evaluation prevalence. Large time gaps and changing underwriting create substantial distribution shift. No model is trained by this notebook.''')
md('''## 4. Loss given default and exposure: separate future outcomes

`lgd_ead_outcomes.parquet` contains loans ever observed seriously delinquent. `use_for_lgd` restricts a smaller subset to eligible 12-month events with a disclosed credit-disposition loss and a positive EAD measurement, with resolution timing appropriate for the partition.

EAD is positive UPB in the first serious-delinquency month, otherwise an explicitly marked immediately preceding calendar-month proxy. The observed loss fraction is signed `actual_loss / observed_ead`; negative values and values above one are preserved. Missing losses or unresolved defaults are not zero. A loan may cure and default again before its later disposition, so this fraction does not isolate a single default episode and requires review before LGD modeling. Resolution-only samples are selected and incomplete.

For eventual expected loss, combine a calibrated PD, an explicit LGD assumption/model and an EAD assumption/model consistently at the same horizon. Current or original balance is not automatically future EAD. Economic scenarios, probability calibration and accounting ECL adjustments remain modeling work.''')
code('''lgd_summary = pd.read_csv(DATA/'summaries/lgd_availability.csv')
display(lgd_summary)
loss = pd.read_parquet(DATA/'lgd_ead_outcomes.parquet')
display(loss.groupby(['split','ead_basis'], dropna=False).size().rename('loans').reset_index())
print('Ready LGD rows:', int(loss.use_for_lgd.fillna(False).sum()))
print('Missing observed loss fraction:', int(loss.observed_loss_fraction.isna().sum()))
display(loss[['loan_identifier','default_date','resolution_date','observed_ead','ead_basis',
              'actual_loss','observed_loss_fraction','use_for_lgd']].head(10))''')
md('''## 5. Inspect full histories without loading them into memory

Monthly Parquet files retain all source performance rows. DuckDB can filter or aggregate them directly. The example only reads schema and counts from Parquet metadata; add a WHERE clause on a loan ID for a detailed history.''')
code('''con = duckdb.connect()
pattern = str(FULL/'*/performance/*.parquet').replace("'", "''")
monthly_count = con.execute(f"SELECT count(*) FROM read_parquet('{pattern}')").fetchone()[0]
assert monthly_count == summary['performance_rows']
print(f'{monthly_count:,} monthly rows are available for alternative landmarks or survival models.')
con.close()
# Release large in-memory objects; no mutation of the saved data.
del train, X_train, y_train, loss''')
md('''## 6. Reproduce or hand off

See `README.md` for environment setup and download instructions. On a new machine, transfer the repository plus `data/modeling/` for ordinary PD/LGD work. Include `data/processed/full_population/` for monthly-history research, and `data/raw_full/` for complete reproduction. Freddie Mac access and usage terms still apply; local data are excluded from Git.

From the repository root, after placing the three annual Standard full ZIPs in `data/raw_full/`:
```bash
python -m src.full_population --root . --years 2008 2018 2023 --workers 2
python -m src.modeling_data --root .
python scripts/verify_handoff.py
python -m pytest -q
```
Ingestion resumes completed quarters after checking source/code identity and file sizes. Modeling refuses to overwrite an existing completed output. The 10 GB cap includes samples, raw files, references, processed files and modeling outputs. Raw text is never fully extracted. Integrity verification checks the delivered file hashes.''')

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}, 'language_info':{'name':'python','version':'3.12'}})
path = ROOT/'notebooks/02_modeling_data_handoff.ipynb'
nbf.write(nb, path)
print(path)
