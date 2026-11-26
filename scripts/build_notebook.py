"""Rebuild the first-step notebook; execution is a separate explicit step."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))

def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
> **Earlier sample investigation.** For the completed full-vintage dataset and modeling inputs, start with [02_modeling_data_handoff.ipynb](02_modeling_data_handoff.ipynb). Sample data below are kept separately and must not be appended to the full-vintage tables.

# From Probability of Default to Expected Credit Loss
## 01 · Data cleaning and inspection

**Question:** Which mortgages are likely to default, how much could a lender lose, and how does that change under economic stress?

This notebook prepares the first evidence base: untouched Freddie Mac annual sample archives, cleaned origination and monthly performance records, and checks of data quality and follow-up. Models, calibration, expected loss, and scenario estimates come later.

**Run all cells in order.** The notebook processes every loan ZIP in `data/raw/` in chunks, writes Parquet, and displays actual results. No loan data means an explicit `awaiting_data` state; there is no synthetic fallback. The combined raw/reference/processed data cap is **10 GB (decimal)**, with processing headroom.

Official sources: [dataset and access](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset), [General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf), [July 2026 layout](https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx), [headers](https://www.freddiemac.com/fmac-resources/research/docs/file_headers_july_2026.zip), and [release notes](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf). Local reference copies and checksums are in `data/reference/`. Release 48 reports performance through **June 2026**.
""")
md("""
### 1. Sample and unit of observation

An origination row represents one mortgage. A performance row represents one mortgage in one reporting month. The linkage key is `loan_identifier`; the monthly key is `(loan_identifier, reporting_period)`. Monthly rows must not be treated as independent loans.

Use the **Standard Dataset annual samples**, containing 50,000 loans per complete vintage. Suggested cohorts are 2006–2008 (crisis-era histories) and 2018–2019 (later underwriting and pandemic-era histories). The inventory below is authoritative for what was actually downloaded. Equal-size annual samples do not reproduce each year's full market volume, so pooled results describe the selected sample portfolio.

Freddie Mac access requires registration/sign-in. Keep original ZIPs in `data/raw/`; do not unzip or combine them with full-population files or another release of the same vintage.
""")
code("""
from pathlib import Path
import sys, os, json, shutil, zipfile
from datetime import datetime, timezone

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'src' / 'freddie_cleaning.py').is_file()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook from the FinalProject folder or its notebooks subfolder.')
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from IPython.display import display, Markdown
from src.freddie_cleaning import (ORIG_COLUMNS, PERF_COLUMNS, inventory_archives,
                                  iter_raw_chunks, clean_origination, clean_performance)
from src.inspection_pipeline import run_inspection

MAX_DATA_BYTES = 10_000_000_000
CHUNK_ROWS = 100_000
pd.set_option('display.max_columns', 16)
pd.set_option('display.max_rows', 40)
sns.set_theme(style='whitegrid', context='notebook', palette='deep')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})

def data_size_bytes():
    return sum(p.stat().st_size for p in (ROOT / 'data').rglob('*') if p.is_file())

print('Project:', ROOT)
print('Executed (UTC):', datetime.now(timezone.utc).isoformat())
print(f'Data footprint: {data_size_bytes()/1e9:.3f} GB / {MAX_DATA_BYTES/1e9:.0f} GB')
print(f'Free disk: {shutil.disk_usage(ROOT).free/1e9:.2f} GB')
""")
code("""
inventory = inventory_archives(ROOT / 'data' / 'raw', max_bytes=MAX_DATA_BYTES)
if inventory.empty:
    display(Markdown('**Awaiting loan files.** Download annual sample ZIPs from Clarity into `data/raw/`, then rerun all cells.'))
else:
    display(inventory[['cohort', 'kind', 'member', 'uncompressed_bytes', 'compressed_bytes']])
    archive_bytes = inventory.drop_duplicates('archive')['archive_bytes'].sum()
    print(f'{inventory.archive.nunique()} archives; {archive_bytes/1e6:.1f} MB compressed; '
          f'{inventory.uncompressed_bytes.sum()/1e9:.3f} GB text inside ZIPs (not extracted).')
    assert data_size_bytes() <= MAX_DATA_BYTES

reference_manifest = pd.read_json(ROOT / 'data' / 'reference' / 'manifest.json')
display(reference_manifest[['file', 'bytes', 'retrieved_date']])
""")
md("""
### 2. Verify the current schema

The July 2026 release changed field locations. Older column lists can silently shift values into the wrong variables. The parser checks **every row** against the current 31-field origination / 35-field performance layout and stops on a mismatch. Identifier, state, ZIP-prefix, and categorical codes are read as strings before any conversion.
""")
code("""
with zipfile.ZipFile(ROOT / 'data' / 'reference' / 'file_headers_july_2026.zip') as z:
    orig_headers = z.read('origination_data_file_header.txt').decode('utf-8-sig').strip().split('|')
    perf_headers = z.read('performance_data_file_header.txt').decode('utf-8-sig').strip().split('|')
assert len(orig_headers) == len(ORIG_COLUMNS) == 31
assert len(perf_headers) == len(PERF_COLUMNS) == 35
schema = pd.concat([
    pd.DataFrame({'table': 'origination', 'position': range(1, 32),
                  'official_name': orig_headers, 'python_name': ORIG_COLUMNS}),
    pd.DataFrame({'table': 'performance', 'position': range(1, 36),
                  'official_name': perf_headers, 'python_name': PERF_COLUMNS})
], ignore_index=True)
with pd.option_context('display.max_rows', 70):
    display(schema)
""")
md("""
### 3. Cleaning rules and a raw-to-clean preview

Cleaning is deterministic and does not learn from future data. Raw archives remain untouched; field audits record conversions and anomalies. We retain all source rows, including duplicates, with archive/member/row provenance.

| Field group | Treatment |
|---|---|
| FICO / VantageScore | Map documented unavailable codes to missing; check range |
| Ratios and counts | Apply each field's sentinel and vintage-specific range |
| Dates | Parse exact `YYYYMM` into calendar-month timestamps; audit malformed values |
| Delinquency | Keep raw code; `XX` unavailable, `RA` REO, numeric `99` valid severe status |
| Recovery / costs | Preserve absence, zero, and negative amounts as distinct where permitted |
| Duplicates / linkage | Count identical rows, conflicting keys, and unmatched identifiers separately |

The inspection flag `serious_delinquency` means an observed numeric status of at least three months delinquent or REO (`RA`). It is not yet the project's future default target. No imputation, scaling, winsorization, or silent duplicate removal occurs here. See `src/freddie_cleaning.py` for the full rules.
""")
code("""
if not inventory.empty and inventory.kind.eq('origination').any():
    first_orig = inventory.loc[inventory.kind.eq('origination')].iloc[0]
    reader = iter_raw_chunks(first_orig.archive, first_orig.member, 'origination', chunksize=1000)
    raw_preview = next(reader)
    reader.close()
    clean_preview, preview_audit = clean_origination(raw_preview)
    preview_fields = ['loan_identifier', 'classic_fico', 'first_payment_date',
                      'original_ltv', 'original_dti', 'mortgage_insurance_percentage']
    display(pd.concat({'raw': raw_preview[preview_fields].head(),
                       'clean': clean_preview[preview_fields].head()}, axis=1))
    display(preview_audit)
else:
    print('Raw-to-clean preview requires a real origination file.')
""")
md("""
### 4. Process all selected records

This cell streams ZIP members into compressed Parquet shards, then uses DuckDB for cross-file checks. It may take several minutes. It checks the combined data footprint during processing, reserves scratch space, and retains the prior successful output if a new run fails. Outputs are under `data/processed/freddie_inspection/`.
""")
code("""
result = run_inspection(ROOT, chunksize=CHUNK_ROWS, max_bytes=MAX_DATA_BYTES)
HAS_DATA = result['status'] == 'complete'
print('Inspection status:', result['status'])
print('Structural checks passed:', result['structural_quality_pass'])
print('Modeling ready:', result['modeling_ready'], '(no prediction labels have been constructed)')
display(result['overview'])
for warning in result['warnings']:
    print('•', warning)
if HAS_DATA:
    print(f'Current data footprint: {result["data_footprint_bytes"]/1e9:.3f} GB')
    display(result['inventory'][['member', 'archive_bytes', 'archive_sha256']])

def show_table(name, rows=25):
    table = result[name]
    if table.empty:
        print(f'{name}: no observations available.')
    else:
        display(table.head(rows))
""")
md("""
### 5. Missingness and anomalies

Missingness can reflect historical disclosure availability or a non-applicable field. It is not necessarily an error. For example, appraisal-method and estimated-LTV coverage differ across historical periods. Review missingness by cohort before choosing features.

Malformed or out-of-range values become missing in cleaned fields and remain traceable to raw data. The audit separates these failures from documented unavailable codes. Correctly unavailable values do not by themselves fail the structural check.
""")
code("""
audit = result['cleaning_audit']
if not audit.empty:
    display(audit.groupby(['kind', 'issue'], as_index=False)['count'].sum())
    problems = audit.loc[audit.issue.isin(['malformed', 'range_violation', 'code_violation', 'empty_member'])]
    display(problems.sort_values('count', ascending=False).head(30))
missing = result['missingness']
if not missing.empty:
    display(missing.sort_values(['kind', 'missing_fraction'], ascending=[True, False]).head(30))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, kind in zip(axes, ['origination', 'performance']):
        m = missing.loc[missing.kind.eq(kind) & missing.column.isin(ORIG_COLUMNS + PERF_COLUMNS)]
        m = m.nlargest(12, 'missing_fraction').sort_values('missing_fraction')
        ax.barh(m.column, 100*m.missing_fraction, color='#39789b')
        ax.set(title=f'{kind.title()}: highest missingness', xlabel='Missing (%)', xlim=(0, 100))
    plt.show()
""")
md("""
### 6. Loan characteristics

These distributions describe the downloaded sample. They are not fitted model inputs yet. Monetary balances are in US dollars; LTV and DTI are percentages, not fractions. Original UPB is rounded in the source. A zero mortgage-insurance percentage is valid and means no insurance.
""")
code("""
orig = result['origination']
show_table('origination_describe', 30)
if not orig.empty:
    fields = {'classic_fico': 'Classic FICO', 'original_ltv': 'Original LTV (%)',
              'original_dti': 'Original DTI (%)', 'original_upb': 'Original UPB ($)',
              'original_interest_rate': 'Original interest rate (%)', 'original_loan_term': 'Term (months)'}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (column, title) in zip(axes.flat, fields.items()):
        values = pd.to_numeric(orig[column], errors='coerce').dropna()
        ax.hist(values, bins=35, color='#39789b', edgecolor='white', linewidth=.3)
        ax.set(title=title, ylabel='Loan records')
    plt.show()
    display(orig.groupby('source_cohort')[list(fields)].median().round(2))
    selected = result['category_distributions'].query("column in ['loan_purpose', 'occupancy_status', 'property_type', 'property_state']")
    display(selected.groupby('column', group_keys=False).head(8))
""")
md("""
### 7. Duplicate keys and linkage

An exact duplicate repeats the original parsed fields. A conflicting key has distinct original records for the same loan (origination) or loan-month (performance). No arbitrary 'keep last' rule is applied because file order does not establish authority. Resolve these issues before making a one-to-many join for modeling.
""")
code("""
show_table('duplicate_quality')
show_table('linkage_quality')
if HAS_DATA:
    d = result['duplicate_quality']
    n = int(d['conflicting_keys'].sum())
    display(Markdown(f'**Conflicting keys requiring investigation: {n:,}.** '
                     'Zero conflicts does not establish complete follow-up or validate a future outcome definition.'))
""")
md("""
### 8. Performance coverage and observed outcomes

Performance starts at Freddie Mac acquisition, not necessarily origination. A loan with twelve records may still have calendar gaps. The coverage table measures distinct months and internal gaps between first and last observations; it does not detect all missing history before acquisition or after exit.

The observed-serious-delinquency count uses all supplied follow-up and is affected by different observation lengths. Do not compare it as a fixed-horizon default rate or use it directly as the training label. Loan age may reset after modification, so future windows must use calendar months.
""")
code("""
show_table('cohort_summary')
coverage = result['loan_coverage']
if not coverage.empty:
    display(coverage[['distinct_months', 'span_months', 'gap_months']].describe().round(2))
    print(f'Loans with internal calendar gaps: {coverage.gap_months.gt(0).sum():,}')
    print(f'Loans with entirely unknown observed delinquency: {coverage.observed_serious_delinquency.isna().sum():,}')
    monthly = result['month_counts']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(monthly.reporting_period, monthly.distinct_loans, color='#39789b')
    axes[0].set(title='Loans observed each month', xlabel='Reporting month', ylabel='Distinct loans')
    axes[1].hist(coverage.distinct_months.dropna(), bins=35, color='#c48b4b', edgecolor='white')
    axes[1].set(title='Observed follow-up length', xlabel='Distinct reporting months', ylabel='Loans')
    plt.show()
""")
md("""
### 9. Inspect one actual loan's history

The example is selected deterministically from observed serious-delinquency loans, if any exist, otherwise from all observed loans. It illustrates field behavior and is not representative of the portfolio. A zero-balance code can describe voluntary payoff or other exits; zero UPB alone does not define default.
""")
code("""
if HAS_DATA and not coverage.empty:
    pool = coverage.loc[coverage.observed_serious_delinquency.fillna(False)]
    example_id = (pool if not pool.empty else coverage).sort_values('loan_identifier').iloc[0].loan_identifier
    with duckdb.connect() as con:
        history = con.execute('''SELECT loan_identifier, reporting_period, current_actual_upb,
                    delinquency_status, delinquency_months, serious_delinquency,
                    zero_balance_code, modification_flag, actual_loss, source_row
                    FROM read_parquet(?, union_by_name=true)
                    WHERE loan_identifier=? ORDER BY reporting_period, source_row''',
                    [result['paths']['performance_glob'], example_id]).df()
    print('Example loan:', example_id)
    display(history.head(12))
    display(history.tail(8))
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, constrained_layout=True)
    axes[0].plot(history.reporting_period, history.current_actual_upb, color='#39789b')
    axes[0].set(ylabel='UPB ($)', title='One observed loan history')
    axes[1].plot(history.reporting_period, history.delinquency_months, color='#b45b49')
    axes[1].set(ylabel='Months delinquent', xlabel='Reporting month')
    plt.show()
""")
md(r"""
### 10. What is available for expected loss?

For a defined horizon and exposure, the project will eventually estimate
\[
\operatorname{ECL}=\sum_i \operatorname{PD}_i\times\operatorname{LGD}_i\times\operatorname{EAD}_i.
\]

This notebook only checks the ingredients. Recovery, expense, and loss fields are usually event-dependent and can be missing before disposition. Do not fill them with zero or treat them as application-time predictors. LGD requires an explicit event definition, recovery window, discounting/cost convention, and handling of unresolved cases. EAD should reflect the relevant future exposure; original principal is not automatically EAD.

**Monthly balance sums below are loan-month totals, not simultaneous portfolio exposure.** A portfolio exposure requires one common reporting date and one row per active loan.
""")
code("""
show_table('loss_field_availability', 30)
show_table('balance_summary', 20)
if HAS_DATA:
    with duckdb.connect() as con:
        snapshot = con.execute('''WITH p AS (SELECT * FROM read_parquet(?, union_by_name=true)),
                   latest AS (SELECT max(reporting_period) AS t FROM p)
                   SELECT reporting_period, count(*) AS records, count(DISTINCT loan_identifier) AS loans,
                          sum(current_actual_upb) AS total_reported_upb,
                          count(*)-count(current_actual_upb) AS missing_upb
                   FROM p,latest WHERE reporting_period=t GROUP BY reporting_period''',
                   [result['paths']['performance_glob']]).df()
    display(snapshot)
    print('Latest-month descriptive snapshot; verify duplicates and exits before treating this as EAD.')
""")
md("""
### 11. Decisions before the modeling notebook

1. **Prediction point and target:** Prefer an acquisition-time or monthly landmark task unless complete inception history is verified. A candidate outcome is incident 90+ day delinquency/REO in months `t+1` through `t+12`, among loans without a prior observed event. This is an analyst definition, not a finalized accounting-default definition.
2. **Censoring and exits:** Positive once the specified event is observed; a negative requires complete valid follow-up for the chosen estimand. Missing months, `XX`, incomplete histories, payoff, and sales need explicit flags and a competing-event or censoring policy. Twelve later rows do not imply twelve complete calendar months.
3. **Out-of-time design:** Split by calendar prediction date with outcome windows matured before each later evaluation period. Keep the same loan out of multiple partitions; consider linked refinances. Fit preprocessing on training data and probability calibration on an earlier held-out block, leaving the final future test untouched. The selected cohorts alone do not determine these cutoffs.
4. **Feature availability:** Use only fields known at the prediction date. Exclude future delinquency, modification, loss/recovery, and termination information. A retrospective corrected release cannot fully prove historical real-time data availability.
5. **Model comparison:** Logistic regression, then boosted trees. Evaluate ranking and probabilities separately (ROC-AUC/PR-AUC, Brier/log loss, calibration), with class prevalence and temporal stability.
6. **Loss and stress:** Estimate PD/LGD/EAD consistently; add external macroeconomic data only after specifying scenarios. Report PD and recovery sensitivity separately. Historical crisis cohorts can inform analysis, but do not by themselves constitute a calibrated recession stress test.

The immediate output is an inspected dataset and a list of issues to resolve. There are no model-performance, causal-stress, or profitability claims in this notebook.
""")
code("""
if HAS_DATA:
    overview = result['overview'].set_index('kind')
    display(Markdown(
        f'**This run:** {int(overview["rows"].get("origination", 0)):,} origination records and '
        f'{int(overview["rows"].get("performance", 0)):,} monthly performance records. '
        f'Data footprint: **{data_size_bytes()/1e9:.3f} GB**. '
        f'Structural quality flag: **{result["structural_quality_pass"]}**. '
        'Review the audit and follow-up tables before defining labels.'))
    print('Saved output:', result['paths']['output_dir'])
else:
    display(Markdown('**No mortgage observations were analyzed in this run.** The schema and workflow are ready; add official loan sample ZIPs and rerun.'))
assert data_size_bytes() <= MAX_DATA_BYTES, 'Data storage ceiling exceeded.'
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3 (project .venv)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12.14'},
}
nbf.validate(nb)
target = ROOT / 'notebooks' / '01_data_cleaning_inspection.ipynb'
nbf.write(nb, target)
print(f'Wrote {target} ({len(cells)} cells)')
