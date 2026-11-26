"""Construct acquisition-time PD inputs and separate future outcome tables.

This module prepares data only. No imputation, calibration, model fitting,
scenario probabilities, accounting ECL, or loan eligibility decisions occur.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import duckdb
import pandas as pd

from .full_population import MAX_BYTES, data_bytes, check_budget, sql_path

TARGET = 'serious_delinquency_12m'
CUTOFF = '2026-06-01'
ORIG_FEATURES = [
    'classic_fico','first_time_homebuyer_indicator','mortgage_insurance_percentage',
    'number_of_units','occupancy_status','original_cltv','original_dti','original_upb',
    'original_ltv','original_interest_rate','channel','prepayment_penalty_indicator',
    'amortization_type','property_state','property_type','postal_code',
    'loan_purpose','original_loan_term','number_of_borrowers','borrowers_top_coded_2plus',
    'harp_indicator','special_eligibility_program','property_valuation_method','interest_only_indicator',
]
BASELINE_FEATURES = ['baseline_upb','baseline_interest_rate','baseline_loan_age',
                     'baseline_remaining_months','months_since_first_payment','anchor_month']
FEATURES = ORIG_FEATURES + BASELINE_FEATURES


def copy_query(con, query: str, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    con.execute(f'COPY ({query}) TO {sql_path(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')


def configure(con, root: Path, scratch: Path):
    scratch.mkdir(parents=True,exist_ok=True)
    con.execute("SET memory_limit='1GB'")
    con.execute('SET threads=2')
    con.execute('SET enable_progress_bar=false')
    con.execute('SET preserve_insertion_order=false')
    con.execute(f'SET temp_directory={sql_path(scratch)}')
    limit=min(1_500_000_000,MAX_BYTES-data_bytes(root)-500_000_000,shutil.disk_usage(root).free-2_000_000_000)
    if limit<=0: raise RuntimeError('No modeling scratch capacity within the storage cap.')
    con.execute(f"SET max_temp_directory_size='{limit}B'")


def build_views(con):
    """Input views o (origination) and p (monthly performance) must be unique."""
    con.execute('''CREATE TEMP TABLE a AS
        SELECT p.loan_identifier, p.reporting_period AS as_of_date,
            p.reporting_period + INTERVAL '12 months' AS horizon_end,
            p.current_actual_upb AS baseline_upb, p.current_interest_rate AS baseline_interest_rate,
            p.loan_age AS baseline_loan_age, p.remaining_months_to_legal_maturity AS baseline_remaining_months,
            p.delinquency_months AS baseline_delinquency_months,
            p.zero_balance_code AS baseline_zero_balance_code,
            p.data_quality_issue AS baseline_data_quality_issue,
            o.data_quality_issue AS origination_data_quality_issue,
            coalesce(p.delinquency_months=0 AND p.current_actual_upb>0 AND p.zero_balance_code IS NULL
                     AND NOT p.data_quality_issue AND NOT o.data_quality_issue,FALSE) AS eligible_at_entry,
            CASE WHEN o.data_quality_issue OR p.data_quality_issue THEN 'data_quality_issue'
                 WHEN p.delinquency_months IS NULL THEN 'unknown_entry_status'
                 WHEN p.delinquency_months<>0 THEN 'not_current_at_entry'
                 WHEN p.current_actual_upb IS NULL OR p.current_actual_upb<=0 THEN 'no_positive_entry_balance'
                 WHEN p.zero_balance_code IS NOT NULL THEN 'terminated_at_entry' ELSE 'eligible' END AS entry_reason,
            o.origination_year, o.origination_quarter AS source_cohort
        FROM p JOIN (SELECT loan_identifier,min(reporting_period) t FROM p GROUP BY loan_identifier) first
             ON p.loan_identifier=first.loan_identifier AND p.reporting_period=first.t
        JOIN o ON p.loan_identifier=o.loan_identifier''')
    con.execute('''CREATE TEMP TABLE exits AS SELECT a.loan_identifier,
            min(p.reporting_period) FILTER (WHERE p.zero_balance_code IS NOT NULL) AS first_exit_date,
            arg_min(p.zero_balance_code,p.reporting_period) FILTER (WHERE p.zero_balance_code IS NOT NULL) AS first_exit_code
        FROM a LEFT JOIN p ON a.loan_identifier=p.loan_identifier
        AND p.reporting_period>a.as_of_date AND p.reporting_period<=a.horizon_end GROUP BY a.loan_identifier''')
    con.execute('''CREATE TEMP TABLE followup AS SELECT a.loan_identifier,
            count(p.reporting_period) AS observed_future_months,
            count(p.delinquency_status) AS known_status_months,
            count(p.reporting_period)-count(p.delinquency_status) AS unknown_status_months,
            min(p.reporting_period) FILTER (WHERE p.serious_delinquency) AS event_date_12m,
            max(p.reporting_period) AS last_observed_in_window
        FROM a JOIN exits e USING(loan_identifier) LEFT JOIN p ON p.loan_identifier=a.loan_identifier
            AND p.reporting_period>a.as_of_date AND p.reporting_period<=a.horizon_end
            AND (e.first_exit_date IS NULL OR p.reporting_period<=e.first_exit_date)
        GROUP BY a.loan_identifier''')
    con.execute(f'''CREATE TEMP TABLE outcomes AS WITH x AS (
        SELECT a.*, e.first_exit_date,e.first_exit_code,f.observed_future_months,f.known_status_months,
            f.unknown_status_months,f.event_date_12m,f.last_observed_in_window,
            12-f.observed_future_months AS unobserved_window_months,
            CASE WHEN NOT a.eligible_at_entry THEN NULL
                 WHEN f.event_date_12m IS NOT NULL THEN 1
                 WHEN f.observed_future_months=12 AND f.known_status_months=12 THEN 0
                 ELSE NULL END::TINYINT AS {TARGET},
            CASE WHEN a.origination_year=2008 AND a.horizon_end<DATE '2018-01-01' THEN 'train'
                 WHEN a.origination_year=2018 AND a.as_of_date<DATE '2018-07-01' THEN 'validation'
                 WHEN a.origination_year=2018 AND a.horizon_end<DATE '2023-01-01' THEN 'calibration'
                 WHEN a.origination_year=2023 AND a.as_of_date>=DATE '2023-01-01' AND a.horizon_end<=DATE '{CUTOFF}' THEN 'test'
                 ELSE 'excluded_time_window' END AS split
        FROM a JOIN exits e USING(loan_identifier) JOIN followup f USING(loan_identifier))
        SELECT *,
            CASE WHEN NOT eligible_at_entry THEN entry_reason
                 WHEN {TARGET}=1 THEN 'observed_event'
                 WHEN {TARGET}=0 THEN 'complete_non_event'
                 WHEN first_exit_code='01' THEN 'voluntary_payoff'
                 WHEN first_exit_code IS NOT NULL THEN 'other_exit_'||first_exit_code
                 WHEN horizon_end>DATE '{CUTOFF}' THEN 'administrative_censoring'
                 WHEN unknown_status_months>0 THEN 'unknown_status'
                 ELSE 'incomplete_history' END AS outcome_status,
            CASE WHEN {TARGET}=1 THEN event_date_12m WHEN {TARGET}=0 THEN horizon_end ELSE NULL END AS label_observation_end,
            eligible_at_entry AND {TARGET} IS NOT NULL AND split<>'excluded_time_window' AS candidate_for_pd
        FROM x''')
    selected=','.join(f'o."{c}"' for c in ORIG_FEATURES)
    con.execute(f'''CREATE TEMP VIEW features AS SELECT a.loan_identifier,a.as_of_date,a.source_cohort,
        o.pre_harp_loan_sequence_number AS refinance_predecessor_id,
        {selected},a.baseline_upb,a.baseline_interest_rate,a.baseline_loan_age,a.baseline_remaining_months,
        date_diff('month',o.first_payment_date,a.as_of_date) AS months_since_first_payment,
        month(a.as_of_date) AS anchor_month FROM a JOIN o USING(loan_identifier)''')

    # Future loss data lives in a separate table. A default can cure; absence of
    # a disclosed resolution is not interpreted as zero loss or zero recovery.
    con.execute('''CREATE TEMP TABLE defaults AS SELECT loan_identifier,
        min(reporting_period) FILTER (WHERE serious_delinquency) AS default_date
        FROM p GROUP BY loan_identifier HAVING default_date IS NOT NULL''')
    con.execute('''CREATE TEMP TABLE resolved AS SELECT d.loan_identifier,
        min(p.reporting_period) AS resolution_date
        FROM defaults d JOIN p USING(loan_identifier)
        WHERE p.reporting_period>=d.default_date AND p.zero_balance_code IN ('02','03','09')
              AND p.actual_loss IS NOT NULL GROUP BY d.loan_identifier''')
    con.execute(f'''CREATE TEMP VIEW loss_outcomes AS WITH base AS (
        SELECT d.loan_identifier,d.default_date,r.resolution_date,
            event.current_actual_upb AS upb_in_default_month,
            prev.current_actual_upb AS prior_calendar_month_upb,
            CASE WHEN event.current_actual_upb>0 THEN event.current_actual_upb
                 WHEN prev.current_actual_upb>0 THEN prev.current_actual_upb ELSE NULL END AS observed_ead,
            CASE WHEN event.current_actual_upb>0 THEN 'default_month_upb'
                 WHEN prev.current_actual_upb>0 THEN 'previous_month_upb_proxy' ELSE 'unavailable' END AS ead_basis,
            res.zero_balance_code AS resolution_code,res.actual_loss,res.zero_balance_removal_upb,
            res.net_sales_proceeds,res.mi_recoveries,res.non_mi_recoveries,res.total_expenses,
            res.delinquent_accrued_interest,res.bankruptcy_cramdown_costs,
            target.split,target.{TARGET},target.candidate_for_pd
        FROM defaults d JOIN p event ON event.loan_identifier=d.loan_identifier AND event.reporting_period=d.default_date
        LEFT JOIN p prev ON prev.loan_identifier=d.loan_identifier AND prev.reporting_period=d.default_date-INTERVAL '1 month'
        LEFT JOIN resolved r ON r.loan_identifier=d.loan_identifier
        LEFT JOIN p res ON res.loan_identifier=d.loan_identifier AND res.reporting_period=r.resolution_date
        JOIN outcomes target ON target.loan_identifier=d.loan_identifier)
        SELECT *, actual_loss/observed_ead AS observed_loss_fraction,
            coalesce(resolution_date IS NOT NULL AND observed_ead>0,FALSE) AS resolved_with_ead,
            coalesce(candidate_for_pd AND {TARGET}=1 AND observed_ead>0 AND resolution_date IS NOT NULL
            AND ((split='train' AND resolution_date<DATE '2018-01-01')
              OR (split IN ('validation','calibration') AND resolution_date<DATE '2023-01-01')
              OR split='test'),FALSE) AS candidate_for_lgd
        FROM base''')


def build_quarter(root: Path, cohort: str, output: Path):
    source=root/'data/processed/full_population'/cohort
    receipt=json.loads((source/'manifest.json').read_text())
    if not receipt['quality']['structural_pass']: raise RuntimeError(f'{cohort} is not structurally valid')
    con=duckdb.connect()
    scratch=output/f'.scratch-{cohort}'
    try:
        configure(con,root,scratch)
        for name,kind in [('o','origination'),('p','performance')]:
            con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_path(source/kind/"*.parquet")})')
        build_views(con)
        for name,query in [('features','SELECT * FROM features'),('outcomes','SELECT * FROM outcomes'),('loss_outcomes','SELECT * FROM loss_outcomes')]:
            check_budget(root,extra=150_000_000)
            copy_query(con,query,output/name/f'{cohort}.parquet')
        count=con.execute('SELECT count(*),count(*) FILTER (WHERE candidate_for_pd) FROM outcomes').fetchone()
        print(f'MODELING {cohort}: {count[0]:,} loans; {count[1]:,} candidate labeled examples',flush=True)
    finally:
        con.close()
        if scratch.exists(): shutil.rmtree(scratch)


def finish(root: Path, stage: Path) -> dict:
    con=duckdb.connect();scratch=stage/'_scratch'
    try:
        configure(con,root,scratch)
        for view,folder in [('f','features'),('t','outcomes'),('l','loss_outcomes')]:
            con.execute(f'CREATE VIEW {view} AS SELECT * FROM read_parquet({sql_path(stage/folder/"*.parquet")})')
        con.execute('''CREATE TEMP VIEW blocked_refinance AS
            SELECT DISTINCT f.loan_identifier FROM f JOIN t current USING(loan_identifier)
            JOIN t prior ON f.refinance_predecessor_id=prior.loan_identifier
            WHERE current.split<>prior.split AND current.as_of_date>prior.as_of_date''')
        for split in ['train','validation','calibration','test']:
            query=f'''SELECT f.*,t.{TARGET} FROM f JOIN t USING(loan_identifier)
                      WHERE t.candidate_for_pd AND t.split='{split}'
                      AND NOT EXISTS (SELECT 1 FROM blocked_refinance b WHERE b.loan_identifier=f.loan_identifier)'''
            check_budget(root,extra=200_000_000)
            copy_query(con,query,stage/'pd'/f'{split}.parquet')
        copy_query(con,'SELECT * FROM blocked_refinance',stage/'excluded_cross_split_refinances.parquet')
        copy_query(con,'''SELECT l.*,NOT EXISTS (SELECT 1 FROM blocked_refinance b WHERE b.loan_identifier=l.loan_identifier)
                    AND l.candidate_for_lgd AS use_for_lgd FROM l''',stage/'lgd_ead_outcomes.parquet')
        counts=con.execute(f'''SELECT split,count(*) loans,count(*) FILTER (WHERE candidate_for_pd) candidate_labeled,
            count(*) FILTER (WHERE {TARGET}=1) positives,count(*) FILTER (WHERE {TARGET}=0) negatives,
            count(*) FILTER (WHERE {TARGET} IS NULL) unresolved_or_ineligible,
            min(as_of_date) first_anchor,max(as_of_date) last_anchor FROM t GROUP BY split ORDER BY split''').df()
        statuses=con.execute('SELECT split,outcome_status,count(*) loans FROM t GROUP BY split,outcome_status ORDER BY split,outcome_status').df()
        actual=[]
        for split in ['train','validation','calibration','test']:
            row=con.execute(f'''SELECT count(*) rows,count(DISTINCT loan_identifier) distinct_loans,
                    sum({TARGET}) positive,min(as_of_date) first_anchor,max(as_of_date) last_anchor
                    FROM read_parquet({sql_path(stage/'pd'/f'{split}.parquet')})''').df().iloc[0].to_dict()
            actual.append({'split':split,**row})
        actual=pd.DataFrame(actual)
        if (actual.rows==0).any(): raise RuntimeError('At least one required modeling split is empty.')
        if not actual.rows.eq(actual.distinct_loans).all(): raise RuntimeError('Duplicate model keys.')
        total,distinct=con.execute('SELECT count(*),count(DISTINCT loan_identifier) FROM t').fetchone()
        if total!=distinct: raise RuntimeError('Loans overlap across origin-quarter outputs.')
        checks=con.execute('''SELECT
            max(horizon_end) FILTER (WHERE split='train' AND candidate_for_pd) <
                min(as_of_date) FILTER (WHERE split='validation' AND candidate_for_pd) AS train_matures_before_validation,
            max(horizon_end) FILTER (WHERE split IN ('train','validation','calibration') AND candidate_for_pd) <
                min(as_of_date) FILTER (WHERE split='test' AND candidate_for_pd) AS development_matures_before_test,
            count(*) FILTER (WHERE candidate_for_pd AND observed_future_months<>12 AND serious_delinquency_12m=0)=0 AS negatives_have_complete_window,
            count(*) FILTER (WHERE candidate_for_pd AND serious_delinquency_12m=0 AND known_status_months<>12)=0 AS negatives_have_known_status
            FROM t''').df().iloc[0].to_dict()
        if not all(checks.values()): raise RuntimeError(f'Label/split invariants failed: {checks}')
        summaries=stage/'summaries';summaries.mkdir()
        counts.to_csv(summaries/'all_loans_by_split.csv',index=False)
        statuses.to_csv(summaries/'outcome_status_counts.csv',index=False)
        actual.to_csv(summaries/'pd_split_counts.csv',index=False)
        con.execute('''SELECT split,count(*) default_loans,count(*) FILTER (WHERE resolved_with_ead) resolved_with_ead,
            count(*) FILTER (WHERE candidate_for_lgd) candidate_for_lgd,
            count(*) FILTER (WHERE observed_loss_fraction<0) negative_observed_loss_fraction,
            count(*) FILTER (WHERE observed_loss_fraction>1) observed_loss_fraction_above_one
            FROM l GROUP BY split ORDER BY split''').df().to_csv(summaries/'lgd_availability.csv',index=False)
        blocked=con.execute('SELECT count(*) FROM blocked_refinance').fetchone()[0]
        contract={'dataset_version':'full-selected-vintages-v1','release':'Freddie Mac SFLLD Release 48',
            'origination_years':[2008,2018,2023],'performance_cutoff':CUTOFF,
            'key':'loan_identifier','target':TARGET,'feature_columns':FEATURES,
            'prediction_point':'first observed performance reporting month; currently performing, positive UPB, no termination, no flagged baseline/origination value violations',
            'outcome_window':'calendar months t+1 through t+12 inclusive',
            'event':'numeric delinquency >=3 months or RA, at or before the first observed termination in the window',
            'negative':'12 distinct future months, all statuses known, no observed event',
            'censoring':'early payoff/other exit, missing history/status and incomplete administrative follow-up retain NULL target; no zero imputation',
            'splits':{'train':'2008 vintage; horizon ends before 2018','validation':'2018 vintage; entry before July 2018',
                      'calibration':'remaining 2018 vintage entries with horizon ending before 2023','test':'2023 vintage with mature 12-month window'},
            'development_note':'2018 validation and calibration are two development subsets used retrospectively before the 2023 test. Their outcome windows can overlap. This is not a claim of deployment during 2018.',
            'refinance_rule':'exclude later loan from ready splits when its disclosed pre-HARP predecessor appears in another split',
            'excluded_known_cross_split_refinances':blocked,'invariants':{k:bool(v) for k,v in checks.items()},
            'all_loans':total,'pd_ready_rows':int(actual.rows.sum()),
            'lgd_note':'actual_loss / positive default-month UPB, or explicit immediately prior month UPB proxy; resolved 02/03/09 dispositions only. Negative or >1 fractions are retained. Selection on observed resolution is not ignorable.',
            'lgd_episode_note':'First observed serious delinquency can cure before a later credit disposition. The loss fraction is not an isolated single-default-episode LGD.',
            'availability_note':'historical corrected release; true real-time predictor/recovery publication availability is not established',
            'completed_utc':datetime.now(timezone.utc).isoformat(),
            'modeling_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (stage/'modeling_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
        return contract
    finally:
        con.close()
        if scratch.exists(): shutil.rmtree(scratch)


def run(root: Path):
    root=root.resolve();check_budget(root)
    full=root/'data/processed/full_population'
    summary=json.loads((full/'summary.json').read_text())
    if not summary['complete'] or summary['years']!=[2008,2018,2023]:
        raise RuntimeError('Complete 2008, 2018 and 2023 full vintages are required by this split contract.')
    output=root/'data/modeling'
    if output.exists(): raise RuntimeError('Existing data/modeling output is retained. Move it aside before an intentional rebuild.')
    stage=Path(tempfile.mkdtemp(prefix='.modeling-stage-',dir=root/'data'))
    try:
        for cohort in sorted(p.name for p in full.iterdir() if p.is_dir() and not p.name.startswith('.')):
            build_quarter(root,cohort,stage)
        result=finish(root,stage)
        check_budget(root)
        stage.rename(output)
        print(f'Modeling handoff complete: {result["pd_ready_rows"]:,} labeled rows; data {data_bytes(root)/1e9:.3f} GB',flush=True)
        return result
    finally:
        if stage.exists(): shutil.rmtree(stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd())
    run(parser.parse_args().root)
