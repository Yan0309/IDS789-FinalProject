"""Synthetic label-boundary tests; these records are never research data."""
import duckdb
import pandas as pd
import pytest

from src.freddie_cleaning import ORIG_COLUMNS, PERF_COLUMNS, clean_origination, clean_performance
from src.modeling_data import FEATURES, TARGET, build_views, finish


def tables(cases):
    orig, perf = [], []
    for number, case in enumerate(cases, 1):
        year = case.get('year', 2018)
        start = pd.Timestamp(case.get('start', f'{year}-01-01'))
        key = case.get('key', f'F{year % 100:02}Q1{number:07}')
        row = dict.fromkeys(ORIG_COLUMNS, '')
        row.update(loan_identifier=key, classic_fico='720', first_payment_date=start.strftime('%Y%m'),
                   original_upb='100000', original_ltv='80', original_cltv='80', original_dti='35',
                   pre_harp_loan_sequence_number=case.get('predecessor', ''))
        orig.append(row)
        for month in case.get('months', range(13)):
            row = dict.fromkeys(PERF_COLUMNS, '')
            row.update(loan_identifier=key, reporting_period=(start + pd.DateOffset(months=month)).strftime('%Y%m'),
                       delinquency_status='0', current_actual_upb='100000', current_interest_rate='5')
            row.update(case.get('updates', {}).get(month, {}))
            perf.append(row)
    o = clean_origination(pd.DataFrame(orig, columns=ORIG_COLUMNS, dtype='string'))[0]
    p = clean_performance(pd.DataFrame(perf, columns=PERF_COLUMNS, dtype='string'))[0]
    return o, p


@pytest.fixture
def db():
    con = duckdb.connect()
    yield con
    con.close()


def load(db, cases):
    o, p = tables(cases)
    db.register('o', o); db.register('p', p)
    build_views(db)
    return db.execute('SELECT * FROM outcomes ORDER BY loan_identifier').df()


@pytest.mark.parametrize('case,label,status', [
    ({}, 0, 'complete_non_event'),
    ({'months': range(4), 'updates': {3: {'delinquency_status': '3'}}}, 1, 'observed_event'),
    ({'months': range(6), 'updates': {5: {'zero_balance_code': '01'}}}, None, 'voluntary_payoff'),
    ({'months': [m for m in range(13) if m != 6]}, None, 'incomplete_history'),
    ({'updates': {6: {'delinquency_status': 'XX'}}}, None, 'unknown_status'),
    ({'updates': {0: {'delinquency_status': '1'}}}, None, 'not_current_at_entry'),
    ({'updates': {3: {'delinquency_status': 'RA'}}}, 1, 'observed_event'),
    ({'updates': {5: {'zero_balance_code': '01'}, 6: {'delinquency_status': '4'}}}, None, 'voluntary_payoff'),
    ({'updates': {12: {'zero_balance_code': '01'}}}, 0, 'complete_non_event'),
    ({'months': range(14), 'updates': {13: {'delinquency_status': '3'}}}, 0, 'complete_non_event'),
    ({'updates': {1: {'delinquency_status': 'XX'}, 2: {'delinquency_status': '3'}}}, 1, 'observed_event'),
])
def test_calendar_window_and_censoring(db, case, label, status):
    outcome = load(db, [case]).iloc[0]
    assert (pd.isna(outcome[TARGET]) if label is None else outcome[TARGET] == label)
    assert outcome.outcome_status == status


def test_loss_sign_and_previous_calendar_month_proxy(db):
    load(db, [{'months': range(4), 'updates': {3: {
        'delinquency_status': 'RA', 'current_actual_upb': '0', 'zero_balance_code': '09',
        'actual_loss': '-5000', 'mi_recoveries': '-20000'}}}])
    loss = db.execute('SELECT * FROM loss_outcomes').df().iloc[0]
    assert loss.observed_ead == 100000
    assert loss.ead_basis == 'previous_month_upb_proxy'
    assert loss.observed_loss_fraction == -0.05
    assert loss.mi_recoveries == -20000
    assert loss.candidate_for_lgd


def test_no_missing_loss_or_exposure_imputation(db):
    load(db, [{'months': [0, 3], 'updates': {3: {'delinquency_status': '3', 'current_actual_upb': '0'}}}])
    loss = db.execute('SELECT * FROM loss_outcomes').df().iloc[0]
    assert pd.isna(loss.observed_ead)
    assert pd.isna(loss.actual_loss)
    assert pd.isna(loss.observed_loss_fraction)
    assert loss.ead_basis == 'unavailable'


def test_split_boundaries_and_future_columns_excluded(db):
    outcomes = load(db, [
        {'year': 2008}, {'year': 2018, 'start': '2018-06-01'},
        {'year': 2018, 'start': '2018-07-01'}, {'year': 2023}])
    assert outcomes.split.tolist() == ['train', 'validation', 'calibration', 'test']
    assert not set(FEATURES).intersection({'loan_identifier','as_of_date','refinance_predecessor_id',
        TARGET,'actual_loss','event_date_12m','observed_ead','resolution_date','delinquency_status'})


def test_handoff_blocks_cross_split_refinance_and_checks_temporal_order(db, tmp_path):
    load(db, [{'year': 2008}, {'year': 2018}, {'year': 2018, 'start': '2018-07-01'},
              {'year': 2023}, {'year': 2023, 'predecessor': 'F08Q10000001'}])
    stage = tmp_path/'data'/'stage'
    for name in ['features', 'outcomes', 'loss_outcomes']:
        folder = stage/name; folder.mkdir(parents=True)
        db.execute(f"COPY (SELECT * FROM {name}) TO '{folder}/fixture.parquet' (FORMAT PARQUET)")
    result = finish(tmp_path, stage)
    assert result['all_loans'] == 5
    assert result['excluded_known_cross_split_refinances'] == 1
    assert result['pd_ready_rows'] == 4
    assert all(result['invariants'].values())
