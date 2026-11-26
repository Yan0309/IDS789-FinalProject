"""Small synthetic format fixtures, never used as modeling observations."""
import io
import json
import zipfile
import pandas as pd
import pytest
from src.freddie_cleaning import ORIG_COLUMNS, PERF_COLUMNS
from src.full_population import WidthCheckedReader, iter_member, inventory, process_quarter


def rows(year=2008,quarter=1):
    loan=f'F{year%100:02d}Q{quarter}0000001'
    o=dict.fromkeys(ORIG_COLUMNS,'')
    o.update(loan_identifier=loan,classic_fico='720',first_payment_date=f'{year}02',maturity_date=f'{year+30}01',
             original_upb='200000',original_ltv='80',original_cltv='80',original_dti='35',number_of_units='1',
             number_of_borrowers='2',original_interest_rate='6',original_loan_term='360',harp_indicator='N')
    p=dict.fromkeys(PERF_COLUMNS,'')
    p.update(loan_identifier=loan,reporting_period=f'{year}02',current_actual_upb='199900',delinquency_status='0')
    return o,p


def annual(root,year=2008,duplicate=False,mismatched_cohort=False):
    directory=root/'data/raw_full';directory.mkdir(parents=True)
    path=directory/f'historical_data_{year}.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as out:
        for q in range(1,5):
            o,p=rows(year,q)
            if mismatched_cohort:
                o['loan_identifier']=p['loan_identifier']=f'F09Q{q}0000001'
            b=io.BytesIO()
            with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr(f'orig_{year}Q{q}.txt','|'.join(o[c] for c in ORIG_COLUMNS)+'\n')
                row='|'.join(p[c] for c in PERF_COLUMNS)+'\n'
                z.writestr(f'perf_{year}Q{q}.txt',row*(2 if duplicate else 1))
            out.writestr(f'historical_data_{year}Q{q}.zip',b.getvalue())
    return path


@pytest.mark.parametrize('payload',[b'a|b\nshort\n',b'a|b|c\n',b'a|b\n\n',b'a|b\nshort'])
def test_width_reader_rejects_short_long_blank_and_unterminated_bad_rows(payload):
    stream=io.BufferedReader(WidthCheckedReader(io.BytesIO(payload),2,'test'))
    with pytest.raises(ValueError,match='Schema mismatch'):
        while stream.read(3): pass


def test_width_reader_keeps_crlf_and_unterminated_valid_line():
    payload=b'a|b\r\nc|d'
    stream=io.BufferedReader(WidthCheckedReader(io.BytesIO(payload),2,'test'))
    assert stream.read()==payload


def test_native_parser_preserves_empty_and_quoted_literal_values():
    o,_=rows();o['seller_name']='"A seller"';o['postal_code']='001'
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z: z.writestr('orig_2008Q1.txt','|'.join(o[c] for c in ORIG_COLUMNS))
    with zipfile.ZipFile(io.BytesIO(b.getvalue())) as z:
        frame=next(iter_member(z,'orig_2008Q1.txt','origination',1))
    assert frame.loc[0,'seller_name']=='"A seller"'
    assert frame.loc[0,'postal_code']=='001'
    assert frame.loc[0,'vantagescore_4']==''


def test_nested_complete_quarter_is_validated_and_resumable(tmp_path):
    annual(tmp_path)
    jobs=inventory(tmp_path,[2008]);assert len(jobs)==4
    result=process_quarter(str(tmp_path),jobs[0],1)
    assert result['quality']['structural_pass']
    assert result['quality']['origination_rows']==1
    assert result['quality']['performance_rows']==1
    assert process_quarter(str(tmp_path),jobs[0],1)==result
    output=tmp_path/'data/processed/full_population/2008Q1'
    file=output/'performance/part-000000.parquet'
    p=pd.read_parquet(file)
    assert p.source_row.tolist()==[1]
    assert '_raw_record_sha256' not in p
    file.unlink()
    with pytest.raises(RuntimeError,match='missing/changed'):
        process_quarter(str(tmp_path),jobs[0],1)


def test_duplicate_month_prevents_publishing(tmp_path):
    annual(tmp_path,duplicate=True)
    job=inventory(tmp_path,[2008])[0]
    with pytest.raises(ValueError,match='Structural validation failed'):
        process_quarter(str(tmp_path),job,1)
    assert not (tmp_path/'data/processed/full_population/2008Q1').exists()


def test_source_cohort_mismatch_is_retained_and_reported(tmp_path):
    annual(tmp_path,mismatched_cohort=True)
    result=process_quarter(str(tmp_path),inventory(tmp_path,[2008])[0],100)
    assert result['quality']['structural_pass']
    assert result['quality']['origination_identifier_cohort_mismatch_rows']==1
    assert result['quality']['performance_identifier_cohort_mismatch_rows']==1
