"""Resumable full-vintage ingestion. Annual ZIPs contain four quarterly ZIPs.

No text extraction, row sampling, imputation, or model fitting. Each completed
quarter is independently validated and can be reused after an interrupted run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import duckdb
import numpy as np
import pandas as pd

from .freddie_cleaning import ORIG_COLUMNS, PERF_COLUMNS, clean_origination, clean_performance

MAX_BYTES = 10_000_000_000


def data_bytes(root: Path) -> int:
    seen, total = set(), 0
    for path in (root / 'data').rglob('*'):
        if path.is_file():
            st = path.stat()
            key = st.st_dev, st.st_ino
            if key not in seen:
                total += st.st_size
                seen.add(key)
    return total


def check_budget(root: Path, extra: int = 0) -> None:
    # 0.5 GB headroom covers two concurrent small shard writes and metadata.
    if data_bytes(root) + extra > MAX_BYTES - 500_000_000:
        raise RuntimeError('Data would exceed the 9.5 GB working limit; completed quarters were retained.')
    if shutil.disk_usage(root).free - extra < 2_000_000_000:
        raise RuntimeError('Less than 2 GB of free disk would remain; completed quarters were retained.')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def code_fingerprint() -> str:
    here = Path(__file__)
    return hashlib.sha256(here.read_bytes() + (here.parent / 'freddie_cleaning.py').read_bytes()).hexdigest()


class WidthCheckedReader(io.RawIOBase):
    """Validate every physical row while allowing pandas' native CSV parser.

    Freddie's format has no quoting/escaped pipes. Checks include short records,
    empty lines, and the final unterminated line; none are silently padded.
    """
    def __init__(self, source, width: int, label: str):
        self.source, self.width, self.label = source, width, label
        self.pending, self.line, self.eof = b'', 0, False

    def readable(self):
        return True

    def _check(self, row: bytes):
        self.line += 1
        count = row.count(b'|') + 1
        if count != self.width:
            raise ValueError(f'Schema mismatch {self.label}, line {self.line}: expected {self.width} fields, got {count}')

    def readinto(self, buffer):
        if self.eof:
            return 0
        block = self.source.read(len(buffer))
        if not block:
            if self.pending:
                self._check(self.pending)
            self.pending = b''
            self.eof = True
            return 0
        pieces = (self.pending + block).split(b'\n')
        self.pending = pieces.pop()
        for row in pieces:
            self._check(row.rstrip(b'\r'))
        buffer[:len(block)] = block
        return len(block)


@contextmanager
def open_quarter(annual: Path, inner_member: str):
    with zipfile.ZipFile(annual) as outer, outer.open(inner_member) as handle, zipfile.ZipFile(handle) as inner:
        yield inner


def iter_member(inner: zipfile.ZipFile, member: str, kind: str, chunksize=100_000):
    columns = ORIG_COLUMNS if kind == 'origination' else PERF_COLUMNS
    with inner.open(member) as source:
        stream = io.BufferedReader(WidthCheckedReader(source, len(columns), member))
        with pd.read_csv(stream, sep='|', names=columns, header=None, dtype='string',
                         na_filter=False, skip_blank_lines=False, quoting=3,
                         encoding='utf-8-sig', on_bad_lines='error', chunksize=chunksize) as reader:
            offset = 1
            for chunk in reader:
                chunk = chunk.reset_index(drop=True)
                chunk.attrs['first_line'] = offset
                offset += len(chunk)
                yield chunk


def inventory(root: Path, years: list[int]) -> list[dict]:
    jobs = []
    for year in years:
        path = root / 'data' / 'raw_full' / f'historical_data_{year}.zip'
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        with zipfile.ZipFile(path) as outer:
            expected = {f'historical_data_{year}Q{q}.zip' for q in range(1, 5)}
            actual = {i.filename for i in outer.infolist() if not i.is_dir()}
            if actual != expected:
                raise ValueError(f'{path.name}: expected all four quarterly ZIPs, got {sorted(actual)}')
            for q in range(1, 5):
                inner_member = f'historical_data_{year}Q{q}.zip'
                with open_quarter(path, inner_member) as z:
                    members = {i.filename for i in z.infolist() if not i.is_dir()}
                    required = {f'orig_{year}Q{q}.txt', f'perf_{year}Q{q}.txt'}
                    if members != required:
                        raise ValueError(f'{inner_member}: unexpected schema members {members}')
                    sizes = {i.filename: i.file_size for i in z.infolist()}
                jobs.append(dict(year=year, quarter=q, cohort=f'{year}Q{q}', archive=str(path.relative_to(root)),
                                 archive_bytes=path.stat().st_size, archive_sha256=digest,
                                 inner_member=inner_member, uncompressed_bytes=sizes,
                                 code_sha256=code_fingerprint()))
    return jobs


def sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def summarize_quarter(root: Path, stage: Path) -> dict:
    scratch = stage / '_scratch'
    scratch.mkdir()
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET memory_limit='768MB'")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f'SET temp_directory={sql_path(scratch)}')
        spill = min(1_500_000_000, MAX_BYTES - data_bytes(root) - 500_000_000,
                    shutil.disk_usage(root).free - 2_000_000_000)
        if spill <= 0:
            raise RuntimeError('Insufficient scratch capacity within storage cap.')
        con.execute(f"SET max_temp_directory_size='{spill}B'")
        for kind in ('origination', 'performance'):
            con.execute(f'CREATE VIEW {kind} AS SELECT * FROM read_parquet({sql_path(stage / kind / "*.parquet")})')
        quality = {}
        for table, key in [('origination', 'loan_identifier'), ('performance', 'loan_identifier, reporting_period')]:
            quality[f'{table}_rows'], quality[f'{table}_loans'] = con.execute(
                f'SELECT count(*),count(DISTINCT loan_identifier) FROM {table}').fetchone()
            quality[f'{table}_duplicate_keys'] = con.execute(
                f'SELECT count(*) FROM (SELECT {key} FROM {table} GROUP BY {key} HAVING count(*)>1)').fetchone()[0]
            predicate = 'loan_identifier IS NULL' + (' OR reporting_period IS NULL' if table == 'performance' else '')
            quality[f'{table}_invalid_keys'] = con.execute(f'SELECT count(*) FROM {table} WHERE {predicate}').fetchone()[0]
            cols = ORIG_COLUMNS if table == 'origination' else PERF_COLUMNS
            counts = con.execute('SELECT ' + ','.join(f'count(*)-count("{c}")' for c in cols) + f' FROM {table}').fetchone()
            pd.DataFrame({'field':cols, 'missing_count':counts, 'rows':quality[f'{table}_rows']}).to_csv(stage / f'{table}_missingness.csv', index=False)
        for name, left, right in [('unmatched_performance_rows','performance','origination'), ('unmatched_origination_rows','origination','performance')]:
            quality[name] = con.execute(f'SELECT count(*) FROM {left} l WHERE NOT EXISTS (SELECT 1 FROM {right} r WHERE l.loan_identifier=r.loan_identifier)').fetchone()[0]
        coverage = con.execute('''SELECT loan_identifier,min(reporting_period) first_period,
            max(reporting_period) last_period,count(DISTINCT reporting_period) observed_months,
            date_diff('month',min(reporting_period),max(reporting_period))+1-count(DISTINCT reporting_period) gap_months,
            count(*) FILTER (WHERE delinquency_status IS NULL) unknown_status_months,
            bool_or(serious_delinquency) observed_serious_delinquency
            FROM performance GROUP BY loan_identifier''').df()
        coverage.to_parquet(stage / 'coverage.parquet', index=False, compression='zstd')
        quality['loans_with_internal_gaps'] = int(coverage.gap_months.gt(0).sum())
        quality['internal_gap_months'] = int(coverage.gap_months.sum())
        quality['first_reporting_period'] = str(coverage.first_period.min().date())
        quality['last_reporting_period'] = str(coverage.last_period.max().date())
        quality['structural_pass'] = all(quality[k] == 0 for k in quality if any(x in k for x in ('duplicate_keys','invalid_keys','unmatched_')))
        return quality
    finally:
        con.close()
        shutil.rmtree(scratch)


def process_quarter(root_string: str, job: dict, chunksize: int) -> dict:
    root = Path(root_string)
    output = root / 'data' / 'processed' / 'full_population' / job['cohort']
    receipt = output / 'manifest.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if all(previous.get(k) == job.get(k) for k in ('archive_sha256','inner_member','code_sha256')):
            for item in previous['parquet_files']:
                file = output / item['path']
                if not file.is_file() or file.stat().st_size != item['bytes']:
                    raise RuntimeError(f'Completed output is missing/changed: {file}')
            print(f"REUSE {job['cohort']}", flush=True)
            return previous
        raise RuntimeError(f'{output}: existing output uses different data/code; preserve it before rebuilding.')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".stage-{job['cohort']}-", dir=output.parent))
    audit = defaultdict(int)
    counts = {}
    try:
        print(f"START {job['cohort']}", flush=True)
        with open_quarter(root / job['archive'], job['inner_member']) as inner:
            for kind, prefix, cleaner in [('origination','orig',clean_origination),('performance','perf',clean_performance)]:
                member = f"{prefix}_{job['cohort']}.txt"
                target = stage / kind
                target.mkdir()
                total = 0
                for part, raw in enumerate(iter_member(inner, member, kind, chunksize)):
                    check_budget(root, extra=150_000_000)
                    clean, field_audit = cleaner(raw)
                    if not clean.origination_quarter.eq(job['cohort']).fillna(False).all():
                        raise ValueError(f'{member}: invalid loan ID or ID belongs to a different quarter.')
                    clean['source_row'] = np.arange(total + 1, total + 1 + len(clean), dtype='int64')
                    clean['source_member'] = member
                    # Archive and nested path are constant in the quarter manifest.
                    clean.to_parquet(target / f'part-{part:06d}.parquet', index=False, compression='zstd')
                    for row in field_audit.to_dict('records'):
                        for metric, value in row.items():
                            if metric.endswith('_count'):
                                audit[(kind,row['field'],metric)] += int(value)
                    total += len(clean)
                    if (part+1) % 10 == 0:
                        print(f"PROGRESS {job['cohort']} {kind} {total:,}", flush=True)
                    check_budget(root)
                if total == 0:
                    raise ValueError(f'{member} is empty')
                counts[kind] = total
        print(f"VALIDATE {job['cohort']} {counts}", flush=True)
        quality = summarize_quarter(root, stage)
        if not quality['structural_pass']:
            (stage / 'quality_failure.json').write_text(json.dumps(quality, indent=2))
            raise ValueError(f"Structural validation failed for {job['cohort']}: {quality}")
        audit_frame = pd.DataFrame([{'table':k,'field':c,'issue':m,'count':n} for (k,c,m),n in sorted(audit.items())])
        audit_frame.to_csv(stage / 'cleaning_audit.csv', index=False)
        quality['value_violation_count'] = int(audit_frame.loc[audit_frame.issue.isin(['malformed_count','range_violation_count','code_violation_count']),'count'].sum())
        result = dict(job, quality=quality, complete=True, release='48',
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      parquet_files=[{'path':str(p.relative_to(stage)), 'bytes':p.stat().st_size} for p in sorted(stage.rglob('*.parquet'))])
        (stage / 'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        check_budget(root)
        stage.rename(output)
        print(f"DONE {job['cohort']}: {counts['origination']:,} loans; {counts['performance']:,} months; data {data_bytes(root)/1e9:.3f} GB", flush=True)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def run(root: Path, years=(2008,2018,2023), workers=2, chunksize=100_000):
    root = root.resolve()
    check_budget(root)
    jobs = inventory(root, list(years))
    dest = root / 'data' / 'processed' / 'full_population'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'input_inventory.json').write_text(json.dumps(jobs,indent=2)+'\n')
    if workers == 1:
        receipts = [process_quarter(str(root),j,chunksize) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(process_quarter,str(root),j,chunksize) for j in jobs]
            receipts = [f.result() for f in as_completed(pending)]
    receipts.sort(key=lambda r:r['cohort'])
    summary = {'complete':True, 'dataset':'Standard full population in selected origination years',
               'years':list(years), 'quarters':len(receipts), 'origination_rows':sum(r['quality']['origination_rows'] for r in receipts),
               'performance_rows':sum(r['quality']['performance_rows'] for r in receipts),
               'data_bytes':data_bytes(root), 'max_bytes':MAX_BYTES,
               'code_sha256':code_fingerprint(), 'completed_utc':datetime.now(timezone.utc).isoformat()}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pd.DataFrame([{'cohort':r['cohort'],**r['quality']} for r in receipts]).to_csv(dest/'quality_by_quarter.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--years',type=int,nargs='+',default=[2008,2018,2023])
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--chunksize',type=int,default=100_000)
    args=parser.parse_args()
    run(args.root,args.years,args.workers,args.chunksize)
