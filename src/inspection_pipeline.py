"""Bounded, repeatable inspection of the Freddie Mac sample ZIP files.

The pipeline reads ZIP members directly, writes cleaned Parquet shards, and uses
DuckDB for cross-file checks.  It never builds prediction targets or fits models.
Generated output is staged separately so a failed run leaves the last completed
inspection intact.  ``data/processed/freddie_inspection`` is owned by this module.
Persistent output writes stop at 95% of the total data budget. DuckDB may use up
to 2 GB of scratch within the remaining total budget and the 2 GB free-disk
reserve; distinct-record checks can spill beyond a fixed 5% scratch allowance.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import duckdb
import numpy as np
import pandas as pd

try:
    from .freddie_cleaning import (
        clean_origination, clean_performance, inventory_archives, iter_raw_chunks,
    )
except ImportError:  # Also support notebooks that add project_root/src to sys.path.
    from freddie_cleaning import (
        clean_origination, clean_performance, inventory_archives, iter_raw_chunks,
    )


OUTPUT_NAME = "freddie_inspection"
MIN_FREE_BYTES = 2_000_000_000
PROVENANCE = {"source_archive", "source_member", "source_row", "source_cohort"}
INTERNAL_COLUMNS = PROVENANCE | {"_raw_record_sha256"}
TABLE_COLUMNS = {
    "inventory": ["archive", "member", "kind", "compressed_bytes", "uncompressed_bytes", "archive_bytes", "archive_sha256"],
    "overview": ["kind", "rows", "columns", "distinct_loans"],
    "cleaning_audit": ["kind", "column", "issue", "count"],
    "missingness": ["kind", "column", "missing_count", "row_count", "missing_fraction"],
    "origination": [],
    "origination_describe": ["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    "category_distributions": ["column", "value", "count", "fraction"],
    "month_counts": ["reporting_period", "rows", "distinct_loans"],
    "loan_coverage": ["loan_identifier", "first_period", "last_period", "distinct_months", "span_months", "gap_months", "record_count", "min_delinquency", "max_delinquency", "observed_serious_delinquency"],
    "cohort_summary": ["cohort", "loans", "performance_rows", "linked_loans", "median_observed_months", "observed_serious_delinquency_loans"],
    "linkage_quality": ["metric", "count"],
    "duplicate_quality": ["kind", "total_rows", "exact_duplicate_excess_rows", "key_duplicate_excess_rows", "conflicting_keys", "invalid_key_rows"],
    "loss_field_availability": ["column", "nonmissing_count", "missing_count", "nonzero_count"],
    "balance_summary": ["column", "count", "mean", "min", "median", "max", "sum"],
}


class StorageBudgetError(RuntimeError):
    """The inspection cannot continue within its on-disk resource budget."""


def _empty_result() -> dict[str, Any]:
    result: dict[str, Any] = {key: pd.DataFrame(columns=columns) for key, columns in TABLE_COLUMNS.items()}
    result.update(status="awaiting_data", modeling_ready=False, structural_quality_pass=False,
                  warnings=["No raw sample ZIP archives were found; no synthetic observations were substituted."], paths={})
    return result


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _data_bytes(root: Path) -> int:
    """Count each physical file once, including generated and scratch files."""
    total = 0
    seen: set[tuple[int, int]] = set()
    for directory in (root / "data",):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue  # A concurrent worker can remove a scratch file.
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    total += stat.st_size
                    seen.add(identity)
    return total


def _check_budget(root: Path, max_bytes: int, *, reserve: bool = True, projected: int = 0) -> int:
    used = _data_bytes(root)
    limit = int(max_bytes * .95) if reserve else max_bytes
    if used + projected > limit:
        raise StorageBudgetError(
            f"Data footprint would exceed {'95% working' if reserve else 'total'} budget: "
            f"{used + projected:,} > {limit:,} bytes. Existing output has been retained."
        )
    free = shutil.disk_usage(root).free
    if free - projected < MIN_FREE_BYTES:
        raise StorageBudgetError(f"Need at least {MIN_FREE_BYTES:,} free disk bytes; found {free:,} before the proposed write.")
    return used


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_inventory(inventory: pd.DataFrame, previous: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Hash each archive once; unchanged size/mtime can reuse the prior manifest."""
    cache: dict[str, dict[str, Any]] = {}
    if previous.is_file():
        try:
            cache = {row["archive"]: row for row in json.loads(previous.read_text())}
        except (OSError, ValueError, KeyError, TypeError):
            cache = {}
    records = []
    info = {}
    for archive in inventory["archive"].drop_duplicates():
        path = Path(archive).resolve()
        stat = path.stat()
        old = cache.get(str(path), {})
        same = old.get("archive_bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns
        digest = old.get("archive_sha256") if same else None
        if not digest:
            digest = _sha256(path)
        row = dict(archive=str(path), archive_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, archive_sha256=digest)
        records.append(row)
        info[str(path)] = row
    inventory = inventory.copy()
    inventory["archive"] = inventory["archive"].map(lambda value: str(Path(value).resolve()))
    for column in ("archive_bytes", "archive_sha256"):
        inventory[column] = inventory["archive"].map(lambda value: info[value][column])
    return inventory, records


def _cohort(row: pd.Series) -> str:
    for name in ("cohort_year", "cohort", "year", "vintage"):
        if name in row and pd.notna(row[name]):
            value = row[name]
            return str(int(value)) if isinstance(value, (float, np.floating)) else str(value)
    match = re.search(r"(?:19|20)\d{2}", Path(row["archive"]).name)
    return match.group(0) if match else Path(row["archive"]).stem


def _raw_hashes(raw: pd.DataFrame) -> list[bytes]:
    """Fingerprint complete original rows without per-field Python hashing.

    The strict ZIP parser supplies strings, including empty strings. JSON array
    serialization preserves field boundaries and the original string values.
    Binary SHA-256 digests halve the stored/hash-aggregation width versus hex.
    A frame-level missing check enables a fallback for direct callers with NA.
    """
    has_missing = raw.isna().to_numpy().any()
    hashes = []
    for values in raw.itertuples(index=False, name=None):
        if has_missing:
            values = [None if pd.isna(value) else str(value) for value in values]
        serialized = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        hashes.append(hashlib.sha256(serialized.encode("utf-8")).digest())
    return hashes


def _audit_counts(audit: pd.DataFrame, kind: str, counts: dict[tuple[str, str, str], int]) -> None:
    if audit is None or audit.empty:
        return
    for row in audit.to_dict("records"):
        column = str(row.get("column", row.get("field", "__row__")))
        if "field" in row and "raw_rows" in row:
            for metric, count in row.items():
                if metric.endswith("_count") and int(count):
                    counts[(kind, column, metric.removesuffix("_count"))] += int(count)
            continue
        issue = str(row.get("issue", row.get("reason", row.get("rule", "unspecified"))))
        count = row.get("count", row.get("n", row.get("affected_rows", 0)))
        counts[(kind, column, issue)] += int(count)


def _columns(connection: duckdb.DuckDBPyConnection, view: str) -> dict[str, str]:
    return {row[0]: row[1] for row in connection.execute(f"DESCRIBE {_q(view)}").fetchall()}


def _valid_loan(column: str = "loan_identifier") -> str:
    return f"({_q(column)} IS NOT NULL AND regexp_full_match(CAST({_q(column)} AS VARCHAR), '[FA][0-9]{{2}}Q[1-4][0-9]{{7}}'))"


def _summarize(connection: duckdb.DuckDBPyConnection, available: set[str]) -> dict[str, pd.DataFrame]:
    tables = {key: pd.DataFrame(columns=columns) for key, columns in TABLE_COLUMNS.items() if key != "inventory"}
    schemas = {kind: _columns(connection, kind) for kind in available}
    overview, missingness, duplicates = [], [], []
    for kind in sorted(available):
        columns = [name for name in schemas[kind] if name not in INTERNAL_COLUMNS]
        loan = "loan_identifier" in columns
        total, unique = connection.execute(
            f"SELECT count(*), {('count(DISTINCT loan_identifier)' if loan else '0')} FROM {_q(kind)}"
        ).fetchone()
        overview.append(dict(kind=kind, rows=total, columns=len(columns), distinct_loans=unique))
        if columns:
            nulls = connection.execute("SELECT " + ",".join(f"count(*)-count({_q(c)})" for c in columns) + f" FROM {_q(kind)}").fetchone()
            missingness.extend(dict(kind=kind, column=c, missing_count=n, row_count=total,
                                    missing_fraction=n / total if total else np.nan) for c, n in zip(columns, nulls))
        valid = _valid_loan() if loan else "FALSE"
        key = ["loan_identifier"] if loan else []
        if kind == "performance" and "reporting_period" in columns:
            key.append("reporting_period")
            valid += ' AND "reporting_period" IS NOT NULL'
        invalid = connection.execute(f"SELECT count(*) FROM {_q(kind)} WHERE NOT ({valid})").fetchone()[0]
        exact = connection.execute(f"SELECT count(*)-count(DISTINCT _raw_record_sha256) FROM {_q(kind)}").fetchone()[0]
        duplicate_excess = conflicts = 0
        if key:
            key_expr = ",".join(_q(c) for c in key)
            duplicate_excess, conflicts = connection.execute(
                f"SELECT coalesce(sum(n-1),0), count(*) FILTER (WHERE versions > 1) FROM "
                f"(SELECT count(*) n,count(DISTINCT _raw_record_sha256) versions FROM {_q(kind)} "
                f"WHERE {valid} GROUP BY {key_expr} HAVING count(*)>1)"
            ).fetchone()
        duplicates.append(dict(kind=kind, total_rows=total, exact_duplicate_excess_rows=exact,
                               key_duplicate_excess_rows=duplicate_excess, conflicting_keys=conflicts, invalid_key_rows=invalid))
    tables["overview"] = pd.DataFrame(overview, columns=TABLE_COLUMNS["overview"])
    tables["missingness"] = pd.DataFrame(missingness, columns=TABLE_COLUMNS["missingness"])
    tables["duplicate_quality"] = pd.DataFrame(duplicates, columns=TABLE_COLUMNS["duplicate_quality"])

    if "origination" in available:
        tables["origination"] = connection.execute("SELECT * EXCLUDE (_raw_record_sha256) FROM origination").df()
        orig = tables["origination"]
        numeric = [c for c in orig.select_dtypes(include="number").columns if c not in INTERNAL_COLUMNS]
        if numeric:
            tables["origination_describe"] = orig[numeric].describe().T.rename_axis("column").reset_index()
        categories = []
        for column, dtype in schemas["origination"].items():
            if column in INTERNAL_COLUMNS or "identifier" in column or "date" in column or "period" in column:
                continue
            if dtype.startswith("VARCHAR") or dtype == "BOOLEAN":
                frequencies = orig[column].astype("string").fillna("<missing>").value_counts(dropna=False)
                # Bounded preview; each displayed share uses all origination rows.
                for value, count in frequencies.head(100).items():
                    categories.append(dict(column=column, value=str(value), count=int(count), fraction=count / len(orig) if len(orig) else np.nan))
        tables["category_distributions"] = pd.DataFrame(categories, columns=TABLE_COLUMNS["category_distributions"])

    if "performance" in available:
        cols = schemas["performance"]
        if "reporting_period" in cols:
            tables["month_counts"] = connection.execute(
                "SELECT reporting_period, count(*) AS rows,count(DISTINCT loan_identifier) AS distinct_loans "
                "FROM performance WHERE reporting_period IS NOT NULL GROUP BY reporting_period ORDER BY reporting_period"
            ).df()
            delinquency_column = next((c for c in ("delinquency_months", "delinquency_status", "current_loan_delinquency_status") if c in cols), None)
            delinquency = f"TRY_CAST({_q(delinquency_column)} AS DOUBLE)" if delinquency_column else "NULL::DOUBLE"
            serious = '"serious_delinquency"' if "serious_delinquency" in cols else f"({delinquency}>=3)"
            connection.execute(
                "CREATE TEMP VIEW coverage AS SELECT loan_identifier,min(reporting_period) first_period,"
                "max(reporting_period) last_period,count(DISTINCT reporting_period) distinct_months,"
                "date_diff('month',min(reporting_period),max(reporting_period))+1 span_months,"
                "date_diff('month',min(reporting_period),max(reporting_period))+1-count(DISTINCT reporting_period) gap_months,"
                f"count(*) record_count,min({delinquency}) min_delinquency,max({delinquency}) max_delinquency,"
                f"bool_or({serious}) observed_serious_delinquency "
                f"FROM performance WHERE {_valid_loan()} GROUP BY loan_identifier"
            )
            tables["loan_coverage"] = connection.execute("SELECT * FROM coverage ORDER BY loan_identifier").df()
        loss_columns = [c for c in cols if any(token in c for token in ("loss", "recover", "proceed", "expense", "costs", "maintenance", "taxes_and_insurance", "accrued_interest")) and c not in INTERNAL_COLUMNS]
        loss = []
        for column in loss_columns:
            values = connection.execute(
                f"SELECT count({_q(column)}),count(*)-count({_q(column)}),count(*) FILTER (WHERE TRY_CAST({_q(column)} AS DOUBLE)<>0) FROM performance"
            ).fetchone()
            loss.append(dict(zip(TABLE_COLUMNS["loss_field_availability"], [column, *values])))
        tables["loss_field_availability"] = pd.DataFrame(loss, columns=TABLE_COLUMNS["loss_field_availability"])

    balances = []
    for kind, cols in schemas.items():
        for column in cols:
            if ("upb" in column or "balance" in column) and column not in INTERNAL_COLUMNS and not any(word in column for word in ("code", "date", "flag")):
                expr = f"TRY_CAST({_q(column)} AS DOUBLE)"
                values = connection.execute(f"SELECT count({expr}),avg({expr}),min({expr}),median({expr}),max({expr}),sum({expr}) FROM {_q(kind)}").fetchone()
                balances.append(dict(zip(TABLE_COLUMNS["balance_summary"], [column, *values])))
    tables["balance_summary"] = pd.DataFrame(balances, columns=TABLE_COLUMNS["balance_summary"])

    linkage = []
    if {"origination", "performance"}.issubset(available):
        for metric, query in {
            "performance_rows_without_origination": "SELECT count(*) FROM performance p WHERE NOT EXISTS (SELECT 1 FROM origination o WHERE p.loan_identifier=o.loan_identifier)",
            "performance_loans_without_origination": "SELECT count(DISTINCT p.loan_identifier) FROM performance p WHERE NOT EXISTS (SELECT 1 FROM origination o WHERE p.loan_identifier=o.loan_identifier)",
            "origination_loans_without_performance": "SELECT count(DISTINCT o.loan_identifier) FROM origination o WHERE NOT EXISTS (SELECT 1 FROM performance p WHERE p.loan_identifier=o.loan_identifier)",
            "linked_loans": "SELECT count(DISTINCT o.loan_identifier) FROM origination o WHERE EXISTS (SELECT 1 FROM performance p WHERE p.loan_identifier=o.loan_identifier)",
        }.items():
            linkage.append(dict(metric=metric, count=connection.execute(query).fetchone()[0]))
        if not tables["loan_coverage"].empty:
            tables["cohort_summary"] = connection.execute(
                "WITH o AS (SELECT DISTINCT source_cohort,loan_identifier FROM origination), "
                "p AS (SELECT source_cohort,count(*) performance_rows FROM performance GROUP BY source_cohort) "
                "SELECT o.source_cohort AS cohort,count(*) AS loans,coalesce(max(p.performance_rows),0) performance_rows,"
                "count(c.loan_identifier) linked_loans,median(c.distinct_months) median_observed_months,"
                "count(*) FILTER (WHERE c.observed_serious_delinquency) observed_serious_delinquency_loans "
                "FROM o LEFT JOIN coverage c USING(loan_identifier) LEFT JOIN p USING(source_cohort) "
                "GROUP BY o.source_cohort ORDER BY o.source_cohort"
            ).df()
    elif available:
        linkage.append(dict(metric="missing_origination_or_performance_dataset", count=1))
    tables["linkage_quality"] = pd.DataFrame(linkage, columns=TABLE_COLUMNS["linkage_quality"])
    return tables


def run_inspection(project_root: str | Path, chunksize: int = 100_000, max_bytes: int = 10_000_000_000) -> dict[str, Any]:
    """Inspect every sample archive within the combined raw/reference/output cap.

    Exact duplicates are SHA-256 fingerprints of the original parsed fields,
    excluding provenance. No duplicates are dropped. Invalid IDs/periods remain
    present in Parquet and are counted; coverage excludes invalid identifiers.
    ``modeling_ready`` stays false because no outcome horizon or labels exist.
    Full-dataset archives are unsupported: the modest sample origination table
    is returned in memory for notebook exploration.
    """
    if chunksize < 1 or max_bytes < 1:
        raise ValueError("chunksize and max_bytes must be positive")
    root = Path(project_root).expanduser().resolve()
    raw_dir = root / "data" / "raw"
    processed = root / "data" / "processed"
    output = processed / OUTPUT_NAME
    _check_budget(root, max_bytes)
    if not raw_dir.is_dir():
        return _empty_result()
    inventory = inventory_archives(raw_dir, max_bytes=max_bytes)
    if inventory.empty:
        return _empty_result()
    required = {"archive", "member", "kind", "dataset"}
    if not required.issubset(inventory.columns):
        raise ValueError(f"Archive inventory must include {sorted(required)}")
    if not inventory["dataset"].eq("sample").all():
        raise ValueError("Full dataset archives are unsupported by run_inspection; use Freddie Mac sample archives only.")
    unknown = inventory.loc[~inventory["kind"].isin(["origination", "performance"])]
    if not unknown.empty:
        raise ValueError("Archive inventory contains unrecognized member kinds; no archive was skipped.")
    inventory, manifest = _manifest_inventory(inventory, output / "archive_manifest.json")
    processed.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".inspection-staging-", dir=processed))
    connection = None
    audit_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    available: set[str] = set()
    chunks_written = 0
    try:
        for source_number, (_, source) in enumerate(inventory.iterrows()):
            kind = str(source["kind"])
            offset = 0
            source_has_rows = False
            source_chunks = 0
            source_label = f"{Path(source['archive']).name}:{source['member']}"
            print(f"Starting member {source_number + 1}/{len(inventory)}: {source_label} ({kind})", flush=True)
            for chunk_number, raw in enumerate(iter_raw_chunks(source["archive"], source["member"], kind, chunksize=chunksize)):
                if raw.empty:
                    continue
                first_row = int(raw.attrs.get("first_line", offset + 1))
                # A conservative uncompressed in-memory estimate prevents a
                # write that could overshoot the working budget between checks.
                projected = int(raw.memory_usage(index=True, deep=True).sum()) + len(raw) * 256
                _check_budget(root, max_bytes, projected=projected)
                hashes = _raw_hashes(raw)
                cleaner = clean_origination if kind == "origination" else clean_performance
                clean, audit = cleaner(raw)
                if len(clean) != len(raw) or not clean.index.equals(raw.index):
                    raise ValueError("Cleaners must preserve row count and index for exact source-row provenance.")
                if "loan_identifier" not in clean:
                    raise ValueError(f"{kind} cleaner must provide loan_identifier")
                if kind == "performance" and "reporting_period" not in clean:
                    raise ValueError("Performance cleaner must provide reporting_period")
                clean = clean.copy()
                clean["source_archive"] = str(Path(source["archive"]).relative_to(root)) if Path(source["archive"]).is_relative_to(root) else str(source["archive"])
                clean["source_member"] = str(source["member"])
                clean["source_row"] = np.arange(first_row, first_row + len(raw), dtype="int64")
                clean["source_cohort"] = _cohort(source)
                clean["_raw_record_sha256"] = hashes
                target_dir = staging / kind / f"source-{source_number:04d}"
                target_dir.mkdir(parents=True, exist_ok=True)
                clean.to_parquet(target_dir / f"part-{chunk_number:06d}.parquet", index=False, compression="zstd")
                _audit_counts(audit, kind, audit_counts)
                offset += len(raw)
                chunks_written += 1
                source_chunks += 1
                available.add(kind)
                source_has_rows = True
                _check_budget(root, max_bytes)
                if source_chunks % 5 == 0:
                    print(f"  {source_label}: {offset:,} rows written in {source_chunks:,} chunks", flush=True)
            if not source_has_rows:
                audit_counts[(kind, "__source__", "empty_member")] += 1
            print(f"Finished {source_label}: {offset:,} rows in {source_chunks:,} chunks", flush=True)
        if not available:
            raise ValueError("Archives were present but no origination or performance rows were read.")
        print("Building inspection summaries and cross-file quality checks...", flush=True)
        scratch = staging / "_scratch"
        scratch.mkdir()
        connection = duckdb.connect()
        connection.execute("SET memory_limit='1GB'")
        connection.execute("SET threads=2")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET temp_directory={_lit(scratch)}")
        scratch_limit = min(2_000_000_000, max_bytes - _data_bytes(root),
                            shutil.disk_usage(root).free - MIN_FREE_BYTES)
        if scratch_limit < 1:
            raise StorageBudgetError("No scratch capacity remains within the total data and free-disk limits.")
        connection.execute(f"SET max_temp_directory_size='{scratch_limit}B'")
        for kind in available:
            glob = str(staging / kind / "**" / "*.parquet")
            connection.execute(f"CREATE VIEW {_q(kind)} AS SELECT * FROM read_parquet({_lit(glob)},union_by_name=true)")
        result = _summarize(connection, available)
        connection.close()
        connection = None
        shutil.rmtree(scratch)
        result["inventory"] = inventory
        result["cleaning_audit"] = pd.DataFrame(
            [dict(kind=k, column=c, issue=i, count=n) for (k, c, i), n in sorted(audit_counts.items())],
            columns=TABLE_COLUMNS["cleaning_audit"],
        )
        duplicates = result["duplicate_quality"]
        linkage = result["linkage_quality"]
        structural = available == {"origination", "performance"}
        structural &= bool((duplicates[["exact_duplicate_excess_rows", "key_duplicate_excess_rows", "conflicting_keys", "invalid_key_rows"]] == 0).all().all())
        if not linkage.empty:
            structural &= bool((linkage.loc[linkage["metric"] != "linked_loans", "count"] == 0).all())
        warnings = [
            "No prediction labels or fixed outcome horizon have been constructed; modeling_ready is false.",
            "Observed serious delinquency means numeric status >=3 months or RA anywhere in the supplied records; entirely unknown statuses stay unknown. It is not a 12-month default label.",
            "Category distributions show up to 100 values per field; fractions use all origination records.",
        ]
        if not structural:
            warnings.append("Structural quality checks require review: inspect duplicate_quality, linkage_quality, and cleaning_audit. All source records were retained.")
        if any(i == "empty_member" for _, _, i in audit_counts):
            structural = False
            warnings.append("At least one recognized ZIP member was empty; see cleaning_audit.")
        if any(i in {"malformed", "range_violation", "code_violation"} and n for (_, _, i), n in audit_counts.items()):
            structural = False
            warnings.append("Some supplied field values failed format, range, or category checks and require review; see cleaning_audit.")
        paths = {
            "output_dir": str(output),
            "origination_glob": str(output / "origination" / "**" / "*.parquet") if "origination" in available else None,
            "performance_glob": str(output / "performance" / "**" / "*.parquet") if "performance" in available else None,
        }
        result.update(status="complete", structural_quality_pass=bool(structural), modeling_ready=False, warnings=warnings, paths=paths)
        (staging / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summary_dir = staging / "summaries"
        summary_dir.mkdir()
        for key in TABLE_COLUMNS:
            # Origination already exists losslessly in Parquet. Do not duplicate
            # every row as CSV merely to support notebook display.
            if key != "origination":
                _check_budget(root, max_bytes, projected=int(result[key].memory_usage(index=True, deep=True).sum()) + 1024)
                result[key].to_csv(summary_dir / f"{key}.csv", index=False)
                _check_budget(root, max_bytes)
        summary = {
            "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "modeling_ready": False, "structural_quality_pass": bool(structural), "warnings": warnings,
            "paths": paths, "archives": len(manifest), "members": len(inventory), "chunks_written": chunks_written,
            "max_data_bytes": max_bytes, "working_data_limit_bytes": int(max_bytes * .95),
            "exact_duplicate_definition": "SHA-256 of original parsed field values, excluding provenance; excess rows beyond the first identical record",
            "tables": {key: len(result[key]) for key in TABLE_COLUMNS},
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        _check_budget(root, max_bytes)
        backup = processed / ".inspection-previous"
        if backup.exists():
            # A fixed backup is owned only if its manifest identifies this output.
            raise RuntimeError(f"Unresolved previous output backup at {backup}; inspect it before rerunning.")
        if output.exists():
            if not (output / "summary.json").is_file() or not (output / "archive_manifest.json").is_file():
                raise RuntimeError(f"Refusing to replace unrecognized output directory {output}")
            output.rename(backup)
        try:
            staging.rename(output)
        except BaseException:
            if backup.exists() and not output.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        result["data_footprint_bytes"] = _data_bytes(root)
        print(f"Inspection complete: {chunks_written:,} chunks; data footprint {result['data_footprint_bytes']:,} bytes", flush=True)
        return result
    finally:
        if connection is not None:
            connection.close()
        if staging.exists():
            shutil.rmtree(staging)
