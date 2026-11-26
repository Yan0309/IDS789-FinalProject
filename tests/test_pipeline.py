"""Tiny synthetic records test mechanics only; they are not Freddie Mac data."""
import json
from pathlib import Path
from unittest.mock import Mock
import zipfile

import duckdb
import pandas as pd
import pytest

from src.freddie_cleaning import ORIG_COLUMNS, PERF_COLUMNS
from src import inspection_pipeline as pipeline


def orig(loan="F20Q10000001", **changes):
    row = dict.fromkeys(ORIG_COLUMNS, "")
    row.update(loan_identifier=loan, classic_fico="720", first_payment_date="202002",
               maturity_date="205001", original_upb="200000", original_ltv="80",
               original_cltv="80", original_dti="35", number_of_units="1",
               number_of_borrowers="1", original_interest_rate="3.5", original_loan_term="360",
               property_state="NC", property_type="SF", occupancy_status="P", loan_purpose="P",
               channel="R", amortization_type="FRM", harp_indicator="N")
    row.update(changes)
    return row


def perf(loan="F20Q10000001", month="202002", status="0", **changes):
    row = dict.fromkeys(PERF_COLUMNS, "")
    row.update(loan_identifier=loan, reporting_period=month, current_actual_upb="199900",
               delinquency_status=status, loan_age="1", remaining_months_to_legal_maturity="359",
               current_interest_rate="3.5")
    row.update(changes)
    return row


def write_archive(root, origins, performances, year=2020):
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"sample_{year}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for kind, rows, columns in (("orig", origins, ORIG_COLUMNS), ("svcg", performances, PERF_COLUMNS)):
            archive.writestr(f"sample_{kind}_{year}.txt", "".join("|".join(str(row[c]) for c in columns) + "\n" for row in rows))
    return path


def parquet_rows(result, kind):
    with duckdb.connect() as connection:
        return connection.execute("SELECT * FROM read_parquet(?,union_by_name=true)", [result["paths"][f"{kind}_glob"]]).df()


def test_no_archives_returns_explicit_awaiting_data(tmp_path):
    result = pipeline.run_inspection(tmp_path)
    assert result["status"] == "awaiting_data"
    assert result["modeling_ready"] is False
    assert all(result[key].empty for key in pipeline.TABLE_COLUMNS)
    assert not (tmp_path / "data" / "processed").exists()


def test_full_dataset_is_rejected_before_processing(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "full_2020Q1.zip", "w") as archive:
        archive.writestr("orig_2020Q1.txt", "|".join(orig()[c] for c in ORIG_COLUMNS) + "\n")
        archive.writestr("perf_2020Q1.txt", "|".join(perf()[c] for c in PERF_COLUMNS) + "\n")
    reader = Mock(side_effect=AssertionError("Full dataset must be rejected before reading rows"))
    monkeypatch.setattr(pipeline, "iter_raw_chunks", reader)
    with pytest.raises(ValueError, match="Full dataset archives are unsupported.*sample archives only"):
        pipeline.run_inspection(tmp_path)
    assert not reader.called
    assert not (tmp_path / "data" / "processed").exists()


def test_member_progress_reports_start_five_chunks_and_finish(tmp_path, capsys):
    write_archive(tmp_path, [orig()], [perf() for _ in range(6)])
    pipeline.run_inspection(tmp_path, chunksize=1)
    output = capsys.readouterr().out
    assert "Starting member 1/2:" in output
    assert "Starting member 2/2:" in output
    assert "5 rows written in 5 chunks" in output
    assert "Finished sample_2020.zip:sample_svcg_2020.txt: 6 rows in 6 chunks" in output
    assert "Inspection complete:" in output


def test_streaming_provenance_coverage_and_loss_availability(tmp_path):
    write_archive(tmp_path, [orig(), orig("F20Q10000002")], [
        perf(), perf(month="202004", status="3", actual_loss="-1250"),
        perf("F20Q10000002", "202002", "XX"),
        perf("F20Q10000002", "202003", "RA"),
    ])
    result = pipeline.run_inspection(tmp_path, chunksize=1)
    assert result["status"] == "complete"
    assert result["structural_quality_pass"] is True
    assert result["modeling_ready"] is False
    performance = parquet_rows(result, "performance")
    assert len(performance) == 4
    assert performance["_raw_record_sha256"].map(len).eq(32).all()
    assert performance["source_row"].tolist() == [1, 2, 3, 4]
    assert performance["source_archive"].str.endswith("sample_2020.zip").all()
    assert performance["source_cohort"].eq("2020").all()
    assert "_raw_record_sha256" not in result["origination"]
    coverage = result["loan_coverage"].set_index("loan_identifier")
    assert coverage.loc["F20Q10000001", "distinct_months"] == 2
    assert coverage.loc["F20Q10000001", "gap_months"] == 1
    assert coverage.loc["F20Q10000001", "max_delinquency"] == 3
    assert coverage["observed_serious_delinquency"].all()  # numeric >=3 and RA
    actual_loss = result["loss_field_availability"].set_index("column").loc["actual_loss"]
    assert actual_loss["nonmissing_count"] == 1
    assert actual_loss["nonzero_count"] == 1
    assert actual_loss["missing_count"] == 3
    assert result["cohort_summary"].iloc[0]["loans"] == 2
    assert result["cohort_summary"].iloc[0]["performance_rows"] == 4
    output = Path(result["paths"]["output_dir"])
    assert json.loads((output / "summary.json").read_text())["chunks_written"] == 6
    assert (output / "summaries" / "loan_coverage.csv").exists()
    assert result["data_footprint_bytes"] < 10_000_000_000


def test_unknown_delinquency_remains_unknown(tmp_path):
    write_archive(tmp_path, [orig()], [perf(status="XX")])
    result = pipeline.run_inspection(tmp_path, chunksize=1)
    coverage = result["loan_coverage"].iloc[0]
    assert pd.isna(coverage["observed_serious_delinquency"])
    assert pd.isna(coverage["max_delinquency"])


def test_duplicates_conflicts_unmatched_and_invalid_keys_are_reported(tmp_path):
    identical = perf()
    write_archive(tmp_path, [orig(), orig(), orig("F20Q10000002")], [
        identical, identical.copy(), perf(current_actual_upb="180000"),
        perf("F20Q10000003", "202003"),
        perf("invalid", "202013"),
    ])
    result = pipeline.run_inspection(tmp_path, chunksize=1)
    assert result["status"] == "complete"
    assert result["structural_quality_pass"] is False
    assert len(result["origination"]) == 3  # No silent deduplication.
    assert len(parquet_rows(result, "performance")) == 5
    quality = result["duplicate_quality"].set_index("kind")
    assert quality.loc["origination", "exact_duplicate_excess_rows"] == 1
    assert quality.loc["origination", "key_duplicate_excess_rows"] == 1
    assert quality.loc["origination", "conflicting_keys"] == 0
    assert quality.loc["performance", "exact_duplicate_excess_rows"] == 1
    assert quality.loc["performance", "key_duplicate_excess_rows"] == 2
    assert quality.loc["performance", "conflicting_keys"] == 1
    assert quality.loc["performance", "invalid_key_rows"] == 1
    linkage = result["linkage_quality"].set_index("metric")["count"]
    assert linkage["performance_rows_without_origination"] == 2
    assert linkage["performance_loans_without_origination"] == 2
    assert linkage["origination_loans_without_performance"] == 1
    audit = result["cleaning_audit"]
    assert ((audit["column"] == "reporting_period") & (audit["issue"] == "malformed")).any()


def test_raw_fingerprints_do_not_collapse_different_cleaned_unknowns(tmp_path):
    write_archive(tmp_path, [orig(classic_fico="9999"), orig(classic_fico="")], [perf()])
    result = pipeline.run_inspection(tmp_path, chunksize=1)
    quality = result["duplicate_quality"].set_index("kind")
    assert quality.loc["origination", "exact_duplicate_excess_rows"] == 0
    assert quality.loc["origination", "conflicting_keys"] == 1
    assert result["origination"]["classic_fico"].isna().all()
    sentinel = result["cleaning_audit"].query("kind=='origination' and column=='classic_fico' and issue=='sentinel'")
    assert sentinel["count"].sum() == 1


def test_binary_raw_fingerprints_preserve_field_boundaries_empty_and_null():
    raw = pd.DataFrame([["a|b", ""], ["a", "b|"], ["a|b", pd.NA], ["a|b", ""]], dtype="string")
    hashes = pipeline._raw_hashes(raw)
    assert all(isinstance(value, bytes) and len(value) == 32 for value in hashes)
    assert hashes[0] == hashes[3]
    assert len(set(hashes[:3])) == 3


def test_rerun_reuses_archive_hash_and_preserves_previous_on_parse_failure(tmp_path, monkeypatch):
    path = write_archive(tmp_path, [orig()], [perf()])
    first = pipeline.run_inspection(tmp_path, chunksize=1)
    output = Path(first["paths"]["output_dir"])
    hasher = Mock(side_effect=AssertionError("Unchanged archive was needlessly hashed"))
    monkeypatch.setattr(pipeline, "_sha256", hasher)
    second = pipeline.run_inspection(tmp_path, chunksize=1)
    assert second["status"] == "complete"
    assert not hasher.called
    monkeypatch.undo()
    before = (output / "summary.json").read_bytes()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("sample_svcg_2020.txt", "malformed|row\n")
    # Duplicate ZIP names themselves are also an explicit inventory error.
    with pytest.raises(ValueError):
        pipeline.run_inspection(tmp_path, chunksize=1)
    assert (output / "summary.json").read_bytes() == before
    assert not list((tmp_path / "data" / "processed").glob(".inspection-staging-*"))


def test_failure_after_first_chunk_keeps_completed_output(tmp_path):
    path = write_archive(tmp_path, [orig()], [perf()])
    first = pipeline.run_inspection(tmp_path, chunksize=1)
    output = Path(first["paths"]["output_dir"])
    before = (output / "summary.json").read_bytes()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample_orig_2020.txt", "|".join(orig()[c] for c in ORIG_COLUMNS) + "\n")
        archive.writestr("sample_svcg_2020.txt", "|".join(perf()[c] for c in PERF_COLUMNS) + "\nmalformed|row\n")
    with pytest.raises(ValueError, match="line 2"):
        pipeline.run_inspection(tmp_path, chunksize=1)
    assert (output / "summary.json").read_bytes() == before
    assert not list((tmp_path / "data" / "processed").glob(".inspection-staging-*"))


def test_combined_data_budget_counts_reference_and_other_processed_files(tmp_path):
    write_archive(tmp_path, [orig()], [perf()])
    reference = tmp_path / "data" / "reference"
    reference.mkdir()
    (reference / "user_reference.bin").write_bytes(b"x" * 10_000)
    with pytest.raises(pipeline.StorageBudgetError, match="budget"):
        pipeline.run_inspection(tmp_path, max_bytes=5_000)
    assert (reference / "user_reference.bin").exists()
    assert not (tmp_path / "data" / "processed" / pipeline.OUTPUT_NAME).exists()


def test_low_free_disk_stops_before_writing(tmp_path, monkeypatch):
    write_archive(tmp_path, [orig()], [perf()])
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 1_000_000})())
    with pytest.raises(pipeline.StorageBudgetError, match="free disk"):
        pipeline.run_inspection(tmp_path)
    assert not (tmp_path / "data" / "processed" / pipeline.OUTPUT_NAME).exists()
