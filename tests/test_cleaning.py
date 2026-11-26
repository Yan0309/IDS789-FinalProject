"""Focused data-contract and credit-code checks; fixtures are synthetic, never analysis data."""
import zipfile

import pandas as pd
import pytest

from src.freddie_cleaning import (
    ORIG_COLUMNS, PERF_COLUMNS, clean_origination, clean_performance,
    inventory_archives, iter_raw_chunks,
)


def orig(**updates):
    row = dict.fromkeys(ORIG_COLUMNS, "")
    row.update(loan_identifier="F18Q20000001", classic_fico="720", vantagescore_4="710",
               first_payment_date="201807", maturity_date="204806", original_cltv="80",
               original_ltv="80", number_of_borrowers="2", number_of_units="1")
    row.update(updates)
    return row


def perf(**updates):
    row = dict.fromkeys(PERF_COLUMNS, "")
    row.update(loan_identifier="F18Q20000001", reporting_period="201807", delinquency_status="00")
    row.update(updates)
    return row


def frame(rows, columns):
    return pd.DataFrame(rows, columns=columns, dtype="string")


def zipped(tmp_path, filename, members):
    path = tmp_path / filename
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


def row_text(row, columns):
    return "|".join(row[c] for c in columns) + "\n"


def test_official_header_widths():
    assert len(ORIG_COLUMNS) == len(set(ORIG_COLUMNS)) == 31
    assert len(PERF_COLUMNS) == len(set(PERF_COLUMNS)) == 35


def test_sentinels_blank_and_valid_zero_are_distinct():
    raw = frame([
        orig(classic_fico="9999", vantagescore_4="9999", original_cltv="999",
             original_ltv="999", original_dti="999", mortgage_insurance_percentage="999",
             number_of_units="99", number_of_borrowers="99", first_time_homebuyer_indicator="9",
             occupancy_status="9", channel="9", loan_purpose="9", property_type="99",
             postal_code="000", property_valuation_method="7"),
        orig(mortgage_insurance_percentage="0", original_dti="0", postal_code="021"),
        orig(mortgage_insurance_percentage="", original_dti=""),
    ], ORIG_COLUMNS)
    clean, audit = clean_origination(raw)
    audit = audit.set_index("field")
    for col in ["classic_fico", "vantagescore_4", "original_cltv", "original_ltv", "original_dti",
                "mortgage_insurance_percentage", "number_of_units", "number_of_borrowers",
                "first_time_homebuyer_indicator", "occupancy_status", "channel", "loan_purpose",
                "property_type", "postal_code", "property_valuation_method"]:
        assert pd.isna(clean.loc[0, col]), col
        assert audit.loc[col, "sentinel_count"] == 1
    assert clean.loc[1, "mortgage_insurance_percentage"] == 0
    assert clean.loc[1, "original_dti"] == 0
    assert clean.loc[1, "postal_code"] == "021"
    assert pd.isna(clean.loc[2, "mortgage_insurance_percentage"])
    assert audit.loc["mortgage_insurance_percentage", "raw_blank_count"] == 1
    assert raw.loc[0, "classic_fico"] == "9999"  # no caller mutation


@pytest.mark.parametrize("status, expected, months", [
    ("00", False, 0), ("0", False, 0), ("02", False, 2), ("03", True, 3),
    ("3", True, 3), ("99", True, 99), ("RA", True, None),
    ("XX", None, None), ("", None, None), ("100", None, None), ("broken", None, None),
])
def test_delinquency_semantics(status, expected, months):
    clean, audit = clean_performance(frame([perf(delinquency_status=status)], PERF_COLUMNS))
    actual = clean.loc[0, "serious_delinquency"]
    if expected is None:
        assert pd.isna(actual)
    else:
        assert actual == expected
    if months is None:
        assert pd.isna(clean.loc[0, "delinquency_months"])
    else:
        assert clean.loc[0, "delinquency_months"] == months
    if status in {"00", "0", "02", "03", "3", "99", "RA"}:
        assert clean.loc[0, "delinquency_status"] == status
    else:
        assert pd.isna(clean.loc[0, "delinquency_status"])
    if status == "XX":
        assert audit.set_index("field").loc["delinquency_status", "sentinel_count"] == 1


def test_zero_balance_does_not_mean_default_and_signed_losses_retained():
    clean, _ = clean_performance(frame([
        perf(current_actual_upb="0", zero_balance_code="01", actual_loss="-123.45", mi_recoveries="-4"),
        perf(current_actual_upb="0", zero_balance_code="", delinquency_status="XX"),
    ], PERF_COLUMNS))
    assert clean.loc[0, "voluntary_payoff"]
    assert not clean.loc[0, "serious_delinquency"]
    assert clean.loc[0, "zero_balance_code"] == "01"
    assert clean.loc[0, "actual_loss"] == -123.45
    assert clean.loc[0, "mi_recoveries"] == -4
    assert pd.isna(clean.loc[1, "serious_delinquency"])
    assert "default" not in clean.columns


def test_malformed_numeric_date_ranges_and_codes_are_audited():
    clean, audit = clean_origination(frame([
        orig(classic_fico="oops", original_dti="66", first_payment_date="201813", channel="Z"),
        orig(classic_fico="299", first_payment_date="20181", original_interest_rate="inf"),
        orig(classic_fico=" 750 ", first_payment_date="201801"),
    ], ORIG_COLUMNS))
    a = audit.set_index("field")
    assert a.loc["classic_fico", "malformed_count"] == 1
    assert a.loc["classic_fico", "range_violation_count"] == 1
    assert a.loc["classic_fico", "whitespace_trimmed_count"] == 1
    assert a.loc["first_payment_date", "malformed_count"] == 2
    assert a.loc["channel", "code_violation_count"] == 1
    assert pd.isna(clean.loc[0, "original_dti"])
    assert clean.loc[2, "first_payment_date"] == pd.Timestamp("2018-01-01")
    assert clean["data_quality_issue"].tolist() == [True, True, False]


def test_loan_id_vintage_controls_ranges_and_borrower_topcoding():
    clean, audit = clean_origination(frame([
        orig(loan_identifier="F18Q10000001", first_payment_date="201812", original_ltv="110", original_cltv="210"),
        orig(loan_identifier="F18Q20000001", first_payment_date="201801", original_ltv="110", original_cltv="210"),
        orig(loan_identifier="F10Q10000001", harp_indicator="Y", original_ltv="110", original_cltv="210"),
        orig(loan_identifier="F10Q10000002", original_ltv="5", original_cltv="5", number_of_borrowers="3"),
        orig(loan_identifier="F99Q10000001"),
    ], ORIG_COLUMNS))
    assert pd.isna(clean.loc[0, "original_ltv"])
    assert pd.isna(clean.loc[0, "original_cltv"])
    assert clean.loc[1, "original_ltv"] == 110
    assert clean.loc[2, "original_ltv"] == 110
    assert clean.loc[0, "borrowers_top_coded_2plus"]
    assert not clean.loc[1, "borrowers_top_coded_2plus"]
    assert pd.isna(clean.loc[3, "number_of_borrowers"])
    assert clean.loc[4, "origination_year"] == 1999
    assert audit.set_index("field").loc["original_ltv", "range_violation_count"] == 2


def test_cltv_must_not_be_below_ltv():
    clean, audit = clean_origination(frame([orig(original_cltv="79", original_ltv="80")], ORIG_COLUMNS))
    assert pd.isna(clean.loc[0, "original_cltv"])
    assert audit.set_index("field").loc["original_cltv", "range_violation_count"] == 1


def test_relief_refinance_flag_alone_does_not_destroy_observed_dti():
    clean, audit = clean_origination(frame([
        orig(harp_indicator="Y", original_dti="30", original_ltv="80"),
        orig(harp_indicator="Y", original_dti="999", original_ltv="80"),
    ], ORIG_COLUMNS))
    assert clean.loc[0, "original_dti"] == 30
    assert pd.isna(clean.loc[1, "original_dti"])
    assert audit.set_index("field").loc["original_dti", "range_violation_count"] == 0
    assert audit.set_index("field").loc["original_dti", "sentinel_count"] == 1


def test_harp_ranges_require_indicator_and_ltv_above_80():
    clean, audit = clean_origination(frame([
        orig(loan_identifier="F10Q10000001", harp_indicator="Y", original_ltv="80", original_cltv="210"),
        orig(loan_identifier="F10Q10000002", harp_indicator="Y", original_ltv="81", original_cltv="210"),
        orig(loan_identifier="F10Q10000003", harp_indicator="Y", original_ltv="5", original_cltv="5"),
        orig(loan_identifier="F10Q10000004", harp_indicator="N", original_ltv="81", original_cltv="210"),
    ], ORIG_COLUMNS))
    assert pd.isna(clean.loc[0, "original_cltv"])
    assert clean.loc[1, "original_cltv"] == 210
    assert pd.isna(clean.loc[2, "original_ltv"])
    assert pd.isna(clean.loc[2, "original_cltv"])
    assert pd.isna(clean.loc[3, "original_cltv"])
    assert audit.set_index("field").loc["original_cltv", "range_violation_count"] == 3


def test_unknown_loan_vintage_is_flagged_and_identifier_retained():
    clean, audit = clean_origination(frame([orig(loan_identifier="unknown", original_ltv="110", original_cltv="210")], ORIG_COLUMNS))
    assert clean.loc[0, "loan_identifier"] == "unknown"
    assert clean.loc[0, "vintage_range_policy_unknown"]
    assert clean.loc[0, "original_ltv"] == 110
    assert clean.loc[0, "data_quality_issue"]
    assert audit.set_index("field").loc["loan_identifier", "malformed_count"] == 1


def test_missing_performance_sentinels_and_negative_ages():
    clean, audit = clean_performance(frame([perf(estimated_ltv="999", mortgage_insurance_cancellation_indicator="7",
                                                loan_age="-2", remaining_months_to_legal_maturity="-1")], PERF_COLUMNS))
    assert pd.isna(clean.loc[0, "estimated_ltv"])
    assert pd.isna(clean.loc[0, "mortgage_insurance_cancellation_indicator"])
    assert clean.loc[0, "loan_age"] == -2
    assert clean.loc[0, "remaining_months_to_legal_maturity"] == -1
    assert audit["sentinel_count"].sum() == 2


@pytest.mark.parametrize("columns, cleaner, populated", [
    (ORIG_COLUMNS, clean_origination, orig), (PERF_COLUMNS, clean_performance, perf),
])
def test_dtypes_stable_for_all_missing_chunk(columns, cleaner, populated):
    missing, _ = cleaner(frame([dict.fromkeys(columns, "")], columns))
    present, _ = cleaner(frame([populated()], columns))
    assert missing.dtypes.equals(present.dtypes)


def test_input_column_order_and_count_are_strict():
    with pytest.raises(ValueError, match="schema/order"):
        clean_origination(frame([orig()], ORIG_COLUMNS[:-1]))
    with pytest.raises(ValueError, match="schema/order"):
        clean_origination(frame([orig()], list(reversed(ORIG_COLUMNS))))


def test_streams_chunks_and_preserves_blank_and_zero(tmp_path):
    content = row_text(perf(current_actual_upb="0"), PERF_COLUMNS) * 3
    archive = zipped(tmp_path, "sample_2018.zip", {"sample_perf_2018.txt": content})
    chunks = list(iter_raw_chunks(archive, "sample_perf_2018.txt", "performance", chunksize=2))
    assert [len(c) for c in chunks] == [2, 1]
    assert chunks[0].loc[0, "current_actual_upb"] == "0"
    assert chunks[0].loc[0, "zero_balance_code"] == ""
    assert chunks[0].attrs["first_line"] == 1
    assert chunks[1].attrs["first_line"] == 3
    assert all(str(d) == "string" for d in chunks[0].dtypes)


@pytest.mark.parametrize("bad", ["", "|".join([""] * 34), "|".join([""] * 36)])
def test_checks_every_line_even_after_yield(tmp_path, bad):
    good = row_text(perf(), PERF_COLUMNS)
    archive = zipped(tmp_path, "sample_2018.zip", {"sample_perf_2018.txt": good + bad + "\n"})
    iterator = iter_raw_chunks(archive, "sample_perf_2018.txt", "performance", chunksize=1)
    assert len(next(iterator)) == 1
    with pytest.raises(ValueError, match="line 2: expected 35 fields"):
        next(iterator)


def test_legacy_width_does_not_get_silently_padded(tmp_path):
    archive = zipped(tmp_path, "sample_2018.zip", {"sample_orig_2018.txt": "|".join([""] * 30) + "\n"})
    with pytest.raises(ValueError, match="expected 31 fields, got 30"):
        list(iter_raw_chunks(archive, "sample_orig_2018.txt", "origination"))


def test_inventory_budget_and_valid_pair(tmp_path):
    archive = zipped(tmp_path, "sample_2018.zip", {"sample_orig_2018.txt": "", "sample_perf_2018.txt": ""})
    inventory = inventory_archives(tmp_path, max_bytes=archive.stat().st_size)
    assert len(inventory) == 2
    assert set(inventory.kind) == {"origination", "performance"}
    assert inventory.attrs["raw_zip_bytes"] == archive.stat().st_size
    with pytest.raises(ValueError, match="budget exceeded"):
        inventory_archives(tmp_path, max_bytes=archive.stat().st_size - 1)


def test_inventory_refuses_duplicate_cohort(tmp_path):
    zipped(tmp_path, "one.zip", {"sample_perf_2018.txt": ""})
    zipped(tmp_path, "two.zip", {"sample_svcg_2018.txt": ""})
    with pytest.raises(ValueError, match="Duplicate cohort"):
        inventory_archives(tmp_path)


def test_inventory_refuses_sample_full_mix(tmp_path):
    zipped(tmp_path, "one.zip", {"sample_orig_2018.txt": ""})
    zipped(tmp_path, "two.zip", {"orig_2018Q1.txt": ""})
    with pytest.raises(ValueError, match="mix sample and full"):
        inventory_archives(tmp_path)


def test_member_kind_mismatch(tmp_path):
    archive = zipped(tmp_path, "sample_2018.zip", {"sample_perf_2018.txt": ""})
    with pytest.raises(ValueError, match="name/kind mismatch"):
        list(iter_raw_chunks(archive, "sample_perf_2018.txt", "origination"))
