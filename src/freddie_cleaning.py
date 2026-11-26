"""Strict, streaming first-pass cleaning for Freddie Mac SFLLD Release 48.

Schema: official July 2026 headers (31 origination / 35 performance fields).
Rules: https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf
Raw ZIPs remain the source of truth. No imputation, row deletion, outcome horizon,
or default classification is performed. Audits are additive across chunks.
"""
from __future__ import annotations

import io
from pathlib import Path
import re
import zipfile
from collections.abc import Iterator

import pandas as pd

ORIG_COLUMNS = [
    "classic_fico", "first_payment_date", "first_time_homebuyer_indicator",
    "maturity_date", "metropolitan_statistical_area", "mortgage_insurance_percentage",
    "number_of_units", "occupancy_status", "original_cltv", "original_dti",
    "original_upb", "original_ltv", "original_interest_rate", "channel",
    "prepayment_penalty_indicator", "amortization_type", "property_state",
    "property_type", "postal_code", "loan_identifier", "loan_purpose",
    "original_loan_term", "number_of_borrowers", "seller_name", "super_conforming_flag",
    "pre_harp_loan_sequence_number", "special_eligibility_program", "harp_indicator",
    "property_valuation_method", "interest_only_indicator", "vantagescore_4",
]
PERF_COLUMNS = [
    "loan_identifier", "reporting_period", "current_actual_upb", "delinquency_status",
    "loan_age", "remaining_months_to_legal_maturity", "defect_settlement_date",
    "modification_flag", "zero_balance_code", "zero_balance_effective_date",
    "current_interest_rate", "current_non_interest_bearing_upb", "last_paid_installment_date",
    "mi_recoveries", "net_sales_proceeds", "non_mi_recoveries", "total_expenses",
    "legal_costs", "maintenance_and_preservation_costs", "taxes_and_insurance",
    "miscellaneous_expenses", "actual_loss", "cumulative_modification_costs",
    "interest_rate_step_indicator", "payment_deferral_flag", "estimated_ltv",
    "zero_balance_removal_upb", "delinquent_accrued_interest", "delinquency_due_to_disaster",
    "borrower_assistance_plan", "current_period_modification_costs",
    "current_interest_bearing_upb", "mortgage_insurance_cancellation_indicator",
    "servicer_name", "bankruptcy_cramdown_costs",
]
_SCHEMAS = {"origination": ORIG_COLUMNS, "performance": PERF_COLUMNS}
_INVENTORY_COLUMNS = ["archive", "member", "kind", "dataset", "cohort", "year",
                      "quarter", "archive_bytes", "compressed_bytes", "uncompressed_bytes"]


def _identify(member: str) -> dict | None:
    name = Path(member).name
    sample = re.fullmatch(r"sample_(orig|perf|svcg)_(\d{4})\.txt", name, re.I)
    full = re.fullmatch(r"(orig|perf)_(\d{4})Q([1-4])\.txt", name, re.I)
    if sample:
        prefix, year = sample.groups()
        return {"kind": "origination" if prefix.lower() == "orig" else "performance",
                "dataset": "sample", "cohort": year, "year": int(year), "quarter": None}
    if full:
        prefix, year, quarter = full.groups()
        return {"kind": "origination" if prefix.lower() == "orig" else "performance",
                "dataset": "full", "cohort": f"{year}Q{quarter}",
                "year": int(year), "quarter": int(quarter)}
    return None


def inventory_archives(raw_dir, max_bytes: int = 10_000_000_000) -> pd.DataFrame:
    """Inventory ZIP members without extraction; reject mixed/duplicate cohorts.

    The limit applies to the total *on-disk ZIP sizes*, including ZIP overhead.
    Each (dataset, cohort, kind) must appear only once. An orig/perf pair is valid.
    Empty directories return an empty inventory; non-ZIP files are not ingested.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {root}")
    archives = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip")
    total = sum(p.stat().st_size for p in archives)
    if total > max_bytes:
        raise ValueError(f"Raw ZIP budget exceeded: {total:,} > {max_bytes:,} bytes")
    rows, seen, datasets = [], set(), set()
    for path in archives:
        found = False
        with zipfile.ZipFile(path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                identity = _identify(member.filename)
                if identity is None:
                    if member.filename.lower().endswith(".txt"):
                        raise ValueError(f"Unrecognized data member: {path.name}:{member.filename}")
                    continue
                found = True
                key = (identity["dataset"], identity["cohort"], identity["kind"])
                if key in seen:
                    raise ValueError(f"Duplicate cohort member: {key} in {path}")
                seen.add(key)
                datasets.add(identity["dataset"])
                rows.append({"archive": str(path.resolve()), "member": member.filename,
                             **identity, "archive_bytes": path.stat().st_size,
                             "compressed_bytes": member.compress_size,
                             "uncompressed_bytes": member.file_size})
        if not found:
            raise ValueError(f"ZIP has no recognized Freddie Mac data members: {path}")
    if len(datasets) > 1:
        raise ValueError("Do not mix sample and full datasets in the same run")
    result = pd.DataFrame(rows, columns=_INVENTORY_COLUMNS)
    result["quarter"] = result["quarter"].astype("Int64")
    result.attrs["raw_zip_bytes"] = total
    return result


def iter_raw_chunks(archive, member: str, kind: str, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    """Read ZIP text incrementally, checking the exact width of *every* line.

    A malformed line (including a blank line) raises with its 1-based line number.
    Earlier yielded chunks may already exist; callers should publish only after
    a complete successful run. The reader never silently skips or repairs rows.
    """
    if kind not in _SCHEMAS:
        raise ValueError(f"kind must be one of {tuple(_SCHEMAS)}")
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    identity = _identify(member)
    if identity is None or identity["kind"] != kind:
        raise ValueError(f"Member name/kind mismatch: {member!r}, {kind!r}")
    columns = _SCHEMAS[kind]
    rows = []
    with zipfile.ZipFile(archive) as zf, zf.open(member) as stream:
        with io.TextIOWrapper(stream, encoding="utf-8-sig", errors="strict", newline="") as text:
            for line_number, line in enumerate(text, start=1):
                fields = line.rstrip("\r\n").split("|")
                if len(fields) != len(columns):
                    raise ValueError(f"Schema mismatch in {archive}:{member}, line {line_number}: "
                                     f"expected {len(columns)} fields, got {len(fields)}")
                rows.append(fields)
                if len(rows) == chunksize:
                    frame = pd.DataFrame(rows, columns=columns, dtype="string")
                    frame.attrs.update(archive=str(archive), member=member,
                                       first_line=line_number - len(rows) + 1)
                    yield frame
                    rows = []
            if rows:
                frame = pd.DataFrame(rows, columns=columns, dtype="string")
                frame.attrs.update(archive=str(archive), member=member,
                                   first_line=line_number - len(rows) + 1)
                yield frame


_ORIG_SENTINELS = {
    "classic_fico": "9999", "vantagescore_4": "9999",
    "mortgage_insurance_percentage": "999", "original_cltv": "999",
    "original_dti": "999", "original_ltv": "999", "number_of_units": "99",
    "number_of_borrowers": "99", "first_time_homebuyer_indicator": "9",
    "occupancy_status": "9", "channel": "9", "loan_purpose": "9",
    "property_type": "99", "postal_code": "000", "property_valuation_method": "7",
}
_PERF_SENTINELS = {"estimated_ltv": "999", "mortgage_insurance_cancellation_indicator": "7",
                   "delinquency_status": "XX"}
_ORIG_CODES = {
    "first_time_homebuyer_indicator": {"Y", "N"}, "occupancy_status": {"P", "I", "S"},
    "channel": {"R", "B", "C", "T"}, "prepayment_penalty_indicator": {"Y", "N"},
    "amortization_type": {"FRM", "ARM"}, "property_type": {"CP", "CO", "PU", "SF", "MH"},
    "loan_purpose": {"P", "C", "N", "R"}, "super_conforming_flag": {"Y", "N"},
    "special_eligibility_program": {"H", "F", "R"}, "harp_indicator": {"Y", "N"},
    "property_valuation_method": {"1", "2", "3", "4"}, "interest_only_indicator": {"Y", "N"},
}


class _Cleaner:
    def __init__(self, raw: pd.DataFrame, columns: list[str], sentinels: dict[str, str]):
        if list(raw.columns) != columns:
            raise ValueError("Input schema/order must exactly match the current official columns")
        self.frame = raw.astype("string").copy()
        self.audit = {}
        self.issue = pd.Series(False, index=raw.index, dtype="boolean")
        for col in columns:
            source = self.frame[col]
            stripped = source.str.strip()
            blank = stripped.isna() | stripped.eq("").fillna(False)
            sentinel = stripped.eq(sentinels.get(col, "\0")).fillna(False)
            self.audit[col] = {"field": col, "raw_rows": len(raw),
                "raw_blank_count": int(blank.sum()), "sentinel_count": int(sentinel.sum()),
                "sentinel_values": sentinels.get(col, ""), "malformed_count": 0,
                "range_violation_count": 0, "code_violation_count": 0,
                "whitespace_trimmed_count": int((source.notna() & source.ne(stripped)).sum())}
            self.frame[col] = stripped.mask(blank | sentinel, pd.NA)

    def _reject(self, col, invalid, reason):
        invalid = invalid.fillna(False).astype("boolean")
        self.audit[col][reason] += int(invalid.sum())
        self.issue |= invalid
        self.frame[col] = self.frame[col].mask(invalid)

    def number(self, col, *, integer=False, low=None, high=None):
        source = self.frame[col]
        pattern = r"[+-]?\d+" if integer else r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        valid = source.str.fullmatch(pattern).fillna(False)
        parsed = pd.to_numeric(source.where(valid), errors="coerce")
        malformed = source.notna() & (~valid | parsed.isna() | parsed.isin([float("inf"), -float("inf")]))
        if integer:
            malformed |= parsed.notna() & ((parsed > 2**63 - 1) | (parsed < -(2**63)))
        self._reject(col, malformed, "malformed_count")
        self.frame[col] = pd.to_numeric(self.frame[col], errors="coerce").astype("Int64" if integer else "Float64")
        invalid = pd.Series(False, index=source.index)
        if low is not None:
            invalid |= self.frame[col].lt(low).fillna(False)
        if high is not None:
            invalid |= self.frame[col].gt(high).fillna(False)
        self._reject(col, invalid, "range_violation_count")

    def date(self, col):
        source = self.frame[col]
        valid = source.str.fullmatch(r"\d{6}").fillna(False)
        parsed = pd.to_datetime(source.where(valid), format="%Y%m", errors="coerce")
        self._reject(col, source.notna() & (~valid | parsed.isna()), "malformed_count")
        self.frame[col] = parsed.astype("datetime64[ns]")

    def codes(self, col, codes):
        self._reject(col, self.frame[col].notna() & ~self.frame[col].isin(codes), "code_violation_count")

    def finish(self):
        self.frame["data_quality_issue"] = self.issue
        for col, row in self.audit.items():
            row["cleaned_missing_count"] = int(self.frame[col].isna().sum())
        return self.frame, pd.DataFrame(self.audit.values())


def _loan_cohort(cleaner: _Cleaner):
    ids = cleaner.frame["loan_identifier"]
    valid = ids.str.fullmatch(r"[FA]\d{2}Q[1-4]\d{7}").fillna(False)
    # Preserve malformed identifiers verbatim for investigation; do not lose keys.
    cleaner.audit["loan_identifier"]["malformed_count"] = int((ids.notna() & ~valid).sum())
    cleaner.issue |= ~valid
    year = pd.to_numeric(ids.where(valid).str[1:3], errors="coerce").astype("Int64")
    year = (year + 2000).mask(year.eq(99).fillna(False), 1999)
    quarter = pd.to_numeric(ids.where(valid).str[4:5], errors="coerce").astype("Int64")
    cleaner.frame["origination_year"] = year
    cleaner.frame["origination_quarter"] = (year.astype("string") + "Q" + quarter.astype("string"))
    return (year < 2018) | ((year == 2018) & (quarter == 1))


def clean_origination(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean one origination chunk, retaining rows and auditable missingness.

    Old-vintage LTV/CLTV limits and borrower top-coding use the loan ID vintage,
    never the first-payment date. HARP ranges require Relief Refinance indicator
    Y *and* original LTV >80; Y alone also includes non-HARP relief refinances.
    Unknown vintage is flagged and only the broad 1..998 LTV limits are applied.
    """
    c = _Cleaner(raw, ORIG_COLUMNS, _ORIG_SENTINELS)
    old = _loan_cohort(c)
    c.frame["vintage_range_policy_unknown"] = old.isna().astype("boolean")
    c.issue |= old.isna()
    for col in ["first_payment_date", "maturity_date"]:
        c.date(col)
    for col in ["classic_fico", "vantagescore_4"]:
        c.number(col, integer=True, low=300, high=850)
    c.number("mortgage_insurance_percentage", integer=True, low=0, high=55)
    c.number("number_of_units", integer=True, low=1, high=4)
    c.number("original_dti", integer=True, low=0, high=65)
    c.number("original_upb", low=0)
    c.number("original_interest_rate", low=0)
    c.number("original_loan_term", integer=True, low=1)
    c.number("number_of_borrowers", integer=True, low=1, high=10)
    for col in ["original_ltv", "original_cltv"]:
        c.number(col, integer=True, low=1, high=998)
    true_harp = (c.frame["harp_indicator"].eq("Y").astype("boolean")
                 & c.frame["original_ltv"].gt(80)).fillna(False)
    old_non_harp = old.fillna(False) & ~true_harp
    for col, old_high in [("original_ltv", 105), ("original_cltv", 200)]:
        c._reject(col, old_non_harp & ((c.frame[col] < 6) | (c.frame[col] > old_high)),
                  "range_violation_count")
    c._reject("original_cltv", c.frame["original_cltv"] < c.frame["original_ltv"], "range_violation_count")
    c._reject("number_of_borrowers", old.fillna(False) & (c.frame["number_of_borrowers"] > 2),
              "range_violation_count")
    c.frame["borrowers_top_coded_2plus"] = (old & c.frame["number_of_borrowers"].eq(2)).astype("boolean")
    for col, codes in _ORIG_CODES.items():
        c.codes(col, codes)
    for col, pattern in [("postal_code", r"\d{3}"), ("metropolitan_statistical_area", r"\d{5}"),
                         ("property_state", r"[A-Z]{2}")]:
        c._reject(col, c.frame[col].notna() & ~c.frame[col].str.fullmatch(pattern), "malformed_count")
    return c.finish()


def clean_performance(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a performance chunk; serious_delinquency is descriptive only.

    Numeric status >=3 or RA is serious; XX/blank/invalid stays unknown.
    Status 99 is valid (capped severe delinquency), never a missing sentinel.
    No default label is inferred from a zero balance or an UPB of zero.
    Negative loss/recovery/cost amounts are retained (official sign convention).
    """
    c = _Cleaner(raw, PERF_COLUMNS, _PERF_SENTINELS)
    _loan_cohort(c)
    for col in ["reporting_period", "defect_settlement_date", "zero_balance_effective_date",
                "last_paid_installment_date"]:
        c.date(col)
    integer_cols = {"loan_age", "remaining_months_to_legal_maturity"}
    numeric_cols = ["current_actual_upb", "current_interest_rate", "current_non_interest_bearing_upb",
                    "mi_recoveries", "net_sales_proceeds", "non_mi_recoveries", "total_expenses",
                    "legal_costs", "maintenance_and_preservation_costs", "taxes_and_insurance",
                    "miscellaneous_expenses", "actual_loss", "cumulative_modification_costs",
                    "zero_balance_removal_upb", "delinquent_accrued_interest",
                    "current_period_modification_costs", "current_interest_bearing_upb",
                    "bankruptcy_cramdown_costs"]
    for col in integer_cols:
        c.number(col, integer=True)  # Negative ages/maturity can be legitimate.
    for col in numeric_cols:
        c.number(col, low=0 if col in {"current_actual_upb", "current_interest_rate",
                    "current_non_interest_bearing_upb", "current_interest_bearing_upb"} else None)
    c.number("estimated_ltv", integer=True, low=1, high=998)
    status = c.frame["delinquency_status"]
    valid = status.str.fullmatch(r"\d{1,2}").fillna(False) | status.eq("RA").fillna(False)
    c._reject("delinquency_status", status.notna() & ~valid, "code_violation_count")
    status = c.frame["delinquency_status"]
    numeric = pd.to_numeric(status.where(status.ne("RA")), errors="coerce").astype("Int64")
    c.frame["delinquency_months"] = numeric
    c.frame["reo_acquisition"] = status.eq("RA").astype("boolean")
    c.frame["serious_delinquency"] = (numeric.ge(3) | c.frame["reo_acquisition"]).astype("boolean")
    c.frame["serious_delinquency"] = c.frame["serious_delinquency"].mask(status.isna(), pd.NA)
    c.codes("zero_balance_code", {"01", "02", "03", "09", "15", "16", "96"})
    c.frame["voluntary_payoff"] = c.frame["zero_balance_code"].eq("01").astype("boolean")
    c.codes("modification_flag", {"Y", "P"})
    c.codes("interest_rate_step_indicator", {"Y", "N"})
    c.codes("payment_deferral_flag", {"C", "P"})
    c.codes("delinquency_due_to_disaster", {"Y"})
    c.codes("borrower_assistance_plan", {"F", "R", "T"})
    c.codes("mortgage_insurance_cancellation_indicator", {"Y", "N"})
    return c.finish()
