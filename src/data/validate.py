"""Validation contracts for raw and canonical PulseOps data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.normalize import load_csv_directory, read_csv_with_fallback
from data.schemas import (
    CANONICAL_DISCHARGE_COLUMNS,
    DAILY_METRIC_TYPE,
    DISCHARGE_METRIC,
    DISCHARGE_RAW_COLUMNS,
    ValidationReport,
)


def validate_discharge_raw(frame: pd.DataFrame) -> ValidationReport:
    """Validate the raw discharge schema and required metric contract."""
    report = ValidationReport("raw discharges")
    missing = DISCHARGE_RAW_COLUMNS.difference(frame.columns)
    if missing:
        report.error(f"Missing raw discharge columns: {sorted(missing)}")
    else:
        report.check("Discharge schema")
    if "Period" in frame:
        periods = pd.to_datetime(frame["Period"], dayfirst=True, errors="coerce")
        if periods.isna().all():
            report.error("No valid discharge periods found")
        else:
            report.check("Date integrity")
    if "Org Code" in frame and frame["Org Code"].isna().any():
        report.error("Missing organisation identifiers")
    else:
        report.check("Organisation identifiers")
    if "Metric" in frame and not frame["Metric"].eq(DISCHARGE_METRIC).any():
        report.error(f"Required metric not present: {DISCHARGE_METRIC}")
    if "Metric Type" in frame and not frame["Metric Type"].eq(DAILY_METRIC_TYPE).any():
        report.error(f"Required metric type not present: {DAILY_METRIC_TYPE}")
    return report


def validate_normalized_discharges(frame: pd.DataFrame) -> ValidationReport:
    """Validate the canonical organisation-day discharge table."""
    report = ValidationReport("normalized discharges")
    missing = CANONICAL_DISCHARGE_COLUMNS.difference(frame.columns)
    if missing:
        report.error(f"Missing canonical columns: {sorted(missing)}")
        return report
    report.check("Discharge schema")
    if frame.empty:
        report.error("Normalized discharge table is empty")
    if frame["Org Code"].isna().any() or frame["Org Code"].astype(str).str.strip().eq("").any():
        report.error("Missing organisation identifiers")
    else:
        report.check("Organisation identifiers")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        report.error("Invalid dates")
    else:
        report.check("Date integrity")
    values = pd.to_numeric(frame["daily_discharges"], errors="coerce")
    if values.isna().any():
        report.error("Non-numeric daily discharge values")
    elif (values < 0).any():
        report.error("Negative daily discharge values")
    else:
        report.check("No negative values")
    if frame.duplicated(["Org Code", "date"]).any():
        report.error("Duplicate organisation/date keys")
    else:
        report.check("Duplicate-key check")
    return report


def validate_data_directory(raw_dir: Path) -> ValidationReport:
    """Validate all raw discharge files in a directory."""
    frame = load_csv_directory(raw_dir)
    return validate_discharge_raw(frame)


def validate_csv(path: Path) -> pd.DataFrame:
    """Load and validate one raw discharge CSV for legacy script entry points."""
    frame, _ = read_csv_with_fallback(path)
    report = validate_discharge_raw(frame)
    if not report.passed:
        raise ValueError("; ".join(report.errors))
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PulseOps NHS discharge data")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/discharges"))
    args = parser.parse_args()
    report = validate_data_directory(args.raw_dir)
    print("PULSEOPS DATA VALIDATION\n")
    for check in report.checks:
        print(f"[OK] {check}")
    for error in report.errors:
        print(f"[FAIL] {error}")
    print(f"\nSTATUS: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
