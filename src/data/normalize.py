"""Normalize raw NHS files into canonical in-memory tables.

Raw files are read only. This module returns derived DataFrames and can write
those tables to data/interim when explicitly requested.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.schemas import DAILY_METRIC_TYPE, DISCHARGE_METRIC

ENCODINGS = ("utf-8", "utf-8-sig", "cp1252")


def read_csv_with_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    """Read an NHS CSV while recording the encoding that successfully decoded it."""
    for encoding in ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            frame.columns = [str(column).strip() for column in frame.columns]
            return frame, encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode {path} using {ENCODINGS}")


def load_csv_directory(directory: Path) -> pd.DataFrame:
    """Load all CSVs in a directory and preserve each source filename."""
    frames = []
    for path in sorted(directory.glob("*.csv")):
        frame, encoding = read_csv_with_fallback(path)
        frame["_source_file"] = path.name
        frame["_source_encoding"] = encoding
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_daily_period(values: pd.Series) -> pd.Series:
    """Parse NHS daily dates without changing the raw text."""
    return pd.to_datetime(values, dayfirst=True, errors="coerce")


def parse_admissions_month(values: pd.Series) -> pd.Series:
    """Parse A&E identifiers such as MSitAE-AUGUST-2026 into month starts."""
    extracted = values.astype(str).str.extract(
        r"(?i)(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)[^0-9]*(202[0-9])"
    )
    return pd.to_datetime(extracted[0] + " " + extracted[1], format="%B %Y", errors="coerce")


def normalize_discharges(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical daily discharge value per organisation and date."""
    selected = frame[
        frame["Level"].astype(str).str.casefold().eq("provider")
        & frame["Metric"].eq(DISCHARGE_METRIC)
        & frame["Metric Type"].eq(DAILY_METRIC_TYPE)
    ].copy()
    selected["date"] = parse_daily_period(selected["Period"])
    selected["daily_discharges"] = pd.to_numeric(selected["Value"], errors="coerce")
    selected = selected.dropna(subset=["date", "Org Code", "daily_discharges"])
    normalized = (
        selected.groupby(["Org Code", "date"], as_index=False)
        .agg(
            **{
                "Org Name": ("Org Name", "first"),
                "daily_discharges": ("daily_discharges", "sum"),
            }
        )
        .sort_values(["Org Code", "date"])
        .reset_index(drop=True)
    )
    normalized["target_next_day"] = normalized.groupby("Org Code")["daily_discharges"].shift(-1)
    next_date = normalized.groupby("Org Code")["date"].shift(-1)
    normalized["target_next_day"] = normalized["target_next_day"].where(
        next_date.eq(normalized["date"] + pd.Timedelta(days=1))
    )
    return normalized


def normalize_beds(frame: pd.DataFrame) -> pd.DataFrame:
    """Return provider-level exact-date G&A bed context without forward filling."""
    selected = frame[
        frame["Level"].astype(str).str.casefold().eq("provider")
        & frame["Type"].eq("Type 1")
    ].copy()
    selected["date"] = parse_daily_period(selected["Period"])
    selected["value_numeric"] = pd.to_numeric(selected["Value"], errors="coerce")
    selected = selected.dropna(subset=["date", "Org Code"])
    normalized = (
        selected.pivot_table(
            index=["Org Code", "Org Name", "date", "Type"],
            columns="Metric",
            values="value_numeric",
            aggfunc="first",
        )
        .reset_index()
    )
    wanted = [
        "Org Code",
        "Org Name",
        "date",
        "Type",
        "G&A beds available",
        "G&A beds occupied",
        "G&A occupancy rate",
    ]
    for column in wanted:
        if column not in normalized:
            normalized[column] = pd.NA
    normalized = normalized[wanted].sort_values(["Org Code", "date"]).reset_index(drop=True)
    normalized["bed_context_available"] = normalized[
        ["G&A beds available", "G&A beds occupied", "G&A occupancy rate"]
    ].notna().any(axis=1).astype(int)
    return normalized


def normalize_ae(frame: pd.DataFrame) -> pd.DataFrame:
    """Return monthly provider-level A&E demand context."""
    normalized = frame.copy()
    normalized["month"] = parse_admissions_month(normalized["Period"])
    normalized = normalized[normalized["month"].notna()].copy()
    measure_columns = [
        column
        for column in normalized.columns
        if column not in {"Period", "Org Code", "Parent Org", "Org name", "month"}
        and not column.startswith("_")
    ]
    for column in measure_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["total_attendances"] = normalized.filter(
        regex=r"^A&E attendances"
    ).sum(axis=1, min_count=1)
    normalized["total_emergency_admissions"] = normalized.filter(
        regex=r"^Emergency admissions|^Other emergency admissions"
    ).sum(axis=1, min_count=1)
    normalized["total_over_4_hours"] = normalized.filter(
        regex=r"^Attendances over 4hrs"
    ).sum(axis=1, min_count=1)
    return normalized[
        [
            "Org Code",
            "month",
            "total_attendances",
            "total_emergency_admissions",
            "total_over_4_hours",
        ]
    ]


def write_interim_tables(raw_root: Path, interim_root: Path) -> dict[str, Path]:
    """Normalize all three raw sources and write derived Parquet tables."""
    interim_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "discharges": interim_root / "discharges_normalized.parquet",
        "beds": interim_root / "beds_normalized.parquet",
        "ae": interim_root / "ae_normalized.parquet",
    }
    normalize_discharges(load_csv_directory(raw_root / "discharges")).to_parquet(
        outputs["discharges"], index=False
    )
    normalize_beds(load_csv_directory(raw_root / "beds")).to_parquet(outputs["beds"], index=False)
    normalize_ae(load_csv_directory(raw_root / "admissions")).to_parquet(outputs["ae"], index=False)
    return outputs
