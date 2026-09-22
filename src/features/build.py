"""Build the leakage-safe canonical forecasting dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)


def _calendar_lags(target: pd.DataFrame) -> pd.DataFrame:
    """Join discharge history by exact calendar offsets, never by row position."""
    result = target.copy()
    source = target[["Org Code", "date", "daily_discharges"]]
    for lag in LAG_DAYS:
        history = source.rename(
            columns={
                "date": "history_date",
                "daily_discharges": f"lag_{lag}_discharge",
            }
        )
        history["date"] = history["history_date"] + pd.Timedelta(days=lag)
        result = result.merge(
            history[["Org Code", "date", f"lag_{lag}_discharge"]],
            on=["Org Code", "date"],
            how="left",
            validate="one_to_one",
        )
    return result


def _calendar_rollings(target: pd.DataFrame) -> pd.DataFrame:
    """Calculate shifted rolling means after inserting missing calendar dates."""
    parts: list[pd.DataFrame] = []
    for organisation_id, group in target.groupby("Org Code", sort=False):
        values = group.set_index("date")["daily_discharges"].asfreq("D")
        rolling = pd.DataFrame({"Org Code": organisation_id, "date": values.index})
        for window in ROLLING_WINDOWS:
            rolling[f"rolling_{window}d_discharge"] = values.shift(1).rolling(
                window,
                min_periods=max(3, window // 2),
            ).mean().to_numpy()
        parts.append(rolling)
    return pd.concat(parts, ignore_index=True)


def _prepare_ae_context(ae: pd.DataFrame) -> pd.DataFrame:
    """Make monthly A&E values available only as previous-month context."""
    context = ae.copy()
    context["feature_month"] = context["month"] + pd.offsets.MonthBegin(1)
    return context.rename(
        columns={
            "total_attendances": "previous_month_total_attendances",
            "total_emergency_admissions": "previous_month_emergency_admissions",
            "total_over_4_hours": "previous_month_over_4_hours",
        }
    )[
        [
            "Org Code",
            "feature_month",
            "previous_month_total_attendances",
            "previous_month_emergency_admissions",
            "previous_month_over_4_hours",
        ]
    ]


def build_forecasting_dataset(
    discharges: pd.DataFrame,
    beds: pd.DataFrame,
    ae: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per organisation/date with leakage-safe candidate features."""
    required_discharge = {"Org Code", "Org Name", "date", "daily_discharges", "target_next_day"}
    missing = required_discharge.difference(discharges.columns)
    if missing:
        raise ValueError(f"Missing discharge columns: {sorted(missing)}")

    target = discharges.copy()
    target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.normalize()
    target = target.dropna(subset=["Org Code", "date"]).sort_values(["Org Code", "date"])
    if target.duplicated(["Org Code", "date"]).any():
        raise ValueError("Discharges must have unique Org Code/date keys")

    target["day_of_week"] = target["date"].dt.dayofweek
    target["is_weekend"] = target["day_of_week"].isin([5, 6]).astype(int)
    target["month"] = target["date"].dt.to_period("M").dt.to_timestamp()
    result = _calendar_lags(target)
    result = result.merge(
        _calendar_rollings(target),
        on=["Org Code", "date"],
        how="left",
        validate="one_to_one",
    )

    bed_columns = [
        "Org Code",
        "date",
        "G&A beds occupied",
        "G&A beds available",
        "G&A occupancy rate",
        "bed_context_available",
    ]
    missing_bed_columns = set(bed_columns).difference(beds.columns)
    if missing_bed_columns:
        raise ValueError(f"Missing bed columns: {sorted(missing_bed_columns)}")
    bed_context = beds[bed_columns].rename(
        columns={
            "G&A beds occupied": "occupied_beds",
            "G&A beds available": "available_beds",
            "G&A occupancy rate": "occupancy_rate",
        }
    )
    result = result.merge(bed_context, on=["Org Code", "date"], how="left", validate="one_to_one")
    result["bed_context_available"] = result["bed_context_available"].fillna(0).astype(int)

    ae_context = _prepare_ae_context(ae)
    result = result.merge(
        ae_context,
        left_on=["Org Code", "month"],
        right_on=["Org Code", "feature_month"],
        how="left",
        validate="many_to_one",
    ).drop(columns="feature_month")
    ae_features = [
        "previous_month_total_attendances",
        "previous_month_emergency_admissions",
        "previous_month_over_4_hours",
    ]
    result["ae_context_available"] = result[ae_features].notna().all(axis=1).astype(int)
    return result.drop(columns="month").reset_index(drop=True)


def build_from_interim(interim_root: Path, output_path: Path) -> pd.DataFrame:
    """Read normalized Parquet tables, build the dataset, and write it to disk."""
    dataset = build_forecasting_dataset(
        pd.read_parquet(interim_root / "discharges_normalized.parquet"),
        pd.read_parquet(interim_root / "beds_normalized.parquet"),
        pd.read_parquet(interim_root / "ae_normalized.parquet"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PulseOps forecasting dataset")
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/forecasting_dataset.parquet"),
    )
    args = parser.parse_args()
    dataset = build_from_interim(args.interim_root, args.output)
    print(f"Wrote {len(dataset):,} rows and {len(dataset.columns):,} columns to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
