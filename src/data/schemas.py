"""Column contracts for the NHS source and canonical PulseOps tables."""

from dataclasses import dataclass, field

DISCHARGE_RAW_COLUMNS = {
    "Period",
    "Level",
    "Region",
    "ICB",
    "Org Code",
    "Org Name",
    "Metric",
    "Metric Type",
    "Metric Group",
    "Value",
}

BED_RAW_COLUMNS = {
    "Period",
    "Level",
    "Region",
    "ICB",
    "Org Code",
    "Org Name",
    "Metric",
    "Type",
    "Value",
}

ADMISSIONS_RAW_COLUMNS = {
    "Period",
    "Org Code",
    "Parent Org",
    "Org name",
}

DISCHARGE_METRIC = "Number of patients discharged"
DAILY_METRIC_TYPE = "Daily metric"

CANONICAL_DISCHARGE_COLUMNS = {
    "Org Code",
    "Org Name",
    "date",
    "daily_discharges",
}


@dataclass
class ValidationReport:
    """Structured validation result suitable for CLI output and tests."""

    dataset: str
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def check(self, name: str) -> None:
        self.checks.append(name)

    def error(self, message: str) -> None:
        self.errors.append(message)
