"""Download and register source datasets."""

from pathlib import Path
from urllib.request import urlretrieve


def download_file(url: str, destination: Path) -> Path:
    """Download a dataset to a deterministic local path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)
    return destination
