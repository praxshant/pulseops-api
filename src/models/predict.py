"""Load persisted PulseOps models and generate predictions."""

from pathlib import Path

import joblib
import pandas as pd


def predict(model_path: Path, features: pd.DataFrame) -> pd.Series:
    """Generate predictions from a persisted scikit-learn model."""
    model = joblib.load(model_path)
    return pd.Series(model.predict(features), index=features.index, name="prediction")
