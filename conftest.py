"""Shared pytest fixtures and path setup for the ECG-monitor test suite.

`pythonpath = ["src", "backend"]` in pyproject.toml makes the phase modules and
the backend importable, so tests can `from ecg_analyzer import ...` directly with
no per-file sys.path hacks.
"""

import numpy as np
import pytest


class FakeModel:
    """Minimal stand-in for the trained RandomForest.

    Mimics the two attributes predict_combined touches: `classes_` (integer class
    indices into CLASSES) and `predict_proba` (returns a scripted probability row).
    Lets us exercise the rules/ML merge and the probability-expansion logic without
    loading the 8 MB joblib or the MIT-BIH data.
    """

    def __init__(self, classes, proba_row):
        self.classes_ = np.asarray(classes)
        self._proba_row = np.asarray(proba_row, dtype=float)

    def predict_proba(self, X):
        # One row in, one row out — ignore the feature values, return the script.
        return np.array([self._proba_row])


@pytest.fixture
def make_model():
    """Factory: make_model(classes, proba_row) -> FakeModel."""
    return FakeModel
