"""Flask backend endpoints (`/api/health`, `/api/analyze`).

Covers the request-validation error matrix (which returns before any heavy DSP)
and the JSON NaN/Inf -> null encoding. Where a success path is needed, the heavy
predict_combined call is monkeypatched so these stay fast and data-free.
"""

import io
import json

import numpy as np
import pytest

import app as backend_app


@pytest.fixture
def client():
    backend_app.app.config["TESTING"] = True
    with backend_app.app.test_client() as c:
        yield c


def _valid_csv_bytes(seconds=12.0, sr=360):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    sig = np.sin(2 * np.pi * 1.2 * t)
    return ("\n".join(f"{v:.5f}" for v in sig)).encode("utf-8")


# ---- /api/health --------------------------------------------------------

def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


# ---- /api/analyze validation matrix ------------------------------------

def test_missing_file_returns_400(client):
    resp = client.post("/api/analyze", data={})
    assert resp.status_code == 400
    assert "file" in resp.get_json()["error"].lower()


def test_empty_filename_returns_400(client):
    data = {"file": (io.BytesIO(b"0.1\n0.2\n"), "")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_non_integer_sampling_rate_returns_400(client):
    data = {
        "file": (io.BytesIO(_valid_csv_bytes()), "ecg.csv"),
        "sampling_rate": "abc",
    }
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "integer" in resp.get_json()["error"].lower()


@pytest.mark.parametrize("bad_sr", ["0", "-1", "999", "44100"])
def test_out_of_range_sampling_rate_returns_400(client, bad_sr):
    # A1: values outside the allow-list must be rejected before check_signal_quality
    # can hit the len(signal)/sr divide (sr=0 -> ZeroDivisionError -> 500).
    data = {
        "file": (io.BytesIO(_valid_csv_bytes()), "ecg.csv"),
        "sampling_rate": bad_sr,
    }
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_empty_numeric_data_returns_400(client):
    data = {"file": (io.BytesIO(b"# only a comment\n"), "ecg.csv")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_short_recording_fails_quality_gate(client):
    data = {"file": (io.BytesIO(b"0.1\n0.2\n0.3\n"), "ecg.csv")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "short" in resp.get_json()["error"].lower()


# ---- A2: NaN/Inf must serialise as null, never the bare token NaN --------

def test_nan_in_report_serialises_as_null(client, monkeypatch):
    canned = {
        "diagnosis": "Normal", "confidence": float("nan"),
        "severity": "NONE", "recommendation": "ok", "warnings": [],
        "method": "clinical_rules",
        "metrics": {"hr_mean": float("inf"), "rr_cv": 0.1},
        "ml_probabilities": {}, "disclaimer": "x", "error": None,
    }
    monkeypatch.setattr(backend_app, "predict_combined", lambda sig, sr, model=None: canned)

    data = {"file": (io.BytesIO(_valid_csv_bytes()), "ecg.csv"), "sampling_rate": "360"}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    raw = resp.get_data(as_text=True)
    assert "NaN" not in raw and "Infinity" not in raw   # valid JSON, no bare tokens
    body = json.loads(raw)                                # strict parse must succeed
    assert body["confidence"] is None
    assert body["metrics"]["hr_mean"] is None
    assert body["metrics"]["rr_cv"] == 0.1
