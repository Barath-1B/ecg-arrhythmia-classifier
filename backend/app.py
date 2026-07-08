# -*- coding: utf-8 -*-
"""
ECG Monitor - Flask Backend API
================================
Wraps the existing Python ECG analysis pipeline behind two HTTP endpoints:

  GET  /api/health    -- liveness check
  POST /api/analyze   -- analyze an uploaded ECG CSV file

Usage
-----
    cd backend
    pip install flask flask-cors
    python app.py

The server starts on http://127.0.0.1:5000
"""

import os
import sys
import io
import uuid
import math
import json
import warnings
import datetime

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — make src/ importable from project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from ecg_analyzer import predict_combined, check_signal_quality
from config import (
    DEFAULT_SAMPLING_RATE, PATIENT_ID_DEFAULT, MODEL_FILENAME,
    ALLOWED_SAMPLING_RATES, MAX_SIGNAL_SAMPLES, CORS_ORIGINS,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Load the trained model once at startup
MODEL_PATH = os.path.join(_ROOT, "models", MODEL_FILENAME)
_model = None
try:
    import joblib
    if os.path.isfile(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        print(f"[startup] Model loaded: {MODEL_PATH}")
    else:
        print(f"[startup] Model not found at {MODEL_PATH} — rules-only mode")
except Exception as e:
    print(f"[startup] Could not load model: {e}")


# ---------------------------------------------------------------------------
# JSON encoder that converts numpy types and NaN/Inf → None
# ---------------------------------------------------------------------------
class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy integer types
        if isinstance(obj, (np.integer,)):
            return int(obj)
        # Handle numpy floating types and plain Python floats
        if isinstance(obj, (np.floating, float)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _sanitize(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None.

    _SafeEncoder.default() is only invoked for types the C encoder can't handle;
    a plain Python float('nan') is emitted as the bare token `NaN`, which is
    invalid JSON. Pre-cleaning here (plus allow_nan=False below as a backstop)
    guarantees every float in the payload is finite or null.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _json_response(data, status=200):
    # allow_nan=False makes a stray non-finite float raise rather than silently
    # emit invalid JSON; _sanitize should already have removed them.
    return app.response_class(
        response=json.dumps(_sanitize(data), cls=_SafeEncoder, allow_nan=False),
        status=status,
        mimetype="application/json",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return _json_response({
        "status": "ok",
        "model_loaded": _model is not None,
        "timestamp": datetime.datetime.now().isoformat(),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    # ---- Parse request ------------------------------------------------
    patient_id = request.form.get("patient_id", PATIENT_ID_DEFAULT).strip() or PATIENT_ID_DEFAULT
    try:
        sr = int(request.form.get("sampling_rate", DEFAULT_SAMPLING_RATE))
    except ValueError:
        return _json_response({"error": "sampling_rate must be an integer"}, 400)

    if sr not in ALLOWED_SAMPLING_RATES:
        return _json_response({
            "error": f"sampling_rate must be one of {list(ALLOWED_SAMPLING_RATES)}"
        }, 400)

    if "file" not in request.files:
        return _json_response({"error": "No file uploaded. Send field name 'file'."}, 400)

    file = request.files["file"]
    if file.filename == "":
        return _json_response({"error": "Empty filename"}, 400)

    # ---- Load CSV (same parse as ecg_analyzer.load_ecg_csv) --------------
    try:
        raw = file.read().decode("utf-8", errors="replace")
        signal = np.atleast_1d(np.loadtxt(io.StringIO(raw), delimiter=",", comments="#"))
        if signal.ndim > 1:
            signal = signal[:, 0]   # multi-column CSV: take first column
        signal = signal.astype(np.float64)
    except Exception as e:
        return _json_response({"error": f"File parse error: {e}"}, 400)

    if signal.size == 0:
        return _json_response({"error": "File is empty or contains no numeric data"}, 400)

    # Bound the compute per request: truncate an oversized signal so the DSP and
    # entropy can't run over millions of samples and tie up the worker.
    if signal.size > MAX_SIGNAL_SAMPLES:
        print(f"[analyze] Signal truncated: {signal.size} -> {MAX_SIGNAL_SAMPLES} samples")
        signal = signal[:MAX_SIGNAL_SAMPLES]

    # ---- Signal quality check -------------------------------------------
    quality_err = check_signal_quality(signal, sr)
    if quality_err:
        return _json_response({"error": quality_err}, 400)

    # ---- Run analysis ---------------------------------------------------
    try:
        report = predict_combined(signal, sr, model=_model)
    except ValueError as e:
        # Log for debugging but don't expose details to client
        print(f"[analyze] ValueError: {e}")
        return _json_response({"error": "Signal processing error: invalid data format"}, 400)
    except Exception as e:
        # Log but return generic message to avoid exposing internals
        print(f"[analyze] Unexpected error: {e}")
        return _json_response({"error": "Analysis failed. Please try again."}, 500)

    if report.get("error"):
        return _json_response(report, 400)

    # ---- Enrich & return ------------------------------------------------
    report["patient_id"] = patient_id
    report["analysis_id"] = str(uuid.uuid4())
    return _json_response(report)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nECG API starting on http://127.0.0.1:5000")
    print(f"Model: {'loaded' if _model else 'NOT loaded (rules-only)'}\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
