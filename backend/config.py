# Backend configuration constants

import os

# ECG analysis defaults
DEFAULT_SAMPLING_RATE = 360

# Sampling rates the API accepts (must mirror the frontend UploadForm select).
# Anything outside this set is rejected with a 400 so bad values can't reach the
# len(signal)/sr divide in check_signal_quality (sr=0 -> ZeroDivisionError -> 500).
ALLOWED_SAMPLING_RATES = (256, 360, 500)

# Upper bound on samples analysed per request. A valid-but-huge numeric CSV that
# slips under MAX_CONTENT_LENGTH would otherwise run the O(N) DSP and entropy over
# millions of samples and tie up a worker. Signals longer than this are truncated.
# 500 Hz * 600 s = 300k samples (10 min at the highest supported rate).
MAX_SIGNAL_SAMPLES = 300_000

# Allowed CORS origins for the browser frontend. Override in deployment via the
# ECG_CORS_ORIGINS env var (comma-separated).
CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ECG_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",") if o.strip()
]

# Patient tracking
PATIENT_ID_DEFAULT = 'UNKNOWN'

# Model paths
MODEL_FILENAME = 'ecg_model.joblib'
