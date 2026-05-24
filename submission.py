import os
import sys
import requests
from pathlib import Path
import pandas as pd

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

"""
Example submission script for the Stolen Model Detection Task.

Submission Requirements (read carefully to avoid automatic rejection):

1. CSV FORMAT
----------------
- The file **must be a CSV** with extension `.csv`.
- It must contain **exactly two columns**, named:
      id, score
  → Column names must match exactly (lowercase, no extra spaces).
  → Column order does not matter, but both must be present.

2. ROW COUNT AND IDENTIFIERS
-------------------------------
- Your file must contain **exactly 360 rows**.
- Each row corresponds to one unique `id` in the range **0–359** (inclusive).
- Every id must appear **exactly once**.
- Do **not** add, remove, or rename any IDs.
- Do **not** include duplicates or missing entries.
- The evaluator checks:
      id.min() == 0
      id.max() == 359
      id.unique().size == 360

3. STEALING CONFIDENCE SCORES
----------------------
- The `score` column must contain **numeric values** representing your model’s predicted confidence
  that the corresponding subset is a **stolen** model.

  Examples of valid score values:
    - Probabilities: values in [0.0, 1.0]
    - Raw model scores: any finite numeric values (will be ranked for TPR@FPR=0.05)

- Do **not** submit string labels like "yes"/"no" or "stolen"/"not stolen".
- The evaluator converts your `score` column to numeric using `pd.to_numeric()`.
  → Any non-numeric, NaN, or infinite entries will cause automatic rejection.

4. TECHNICAL LIMITS
----------------------
- Maximum file size: **20 MB**
- Encoding: UTF-8 recommended.
- Avoid extra columns, blank lines, or formulas.
- Ensure all values are numeric and finite.
- Supported data types: int, float (e.g., float32, float64)

5. VALIDATION SUMMARY
------------------------
Your submission will fail if:
- Columns don’t match exactly ("id", "score")
- Row count differs from 360
- Any id is missing, duplicated, or outside [0, 359]
- Any score value is NaN, Inf, or non-numeric
- File is too large or not a valid CSV

One key metric is computed:
  1. **TPR@FPR=0.05 (True Positive Rate at False Positive Rate = 0.05)** 
  — measures the ability to correctly identify stolen models while keeping the false positive rate at 5%.
"""
BASE_URL = "http://34.63.153.158"
API_KEY = "6876cc0a3670a9367156d22daae5f9ea"  # replace with your actual API key

TASK_ID = "19-stolen-model-detection"
FILE_PATH = "submission.csv"  # replace with your actual file path

SUBMIT = True  # Set to True to enable submission

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if SUBMIT:
    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                # (fieldname) -> (filename, fileobj, content_type)
                "file": (os.path.basename(FILE_PATH), f, "csv"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),  # (connect timeout, read timeout)
            )
        # Helpful output even on non-2xx
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        submission_id = body.get("submission_id")
        print("Successfully submitted.")
        print("Server response:", body)
        if submission_id:
            print(f"Submission ID: {submission_id}")

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)
