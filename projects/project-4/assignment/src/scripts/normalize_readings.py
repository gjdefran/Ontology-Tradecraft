import pandas as pd
import json
from dateutil import parser as dateparser
from pathlib import Path
import datetime
import os
import argparse
# Determine repo root (adjust depth if needed)
REPO_ROOT = Path(__file__).resolve().parent.parent  # go up one level from script

# Defaults relative to repo
DEFAULT_IN_A = REPO_ROOT / "src" / "data" / "sensor_A.csv"
DEFAULT_IN_B = REPO_ROOT / "src" / "data" / "sensor_B.json"
DEFAULT_OUT = REPO_ROOT / "src" / "data" / "readings_normalized.csv"

# Environment overrides
env_in_a = os.getenv("IN_A", DEFAULT_IN_A)
env_in_b = os.getenv("IN_B", DEFAULT_IN_B)
env_out = os.getenv("OUT", DEFAULT_OUT)

# CLI overrides
parser = argparse.ArgumentParser()
parser.add_argument("--in_a", type=Path, default=env_in_a)
parser.add_argument("--in_b", type=Path, default=env_in_b)
parser.add_argument("--out", type=Path, default=env_out)
args = parser.parse_args()

IN_A, IN_B, OUT = args.in_a, args.in_b, args.out

# The data folder to watch for additional CSVs
DATA_DIR = IN_A.parent

# Files to exclude from auto-loading (already handled or generated)
EXCLUDE_CSV_NAMES = {IN_A.name, OUT.name}

# Column mapping used for "CSV like Sensor A"
# (case-insensitive matching will be applied)
RENAME_MAP_SENSOR_A = {
    "device name": "artifact_id",
    "reading type": "sdc_kind",
    "units": "unit_label",
    "reading value": "value",
    "time (local)": "timestamp",
}
CANONICAL_COLS = ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"]

# -----------------------------
# Helper: load a CSV in "Sensor A" schema
# -----------------------------
def load_csv_like_sensor_a(path: Path) -> pd.DataFrame:
    """
    Reads a CSV that has the same schema as Sensor A and returns a DataFrame
    with columns renamed to the canonical set.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])

    # Build a case-insensitive mapping from original headers -> canonical names
    lower_to_original = {c.lower().strip(): c for c in df.columns}
    rename_dict = {}
    for src_lower, target in RENAME_MAP_SENSOR_A.items():
        if src_lower in lower_to_original:
            rename_dict[lower_to_original[src_lower]] = target

    df = df.rename(columns=rename_dict)
    # Keep only the canonical columns that are present
    keep_cols = [c for c in CANONICAL_COLS if c in df.columns]
    df = df[keep_cols]

    # Quick validation: if too many required columns are missing, warn & skip
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] {path.name}: missing columns after rename: {missing}. "
              f"Rows may be dropped later if critical columns are absent.")
    return df

# -----------------------------
# Load Sensor A (CSV)
# -----------------------------
df_a = load_csv_like_sensor_a(IN_A)

# -----------------------------
# Load Sensor B (JSON) - unchanged logic
# -----------------------------
with open(IN_B, "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for reading in data.get("readings", []):
    entity_id = reading.get("entity_id")
    for entry in reading.get("data", []):
        kind = entry.get("kind")
        value = entry.get("value")
        unit = entry.get("unit")
        timestamp = entry.get("time")
        records.append({
            "artifact_id": entity_id,
            "sdc_kind": kind,
            "unit_label": unit,
            "value": value,
            "timestamp": timestamp
        })

df_b = pd.DataFrame(records)

# -----------------------------
# Auto-load any new CSVs in DATA_DIR that follow Sensor A schema
# -----------------------------
extra_csv_paths = [
    p for p in DATA_DIR.glob("*.csv")
    if p.name not in EXCLUDE_CSV_NAMES
]
extra_frames = []
for p in sorted(extra_csv_paths):
    try:
        df_extra = load_csv_like_sensor_a(p)
        if not df_extra.empty:
            print(f"[INFO] Loaded extra CSV: {p.name} (rows={len(df_extra)})")
            extra_frames.append(df_extra)
        else:
            print(f"[INFO] Skipping {p.name}: no rows after basic load.")
    except Exception as exc:
        print(f"[ERROR] Failed to load {p.name}: {exc}")

# -----------------------------
# Concatenate A + extras + B
# -----------------------------
df_parts = [df_a] + extra_frames + [df_b]
df = pd.concat(df_parts, ignore_index=True)

# -----------------------------
# Trim whitespace + basic normalization
# -----------------------------
for col in ["artifact_id", "sdc_kind", "unit_label"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df["value"] = pd.to_numeric(df["value"], errors="coerce")

# -----------------------------
# Timestamp parsing to ISO 8601 (UTC, Z)
# -----------------------------
def to_iso8601(x):
    try:
        dt = dateparser.parse(str(x))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None

df["timestamp"] = df["timestamp"].apply(to_iso8601)

# -----------------------------
# Unit normalization
# -----------------------------
UNIT_MAP = {
    "celsius": "C", "°c": "C", "c": "C",
    "kilogram": "kg", "kg": "kg",
    "meter": "m", "m": "m",
}

# Preserve original unit if we don't have a mapping
orig_unit = df["unit_label"]
normalized = orig_unit.str.lower().map(UNIT_MAP)
df["unit_label"] = normalized.fillna(orig_unit)

# -----------------------------
# Drop rows with missing critical values
# -----------------------------
df = df.dropna(subset=["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"])

# Optional: remove exact duplicates (if multiple files overlap)
df = df.drop_duplicates(subset=["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"])

# -----------------------------
# Sort for readability
# -----------------------------
df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)

# -----------------------------
# Write output
# -----------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(df)} rows.")