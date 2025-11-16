#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import pandas as pd

# --- Directory resolution (same style you showed) ---
NB_DIR = Path.cwd().resolve()
SRC_DIR = None
for p in [NB_DIR] + list(NB_DIR.parents):
    candidate = p / "assignment" / "src"
    if candidate.exists():
        SRC_DIR = candidate
        break
    candidate = p / "src"
    if candidate.exists():
        SRC_DIR = candidate
        break

if SRC_DIR is None:
    raise FileNotFoundError(
        f"Could not find a 'src' directory above {NB_DIR}. "
        "If your project root is at e.g. .../assignment/, open the script there or set SRC_DIR manually."
    )

DATA_DIR = SRC_DIR / "data"

# --- CLI ---
parser = argparse.ArgumentParser(
    description="Normalize and enrich ArtifactOntology definitions (LLM preprocess)."
)
parser.add_argument(
    "--input", "-i",
    default=str(DATA_DIR / "ArtifactOntology-definitions.xlsx"),
    help="Input Excel file (default: data/ArtifactOntology-definitions.xlsx)"
)
parser.add_argument(
    "--output", "-o",
    default=str(DATA_DIR / "definitions_enriched.csv"),
    help="Output CSV file (default: data/definitions_enriched.csv)"
)
args = parser.parse_args()

in_path = Path(args.input)
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

# --- Load input (case-insensitive columns) ---
df = pd.read_excel(in_path, engine="openpyxl")

# Standardize columns to lowercase once so downstream selections are stable
df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)

# Ensure expected columns exist (after lowercasing)
expected_cols = {"iri", "label", "definition"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Missing expected columns: {sorted(missing)} in {in_path}. "
        f"Found columns: {sorted(df.columns)}"
    )

# --- Enrichment logic (unchanged from your current script) ---

# Simple dictionaries for abbreviation expansion and typo fixes
abbr_map = [
    (r"\bAC\b", "alternating current"),
    (r"\bDC\b", "direct current"),
    (r"\blbs\.?\b", "pounds"),
    (r"\blb\.?\b", "pound"),
    (r"\bmm\b", "millimeters"),
    (r"\bUHF\b", "ultra high frequency"),
    (r"\bVHF\b", "very high frequency"),
    (r"\bGHz\b", "gigahertz"),
    (r"\bMHz\b", "megahertz"),
    (r"\bi\.e\.\b", "that is"),
    (r"\be\.g\.\b", "for example"),
]
typo_map = {
    'communiction': 'communication',
    'reflfecting': 'reflecting',
    'relflecting': 'reflecting',
    'continous': 'continuous',
    'Electornic': 'Electronic',
    'defract': 'diffract',
    'spefic': 'specific',
    'specifc': 'specific',
    'Celcius': 'Celsius',
    'Hyrdraulic': 'Hydraulic',
    'electornic': 'electronic',
    'increainsg': 'increasing',
    'lable': 'label',
    'Combusion': 'Combustion',
}

# Words where an initial vowel does not imply 'an' and vice versa (simplified exceptions)
use_an_exceptions = {'hour', 'honest', 'honor', 'heir'}
use_a_exceptions = {'university', 'unit', 'user', 'euro', 'one', 'ubiquitous'}

def choose_article(genus: str) -> str:
    g = genus.strip()
    if not g:
        return 'a'
    first_word = g.split()[0]
    fw_lower = first_word.lower()
    if fw_lower in use_a_exceptions:
        return 'a'
    if fw_lower in use_an_exceptions:
        return 'an'
    return 'an' if re.match(r'^[aeiou]', fw_lower) else 'a'

# Basic cleaner
def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply targeted typo fixes
def fix_typos(text: str) -> str:
    for bad, good in typo_map.items():
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text)
    return text

# Expand abbreviations conservatively
def expand_abbr(text: str) -> str:
    for pat, rep in abbr_map:
        text = re.sub(pat, rep, text)
    # Normalize decimal units like '12.7mm' -> '12.7 millimeters'
    text = re.sub(r"(\d+(?:\.\d+)?)\s?mm\b", r"\1 millimeters", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?lbs\b", r"\1 pounds", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?lb\b", r"\1 pound", text)
    return text

# Ensure sentence ends with a period
def ensure_period(text: str) -> str:
    text = text.strip()
    if not text.endswith('.'):
        text += '.'
    return text

# Convert definition into canonical form: “X is a Y that Zs”
def canonicalize_definition(label: str, definition: str) -> str:
    if not isinstance(definition, str):
        return definition
    d = definition.strip()

    # Fix typos and abbreviations early for better parsing
    d = fix_typos(d)
    d = expand_abbr(d)
    d = normalize_whitespace(d)

    # 1) Starts with A/An <Genus> that/which <Rest>
    m = re.match(r"^(A|An)\s+(.+?)\s+(that|which)\s+(.*)$", d, flags=re.IGNORECASE)
    if m:
        genus = m.group(2).strip()
        rest = m.group(4).strip()
        article = choose_article(genus)
        out = f"{label} is {article} {genus} that {rest}"
        return ensure_period(out)

    # 2) Starts with <Genus> that/which <Rest> (no initial article)
    m = re.match(r"^([A-Z][^,.;]+?)\s+(that|which)\s+(.*)$", d)
    if m:
        genus = m.group(1).strip()
        rest = m.group(3).strip()
        article = choose_article(genus)
        out = f"{label} is {article} {genus} that {rest}"
        return ensure_period(out)

    # 3) A/An <Genus> consists/consisting of ...
    m = re.match(r"^(A|An)\s+(.+?)\s+(consists?\b|consisting\b)\s+(.*)$", d, flags=re.IGNORECASE)
    if m:
        genus = m.group(2).strip()
        rest = m.group(3).lower() + ' ' + m.group(4).strip()
        article = choose_article(genus)
        out = f"{label} is {article} {genus} that {rest}"
        return ensure_period(out)

    # 4) If definition already starts with the label
    if re.match(rf"^{re.escape(label)}\b", d):
        return ensure_period(d)

    # 5) A/An <Genus> of/for/in ...
    m = re.match(r"^(A|An)\s+(.+)$", d)
    if m:
        genus_phrase = m.group(2).strip()
        primary_genus = genus_phrase.split(' of ')[0].split(' for ')[0].split(' in ')[0]
        article = choose_article(primary_genus)
        out = f"{label} is {article} {genus_phrase}"
        return ensure_period(out)

    # Fallback
    out = f"{label} is described as: {d}"
    return ensure_period(out)

# --- Build enriched definitions ---
out_rows = []
for _, row in df.iterrows():
    iri = row.get('iri', '')
    label = row.get('label', '')
    definition = row.get('definition', '')
    if pd.isna(label):
        label = ''
    if pd.isna(definition):
        definition = ''
    enriched = canonicalize_definition(str(label), str(definition))
    out_rows.append({
        'iri': iri,
        'label': label,
        'definition_enriched': enriched
    })

out_df = pd.DataFrame(out_rows)

# --- Write CSV ---
out_df.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")