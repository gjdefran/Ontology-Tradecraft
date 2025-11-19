#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import pandas as pd

# --- Directory resolution (compatible with original) ---
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
    # Fallback to current folder
    SRC_DIR = NB_DIR

DATA_DIR = SRC_DIR / "data"

# --- CLI ---
parser = argparse.ArgumentParser(
    description=(
        "Normalize and enrich ArtifactOntology definitions (LLM preprocess) "
        "with targeted corrections for ambiguous/inaccurate definitions."
    )
)
parser.add_argument(
    "--input", "-i",
    default=str(DATA_DIR / "ArtifactOntology-definitions.xlsx"),
    help="Input Excel file (default: data/ArtifactOntology-definitions.xlsx)"
)
parser.add_argument(
    "--output", "-o",
    default=str(DATA_DIR / "definitions_enriched.csv"),
    help="Output CSV file (default: data/definitions_enriched_fixed.csv)"
)
args = parser.parse_args()

in_path = Path(args.input)
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

# --- Load input (case-insensitive columns) ---
df = pd.read_excel(in_path, engine="openpyxl")
df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
expected_cols = {"iri", "label", "definition"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Missing expected columns: {sorted(missing)} in {in_path}. "
        f"Found columns: {sorted(df.columns)}"
    )

# --- Abbreviation and typo expansions (kept from baseline) ---
abbr_map = [
    (r"AC", "alternating current"),
    (r"DC", "direct current"),
    (r"lbs\.?", "pounds"),
    (r"lb\.?", "pound"),
    (r"mm", "millimeters"),
    (r"UHF", "ultra high frequency"),
    (r"VHF", "very high frequency"),
    (r"GHz", "gigahertz"),
    (r"MHz", "megahertz"),
    (r"i\.e\.", "that is"),
    (r"e\.g\.", "for example"),
]

# Baseline typo map with extra terms

# NOTE: we intentionally DO NOT change the class/kind of any entity; all fixes are
# phrased to better match the label without altering ontological commitments.

typo_map = {
    'communiction': 'communication',
    'reflfecting': 'reflecting',
    'relflecting': 'reflecting',
    'continous': 'continuous',
    'Electornic': 'Electronic',
    'defract': 'diffract',
    'specifc': 'specific',
    'spefic': 'specific',
    'Celcius': 'Celsius',
    'Hyrdraulic': 'Hydraulic',
    'electornic': 'electronic',
    'increainsg': 'increasing',
    'lable': 'label',
    'Combusion': 'Combustion',
    'tired wheels': 'wheels',  # generic typo fix used in several vehicle defs
    'transmiting': 'transmitting',
    'fiat object part': 'flat object part',
}

use_an_exceptions = {'hour', 'honest', 'honor', 'heir'}
use_a_exceptions  = {'university', 'unit', 'user', 'euro', 'one', 'ubiquitous'}

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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def fix_typos(text: str) -> str:
    t = str(text)
    for bad, good in typo_map.items():
        t = re.sub(rf"{re.escape(bad)}", good, t)
    return t


def expand_abbr(text: str) -> str:
    t = str(text)
    for pat, rep in abbr_map:
        t = re.sub(pat, rep, t)
    # normalize common inline units
    t = re.sub(r"(\d+(?:\.\d+)?)\s?mm", r"\1 millimeters", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s?lbs", r"\1 pounds", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s?lb", r"\1 pound", t)
    return t


def ensure_period(text: str) -> str:
    text = str(text).strip()
    return text if not text or text.endswith('.') else text + '.'


def canonicalize_definition(label: str, definition: str) -> str:
    """Bring into the canonical "X is a Y that Z" style where possible."""
    if not isinstance(definition, str):
        return definition
    d = definition.strip()
    d = fix_typos(expand_abbr(normalize_whitespace(d)))

    m = re.match(r"^(A|An)\s+(.+?)\s+(that|which)\s+(.+)$", d, flags=re.IGNORECASE)
    if m:
        genus = m.group(2).strip()
        rest  = m.group(4).strip()
        article = choose_article(genus)
        out = f"{label} is {article} {genus} that {rest}"
        return ensure_period(out)

    m = re.match(r"^([A-Z][^,.;]+?)\s+(that|which)\s+(.+)$", d)
    if m:
        genus = m.group(1).strip()
        rest  = m.group(3).strip()
        article = choose_article(genus)
        out = f"{label} is {article} {genus} that {rest}"
        return ensure_period(out)

    m = re.match(r"^(A|An)\s+(.+?)\s+(consists?|consisting)\s+(.+)$", d, flags=re.IGNORECASE)
    if m:
        genus = m.group(2).strip()
        rest  = m.group(3).lower() + ' ' + m.group(4).strip()
        article = choose_article(genus)
        out = f"{label} is {article} {genus} that {rest}"
        return ensure_period(out)

    if re.match(rf"^{re.escape(label)}", d):
        return ensure_period(d)

    m = re.match(r"^(A|An)\s+(.+)$", d)
    if m:
        genus_phrase = m.group(2).strip()
        primary_genus = genus_phrase.split(' of ')[0].split(' for ')[0].split(' in ')[0]
        article = choose_article(primary_genus)
        return ensure_period(f"{label} is {article} {genus_phrase}")

    return ensure_period(f"{label} is described as: {d}")

# --- Targeted corrections for egregious ambiguous definitions ---
# IMPORTANT: Do not change class. Only clarify scope/wording to better match label.

CORRECTIONS = {
    # Vehicles / wheels typo & scope
    "Automobile": (
        "A Ground Motor Vehicle that is designed to transport a small number of passengers on roads."),
    "Bus": (
        "A Ground Motor Vehicle that is designed to transport many passengers on roads; wheel and axle configurations may vary."),
    "Motorcycle": (
        "A Ground Motor Vehicle that is designed primarily to transport one or two passengers on two wheels; some variants include a sidecar or a third wheel."),

    # Motion control / brakes beyond vehicles
    "Brake": (
        "A Material Artifact that is designed to inhibit motion in a moving system by absorbing or dissipating energy (e.g., in vehicles or machinery)."),

    # Transport infrastructure elements
    "Bridge": (
        "A Land Transportation Artifact that is designed to span physical obstacles such as land or water while maintaining passage beneath, enabling persons and vehicles to pass over the obstacle."),
    "Tunnel": (
        "A Land Transportation Artifact that is designed to enable ground vehicles to travel beneath surrounding soil, rock, or a water body (including riverbeds or seabeds)."),
    "Trail": (
        "A Land Transportation Artifact that is designed to enable pedestrian or non-motorized travel through natural or constructed environments, including forests, moors, or urban greenways."),
    "Transportation Infrastructure": (
        "An Infrastructure System that has continuant part one or more Transportation Artifacts and bears a function that, if realized, is realized in acts of transportation of passengers and/or cargo."),

    # Thermal / heat sink scope
    "Heat Sink": (
        "A Material Artifact that is designed to passively dissipate heat from components or systems in order to control temperature."),

    # Hydraulic PTU generalization without changing class
    "Hydraulic Power Transfer Unit": (
        "A Material Artifact that is designed to transfer hydraulic power between hydraulic systems; in aircraft, it can transfer power between independent systems when one has failed or is offline."),

    # System clock embodiment broader than "computer" only
    "Material Copy of a System Clock": (
        "A Material Copy of a Timekeeping Instrument that is part of a computing device or embedded system and is designed to issue a steady high-frequency signal to synchronize internal components."),

    # Railcar terminology
    "Railcar": (
        "A Rail Transport Vehicle that consists of a single vehicle designed to carry passengers or cargo; it may be unpowered (hauled) or self-propelled and may couple with other railcars to form a train."),

    # Telecom wording fixes
    "Telecommunication Network": (
        "A Communication System that is designed to enable the transmission of information between telecommunication endpoints via interconnected network nodes and lines."),
    "Public Address System": (
        "A Communication System that is designed to amplify and distribute audio so a speaker can be heard by multiple listeners within a shared space."),
    "Wired Communication Artifact Function": (
        "A Communication Artifact Function that is realized in a process that conveys meaningful signs via physical wired media, such as metallic conductors or optical fiber."),

    

    # Weapons / RPG target scope
    "Rocket-Propelled Grenade": (
        "An unguided, shoulder-fired rocket with an explosive warhead that is designed to be used against armored and material targets."),

    # Barcode 2D modules
    "Material Copy of a Two-Dimensional Barcode": (
        "A Material Copy of a Barcode that is designed to bear modules (cells) arranged in a two-dimensional pattern that concretize some Directive Information Content Entity."),

    # Grammar / number fixes
    "Complex Optical Lens": (
        "An Optical Lens consisting of more than one Simple Optical Lens."),
    "Coin": (
        "A Portion of Cash that consists of a flat, portable, round piece of metal designed to bear some specified Financial Value."),
    "Banknote": (
        "A Portion of Cash that consists of a portable slip of paper or fabric designed to bear some specified Financial Value."),

    # RF band instruments: make band membership explicit
    "High Frequency Communication Instrument": (
        "A Radio Communication Instrument that is designed to transmit and/or receive radio signals in the high-frequency (HF) band."),
    "Very High Frequency Communication Instrument": (
        "A Radio Communication Instrument that is designed to transmit and/or receive radio signals in the very high frequency (VHF) band."),
    "Ultra High Frequency Communication Instrument": (
        "A Radio Communication Instrument that is designed to transmit and/or receive radio signals in the ultra high frequency (UHF) band."),

    # Propelling nozzle mechanism wording
    "Propelling Nozzle": (
        "A Nozzle that is designed to accelerate and shape engine exhaust to produce thrust as part of a reaction (jet) engine, typically by choking flow at the throat and expanding the exhaust to form a high-speed jet."),
}

# Pattern-level tweaks for recurrent phrasing (kept conservative)

def apply_pattern_fixes(label: str, definition: str) -> str:
    d = definition
    # Remove vague quantifier "significant distances" in telecom contexts
    if label in {"Telecommunication Network", "Telecommunication Instrument"}:
        d = re.sub(r"significant distances?", "long or short distances", d)
    # Replace parenthetical "(usually ...)" qualifiers that weaken definition
    d = re.sub(r"\(usually[^)]+\)\s*", "", d)
    return d


def apply_corrections(label: str, definition: str):
    """
    Returns: (corrected_definition, applied_flag, reason)
    """
    # If a targeted correction exists, apply it exactly.
    if label in CORRECTIONS:
        return CORRECTIONS[label], True, "targeted_correction"
    # Otherwise apply safe pattern tweaks
    tweaked = apply_pattern_fixes(label, definition)
    if tweaked != definition:
        return tweaked, True, "pattern_fix"
    return definition, False, "none"

# --- Build enriched + corrected definitions ---
rows = []
for _, row in df.iterrows():
    iri = row.get('iri', '')
    label = str(row.get('label', ''))
    definition = str(row.get('definition', ''))
    if pd.isna(label):
        label = ''
    if pd.isna(definition):
        definition = ''

    # First normalize to canonical style
    enriched = canonicalize_definition(label, definition)
    # Then apply corrections (without changing class)
    corrected, applied, reason = apply_corrections(label, enriched)

    rows.append({
        'iri': iri,
        'label': label,
        'definition_original': definition,
        'definition_enriched': corrected,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")
