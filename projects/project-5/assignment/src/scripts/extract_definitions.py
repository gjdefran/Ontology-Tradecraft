from pathlib import Path
import pandas as pd
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDFS, XSD

# Start from the notebook's current directory
NB_DIR = Path.cwd().resolve()

# Walk up to find a directory that contains a 'src' folder
SRC_DIR = None
for p in [NB_DIR] + list(NB_DIR.parents):
    candidate = p / "assignment" / "src"
    if candidate.exists():
        SRC_DIR = candidate
        break
    candidate = p / "src"  # also allow projects that don't have the extra 'assignment' level
    if candidate.exists():
        SRC_DIR = candidate
        break

if SRC_DIR is None:
    raise FileNotFoundError(
        f"Could not find a 'src' directory above {NB_DIR}. "
        "If your project root is at e.g. .../assignment/, open the notebook there or set SRC_DIR manually."
    )

DATA_DIR = SRC_DIR / "data"

print("Notebook CWD:   ", NB_DIR)
print("Resolved SRC_DIR:", SRC_DIR)
print("Resolved DATA_DIR:", DATA_DIR)
print("TTL files found:", [p.name for p in SRC_DIR.glob("*.ttl")])


TTL = {
    "ArtifactOntology": SRC_DIR / "ArtifactOntology.ttl",
}

OUT = {
    "ArtifactOntology":  DATA_DIR / "ArtifactOntology-definitions.xlsx",
}

CLASS_DEF_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX cco:  <https://www.commoncoreontologies.org/>
SELECT DISTINCT ?iri ?label ?definition
WHERE {
  # Candidate resources in the endpoint whose IRI starts with the cco: namespace.
  # Adjust the cco: prefix above if your deployment uses a different base IRI.
  ?iri ?p ?o .
  FILTER( STRSTARTS(STR(?iri), STR(cco:)) )

  # Retrieve a label: prefer skos:prefLabel over rdfs:label if both exist.
  OPTIONAL {
    {
      ?iri skos:prefLabel ?prefLabel .
    } UNION {
      ?iri rdfs:label ?rdfsLabel .
    }
    BIND(COALESCE(?prefLabel, ?rdfsLabel) AS ?rawLabel)
  }

  # Retrieve a definition if present
  OPTIONAL { ?iri skos:definition ?rawDef }

  # Prefer English values when available; otherwise let any language through.
  BIND(
    IF(LANGMATCHES(LANG(?rawLabel), "en") || LANG(?rawLabel) = "", ?rawLabel,
      # If not English, try to pick an English label if there is one
      ?rawLabel
    ) AS ?label
  )
  BIND(
    IF(LANGMATCHES(LANG(?rawDef), "en") || LANG(?rawDef) = "", ?rawDef,
      ?rawDef
    ) AS ?definition
  )
}
ORDER BY LCASE(STR(COALESCE(?label, ""))) STR(?iri)
"""

def run_local_select(ttl_path: Path, query: str) -> pd.DataFrame:
    g = Graph()
    g.parse(ttl_path)  # rdflib guesses Turtle from .ttl
    res = g.query(query)
    rows = []
    for row in res:
        d = row.asdict()  # {'iri': rdflib.term.URIRef(...), 'label': rdflib.term.Literal(...)} or missing
        iri_term   = d.get("iri")
        label_term = d.get("label")
        definition_term =d.get("definition")
        rows.append({
            "iri":   str(iri_term)   if iri_term is not None else None,
            "label": str(label_term) if label_term is not None else None,
            "definition": str(definition_term) if definition_term is not None else None,
        })
    return pd.DataFrame(rows, columns=["iri", "label", "definition",])

for key, ttl_path in TTL.items():
    if not ttl_path.exists():
        print(f"⚠️  Missing {ttl_path} — skipping {key}")
        continue
    print(f"Querying {ttl_path} …")
    df = run_local_select(ttl_path, CLASS_DEF_QUERY)
    df.to_excel(OUT[key], index=False)
    print(f"✅ Saved {len(df)} rows to {OUT[key]}")

