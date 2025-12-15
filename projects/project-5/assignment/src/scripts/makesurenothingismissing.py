import pandas as pd
from rdflib import Graph, RDF

# === CONFIGURATION ===
TTL_FILE = "C:/Users/gregd/Documents/UB/ontology tradecraft/Ontology-Tradecraft/projects/project-5/assignment/src/ArtifactOntology.ttl"  # Path to your ontology TTL file
CSV_FILE = "C:/Users/gregd/Documents/UB/ontology tradecraft/Ontology-Tradecraft/projects/project-5/assignment/src/data/ArtifactOntology-definitions.xlsx"  # Path to your CSV file with iri, label, definition
CCO_NAMESPACE = "https://www.commoncoreontologies.org/"  # Namespace for CCO classes
OUTPUT_FILE = "comparison_results.txt"  # File to save comparison results

# === STEP 1: Parse TTL and extract all CCO class IRIs ===
g = Graph()
g.parse(TTL_FILE)

# Extract all subjects that are classes and start with CCO namespace
classes_in_ttl = set(str(s) for s in g.subjects(RDF.type, None) if str(s).startswith(CCO_NAMESPACE))

# === STEP 2: Load Excel and extract IRIs ===
df = pd.read_excel(CSV_FILE)
csv_iris = set(df['iri'].dropna())

# === STEP 3: Compare sets ===
missing_in_csv = classes_in_ttl - csv_iris
extra_in_csv = csv_iris - classes_in_ttl

# === STEP 4: Print summary ===
print(f"Total classes in TTL: {len(classes_in_ttl)}")
print(f"Total IRIs in CSV: {len(csv_iris)}")
print(f"Missing in CSV: {len(missing_in_csv)}")
print(f"Extra in CSV: {len(extra_in_csv)}")

# === STEP 5: Save details to file ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(f"Total classes in TTL: {len(classes_in_ttl)}\n")
    f.write(f"Total IRIs in CSV: {len(csv_iris)}\n")
    f.write(f"Missing in CSV ({len(missing_in_csv)}):\n")
    for iri in sorted(missing_in_csv):
        f.write(f"  {iri}\n")
    f.write(f"\nExtra in CSV ({len(extra_in_csv)}):\n")
    for iri in sorted(extra_in_csv):
        f.write(f"  {iri}\n")

print(f"Comparison results saved to {OUTPUT_FILE}")