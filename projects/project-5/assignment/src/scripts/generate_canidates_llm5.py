
import re
import csv

# File paths
ontology_file = "../cco_merged.ttl"
definitions_file = "../data/definitions_enriched.csv"
output_file = "../generated/generated_axioms.ttl"

# Prefixes for TTL
prefixes = """@prefix : <https://www.commoncoreontologies.org/CommonCoreOntologiesMerged/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix obo: <http://purl.obolibrary.org/obo/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix cco: <https://www.commoncoreontologies.org/> .

"""

# Ontology properties and classes IRIs
bearer_of = "http://purl.obolibrary.org/obo/BFO_0000196"
inheres_in = "http://purl.obolibrary.org/obo/BFO_0000197"
realized_by = "http://purl.obolibrary.org/obo/BFO_0000054"
has_participant = "http://purl.obolibrary.org/obo/BFO_0000057"
process_class = "http://purl.obolibrary.org/obo/BFO_0000015"
artifact_function_class = "cco:ont00000323"
material_artifact_class = "cco:ont00000995"

# Regex patterns
class_pattern = re.compile(r"^(cco:[^\s]+) rdf:type owl:Class")
label_pattern = re.compile(r"rdfs:label \"(.+?)\"@en")
subclass_pattern = re.compile(r"rdfs:subClassOf ([^\s]+)")

# Parse ontology classes and labels
classes = {}
subclasses = {}
current_class = None

with open(ontology_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        class_match = class_pattern.search(line)
        if class_match:
            current_class = class_match.group(1)
            classes[current_class] = {"label": None}
            continue
        if current_class:
            label_match = label_pattern.search(line)
            if label_match:
                classes[current_class]["label"] = label_match.group(1)
            subclass_match = subclass_pattern.search(line)
            if subclass_match:
                subclasses.setdefault(current_class, []).append(subclass_match.group(1))

# Identify Artifact Function IRI
artifact_function_iri = None
for iri, data in classes.items():
    if data["label"] and data["label"].lower() == "artifact function":
        artifact_function_iri = iri
        break

# Identify function classes: subclasses of Artifact Function or label contains 'function'
function_classes = set()
for iri, data in classes.items():
    label = data.get("label")
    if label and "function" in label.lower():
        function_classes.add(iri)

# Also add subclasses of Artifact Function recursively
def is_subclass_of(iri, target_iri, visited=None):
    if visited is None:
        visited = set()
    if iri == target_iri:
        return True
    if iri in visited:
        return False
    visited.add(iri)
    for parent in subclasses.get(iri, []):
        if is_subclass_of(parent, target_iri, visited):
            return True
    return False

for iri in list(classes.keys()):
    if artifact_function_iri and is_subclass_of(iri, artifact_function_iri):
        function_classes.add(iri)

# Read definitions CSV
artifacts = []
with open(definitions_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iri = row.get("iri") or row.get("IRI")
        label = row.get("label") or row.get("Label")
        definition = row.get("definition_enriched") or row.get("Definition")
        if iri and label and definition:
            artifacts.append({
                "iri": iri.strip(),
                "label": label.strip(),
                "definition": definition.strip()
            })

# Helper to format IRIs for TTL output
def format_iri(iri):
    if iri.startswith("http://") or iri.startswith("https://"):
        return f"<{iri}>"
    return iri  # prefixed names like cco:ont00000323

# Extract superclass label from definition (pattern: 'is a/an Y')
def extract_superclass(definition):
    match = re.search(r"is an? ([A-Za-z \-]+?)(?: that| which| who|$|\.)", definition)
    if match:
        return match.group(1).strip()
    return None

# Detect function phrase in definition (simple heuristic)
def detect_function_phrase(definition):
    patterns = [
        r"designed to ([a-zA-Z \-]+?)(?:\.|,|;|$)",
        r"used to ([a-zA-Z \-]+?)(?:\.|,|;|$)",
        r"functions by ([a-zA-Z \-]+?)(?:\.|,|;|$)",
        r"intended to ([a-zA-Z \-]+?)(?:\.|,|;|$)"
    ]
    for pat in patterns:
        match = re.search(pat, definition.lower())
        if match:
            return match.group(1).strip()
    return None

# Match function phrase to function class label (token overlap)
def match_function_label(phrase):
    phrase_tokens = set(phrase.lower().split())
    best_match = None
    best_score = 0
    for iri in function_classes:
        label = classes.get(iri, {}).get("label")
        if not label:
            continue
        label_tokens = set(label.lower().split())
        score = len(phrase_tokens.intersection(label_tokens))
        if score > best_score:
            best_score = score
            best_match = iri
    if best_score > 0:
        return best_match
    return None

# Generate axioms
axioms = [prefixes]

for artifact in artifacts:
    iri = artifact["iri"]
    label = artifact["label"]
    definition = artifact["definition"]

    # Add label annotation
    axioms.append(f"{format_iri(iri)} rdfs:label \"{label}\"@en .")

    # Extract superclass and map to IRI if possible
    superclass_label = extract_superclass(definition)
    if superclass_label:
        superclass_iri = None
        # Try to find superclass IRI by label (case-insensitive)
        for c_iri, c_data in classes.items():
            if c_data.get("label") and c_data["label"].lower() == superclass_label.lower():
                superclass_iri = c_iri
                break
        if superclass_iri:
            axioms.append(f"{format_iri(iri)} rdfs:subClassOf {format_iri(superclass_iri)} .")
        else:
            # Default to Material Artifact if not found
            axioms.append(f"{format_iri(iri)} rdfs:subClassOf {material_artifact_class} .")
    else:
        axioms.append(f"{format_iri(iri)} rdfs:subClassOf {material_artifact_class} .")

    # Detect function phrase and match to function class
    func_phrase = detect_function_phrase(definition)
    if func_phrase:
        func_iri = match_function_label(func_phrase)
        if func_iri:
            # Function bearer axiom
            axioms.append(
                f"{format_iri(iri)} rdfs:subClassOf [ a owl:Restriction ; owl:onProperty {format_iri(bearer_of)} ; owl:someValuesFrom {format_iri(func_iri)} ] ."
            )
            # Function inheres_in axiom
            axioms.append(
                f"{format_iri(func_iri)} rdfs:subClassOf [ a owl:Restriction ; owl:onProperty {format_iri(inheres_in)} ; owl:someValuesFrom {format_iri(iri)} ] ."
            )
            # Function subclass of Artifact Function
            axioms.append(
                f"{format_iri(func_iri)} rdfs:subClassOf {artifact_function_class} ."
            )
            # Function realization axiom
            process_restriction = (
                f"[ a owl:Class ; owl:intersectionOf ( {format_iri(process_class)} "
                f"[ a owl:Restriction ; owl:onProperty {format_iri(has_participant)} ; owl:someValuesFrom {format_iri(iri)} ] ) ]"
            )
            axioms.append(
                f"{format_iri(func_iri)} rdfs:subClassOf [ a owl:Restriction ; owl:onProperty {format_iri(realized_by)} ; owl:someValuesFrom {process_restriction} ] ."
            )

# Write axioms to TTL file
with open(output_file, "w", encoding="utf-8") as f:
    for ax in axioms:
        f.write(ax + "\n")

print(f"Generated {len(axioms)} axioms and saved to {output_file}.")
