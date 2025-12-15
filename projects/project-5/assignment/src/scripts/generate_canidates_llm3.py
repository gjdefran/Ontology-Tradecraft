
import csv
import re

# Input and output files
input_csv = "../data/definitions_enriched.csv"
output_ttl = "../generated/generated_axioms.ttl"

# Prefixes for TTL output
prefixes = """@prefix : <https://www.commoncoreontologies.org/CommonCoreOntologiesMerged/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix obo: <http://purl.obolibrary.org/obo/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

"""

# Ontology properties and classes IRIs
bearer_of = "obo:BFO_0000196"    # bearer_of
inheres_in = "obo:BFO_0000197"    # inheres_in
realized_by = "obo:BFO_0000054"   # realized_by (corrected to BFO_0000054)
has_participant = "obo:BFO_0000057"  # has_participant
process_class = "obo:BFO_0000015"     # process
artifact_function_class = "cco:ont00000323"  # Artifact Function
material_artifact_class = "cco:ont00000995"  # Material Artifact

# Helper: create Turtle restriction block
def create_restriction(on_property, some_values_from):
    return f"[ a owl:Restriction ; owl:onProperty <{on_property}> ; owl:someValuesFrom <{some_values_from}> ]"

# Extract superclass label from definition (pattern: "X is a Y")
def extract_superclass(definition):
    # Look for 'is a' or 'is an' followed by class name (words and spaces)
    match = re.search(r"is an? ([A-Za-z \-]+?)(?: that| which| who|$|\.)", definition)
    if match:
        return match.group(1).strip()
    return None

# Detect function phrase from definition (verbs like 'designed to', 'used for', 'functions by')
def detect_function_phrase(definition):
    # Try to find phrases like 'designed to [verb phrase]', 'used to [verb phrase]', 'functions by [verb phrase]'
    patterns = [
        r"designed to ([a-zA-Z \-']+?)(?:\.|,|;|$)",
        r"used to ([a-zA-Z \-']+?)(?:\.|,|;|$)",
        r"functions by ([a-zA-Z \-']+?)(?:\.|,|;|$)",
        r"intended to ([a-zA-Z \-']+?)(?:\.|,|;|$)"
    ]
    for pat in patterns:
        match = re.search(pat, definition.lower())
        if match:
            return match.group(1).strip()
    return None

# Map label to IRI format (replace spaces and special chars with underscores)
def label_to_iri(label):
    # Remove apostrophes, replace spaces and hyphens with underscores
    clean_label = label.replace("'", "").replace(" ", "_").replace("-", "_")
    return f"cco:{clean_label}"

# Load all function labels from the ontology for matching (simulate from merged ontology)
# For demo, we will hardcode a small set of known function labels from the ontology
known_functions = {
    "inhibiting motion artifact function": "cco:ont00000937",
    "artifact function": artifact_function_class,
    "motion artifact function": "cco:ont00000448",
    "cleaning artifact function": "cco:ont00000665",
    "heating artifact function": "cco:ont00001013",
    "cooling artifact function": "cco:ont00000002",
    "communication artifact function": "cco:ont00000727",
    "electrical artifact function": "cco:ont00000098",
    "imaging artifact function": "cco:ont00000601",
    "damaging artifact function": "cco:ont00001153",
    "service artifact function": "cco:ont00001301",
    "measurement artifact function": "cco:ont00001100",
    "propulsion artifact function": "cco:ont00001252",
    "bearing artifact function": "cco:ont00001323",
    "inhibiting motion artifact function": "cco:ont00000937",
    # Add more as needed...
}

# Try to match detected function phrase to known function labels (simple substring match)
def match_function_label(phrase):
    phrase = phrase.lower()
    for func_label in known_functions.keys():
        # Check if all words in func_label appear in phrase (loose matching)
        if all(word in phrase for word in func_label.split()):
            return known_functions[func_label]
    return None

# Main processing
axioms = []

with open(input_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        iri = row['iri'].strip()
        label = row['label'].strip()
        definition = row['definition_enriched'].strip()

        # Add label annotation
        axioms.append(f"<{iri}> rdfs:label \"{label}\" .")

        # Extract superclass from definition
        superclass_label = extract_superclass(definition)
        if superclass_label:
            superclass_iri = label_to_iri(superclass_label)
            axioms.append(f"<{iri}> rdfs:subClassOf <{superclass_iri}> .")
        else:
            # Default superclass if none found
            axioms.append(f"<{iri}> rdfs:subClassOf <{material_artifact_class}> .")

        # Detect function phrase and map to function class IRI
        func_phrase = detect_function_phrase(definition)
        if func_phrase:
            func_iri = match_function_label(func_phrase)
            if func_iri:
                # Function bearer axiom: artifact bears function
                axioms.append(f"<{iri}> rdfs:subClassOf [ a owl:Restriction ; owl:onProperty <{bearer_of}> ; owl:someValuesFrom <{func_iri}> ] .")
                # Function inheres_in axiom: function inheres in artifact
                axioms.append(f"<{func_iri}> rdfs:subClassOf [ a owl:Restriction ; owl:onProperty <{inheres_in}> ; owl:someValuesFrom <{iri}> ] .")
                # Function subclass of Artifact Function
                axioms.append(f"<{func_iri}> rdfs:subClassOf <{artifact_function_class}> .")
                # Function realization axiom: function realized by process with artifact participant
                process_restriction = f"[ a owl:Class ; owl:intersectionOf ( <{process_class}> [ a owl:Restriction ; owl:onProperty <{has_participant}> ; owl:someValuesFrom <{iri}> ] ) ]"
                axioms.append(f"<{func_iri}> rdfs:subClassOf [ a owl:Restriction ; owl:onProperty <{realized_by}> ; owl:someValuesFrom {process_restriction} ] .")

# Write axioms to TTL file
with open(output_ttl, 'w', encoding='utf-8') as f:
    f.write(prefixes)
    for ax in axioms:
        f.write(ax + "\n")

print(f"Generated {len(axioms)} axioms and saved to {output_ttl}.")
