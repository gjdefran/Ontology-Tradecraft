
import csv
import re

# File paths
merged_ontology_file = "../cco_merged.ttl"
definitions_file = "../data/definitions_enriched.csv"
output_ttl_file = "../generated/generated_axioms.ttl"

# Prefixes for TTL output
prefixes = """@prefix : <https://www.commoncoreontologies.org/CommonCoreOntologiesMerged/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix obo: <http://purl.obolibrary.org/obo/> .
@prefix cco: <https://www.commoncoreontologies.org/> .

"""

# Ontology properties and classes IRIs
bearer_of = "http://purl.obolibrary.org/obo/BFO_0000196"  # bearer_of
inheres_in = "http://purl.obolibrary.org/obo/BFO_0000197"  # inheres_in
realized_by = "http://purl.obolibrary.org/obo/BFO_0000054"  # realized_by
has_participant = "http://purl.obolibrary.org/obo/BFO_0000057"  # has_participant
process_class = "http://purl.obolibrary.org/obo/BFO_0000015"  # process
artifact_function_class = "cco:ont00000323"  # Artifact Function
material_artifact_class = "cco:ont00000995"  # Material Artifact

# Helper: format IRI for TTL output
def format_iri(iri):
    # If full IRI (http or https), wrap in <>
    if iri.startswith("http://") or iri.startswith("https://"):
        return f"<{iri}>"
    # Otherwise assume prefixed name, output as is (no <>)
    return iri

# Extract function classes and labels from merged ontology file
def extract_function_classes(ontology_file):
    func_map = {}
    with open(ontology_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        # Look for class declarations with 'Artifact Function' in label or comment
        if 'rdf:type owl:Class' in line:
            # Look ahead for label and definition lines
            label = None
            iri_match = re.search(r'^(?:<([^>]+)>|(\S+))', line.strip())
            iri = None
            if iri_match:
                iri = iri_match.group(1) or iri_match.group(2)
            # Search next few lines for rdfs:label and definition
            for j in range(i+1, min(i+10, len(lines))):
                lab_match = re.search(r'rdfs:label\s+"([^"]+)"@en', lines[j])
                if lab_match:
                    label = lab_match.group(1).strip()
                def_match = re.search(r'skos:definition\s+"([^"]+)"@en', lines[j])
                if def_match and label and 'artifact function' in def_match.group(1).lower():
                    # This is an artifact function class
                    func_map[label.lower()] = iri
                    break
    return func_map

# Extract superclass label from definition (pattern: "is a Y" or "is an Y")
def extract_superclass(definition):
    match = re.search(r'is an? ([A-Za-z \-]+?)(?: that| which| who|$|\.)', definition)
    if match:
        return match.group(1).strip()
    return None

# Detect function phrase from definition (phrases like 'designed to ...', 'used to ...', 'functions by ...')
def detect_function_phrase(definition):
    patterns = [
        r'designed to ([a-zA-Z \-]+?)(?:\.|,|;|$)',
        r'used to ([a-zA-Z \-]+?)(?:\.|,|;|$)',
        r'functions by ([a-zA-Z \-]+?)(?:\.|,|;|$)',
        r'intended to ([a-zA-Z \-]+?)(?:\.|,|;|$)'
    ]
    for pat in patterns:
        match = re.search(pat, definition.lower())
        if match:
            return match.group(1).strip()
    return None

# Match detected function phrase to known function labels (simple token overlap)
def match_function_label(phrase, func_map):
    phrase_tokens = set(phrase.lower().split())
    best_match = None
    best_score = 0
    for label in func_map.keys():
        label_tokens = set(label.split())
        score = len(phrase_tokens.intersection(label_tokens))
        if score > best_score:
            best_score = score
            best_match = label
    if best_score > 0:
        return func_map[best_match]
    return None

# Convert label to prefixed IRI local name (replace spaces and hyphens with underscores)
def label_to_prefixed_iri(label):
    clean_label = label.replace("'", "").replace(" ", "_").replace("-", "_")
    return f"cco:{clean_label}"

# Main processing
def main():
    # Step 1: Extract function classes from merged ontology
    function_map = extract_function_classes(merged_ontology_file)

    # Step 2: Read enriched definitions CSV
    artifacts = []
    with open(definitions_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            artifacts.append({
                'iri': row['iri'].strip(),
                'label': row['label'].strip(),
                'definition': row['definition_enriched'].strip()
            })

    axioms = [prefixes]

    for artifact in artifacts:
        iri = artifact['iri']
        label = artifact['label']
        definition = artifact['definition']

        # Add label annotation
        axioms.append(f"{format_iri(iri)} rdfs:label \"{label}\"@en .")

        # Extract superclass
        superclass_label = extract_superclass(definition)
        if superclass_label:
            superclass_iri = label_to_prefixed_iri(superclass_label)
            axioms.append(f"{format_iri(iri)} rdfs:subClassOf {superclass_iri} .")
        else:
            # Default superclass
            axioms.append(f"{format_iri(iri)} rdfs:subClassOf {material_artifact_class} .")

        # Detect function phrase and match to function IRI
        func_phrase = detect_function_phrase(definition)
        if func_phrase:
            func_iri = match_function_label(func_phrase, function_map)
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

    # Write to TTL file
    with open(output_ttl_file, 'w', encoding='utf-8') as f:
        for ax in axioms:
            f.write(ax + "\n")

    print(f"Generated {len(axioms)} axioms and saved to {output_ttl_file}.")


if __name__ == "__main__":
    main()

