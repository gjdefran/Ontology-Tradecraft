#!/usr/bin/env python3
"""
Split an OWL/TTL ontology into train and validation sets for mowl embeddings.

This script ensures:
1. All class and property declarations appear in BOTH train.ttl and valid.ttl
2. Axioms (SubClassOf, EquivalentClass, etc.) are split 80/20 between train and valid
3. Prefixes and ontology metadata are preserved in both files
4. Each output file gets a UNIQUE ontology IRI to avoid OWL API conflicts
5. External classes/properties referenced in axioms get declarations added
6. owl:imports statements are removed (mOWL cannot resolve them)

Specifically designed to handle Common Core Ontology (CCO) structure with:
- SKOS annotations (definition, altLabel, scopeNote, example)
- CCO-specific annotation properties (ont00001760, ont00001754)
- Complex OWL restrictions and equivalentClass axioms
- External references to BFO, CCO imports
"""

import random
from pathlib import Path
from collections import defaultdict
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, BNode, Literal
from rdflib.namespace import XSD

# =============================================================================
# CONFIGURATION - Edit these paths for your workflow
# =============================================================================

# Input ontology file path
INPUT_FILE = "../ArtifactOntology.ttl"

# Output directory (train.ttl and valid.ttl will be created here)
OUTPUT_DIR = ".."

# Ratio of axioms for training set (0.8 = 80% train, 20% valid)
TRAIN_RATIO = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================

# Define namespaces
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
CCO = Namespace("https://www.commoncoreontologies.org/")
DC = Namespace("http://purl.org/dc/elements/1.1/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
OBO = Namespace("http://purl.obolibrary.org/obo/")

# Predicates that define class/property declarations (must be in both files)
DECLARATION_PREDICATES = {
    RDF.type,
    RDFS.label,
    RDFS.comment,
    RDFS.isDefinedBy,
    OWL.deprecated,
    OWL.versionIRI,
    OWL.versionInfo,
    OWL.imports,
    # SKOS annotation properties
    SKOS.prefLabel,
    SKOS.altLabel,
    SKOS.definition,
    SKOS.scopeNote,
    SKOS.example,
    SKOS.note,
    SKOS.editorialNote,
    SKOS.historyNote,
    SKOS.changeNote,
    # Dublin Core
    DC.description,
    DC.creator,
    DC.contributor,
    DC.date,
    DC.rights,
    DC.title,
    DCTERMS.description,
    DCTERMS.creator,
    DCTERMS.contributor,
    DCTERMS.license,
    DCTERMS.rights,
    DCTERMS.title,
    # CCO-specific annotation properties
    CCO["ont00001760"],  # is curated in ontology
    CCO["ont00001754"],  # see also / reference URL
    CCO["ont00001755"],  # alternative term
    CCO["ont00001756"],  # term editor
    CCO["ont00001757"],  # elucidation
    CCO["ont00001759"],  # curator note
}

# Object types that indicate a declaration
DECLARATION_TYPES = {
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    OWL.NamedIndividual,
    OWL.Ontology,
    RDFS.Class,
    RDF.Property,
    RDFS.Datatype,
    OWL.FunctionalProperty,
    OWL.InverseFunctionalProperty,
    OWL.TransitiveProperty,
    OWL.SymmetricProperty,
    OWL.AsymmetricProperty,
    OWL.ReflexiveProperty,
    OWL.IrreflexiveProperty,
}

# Predicates that represent axioms (to be split)
AXIOM_PREDICATES = {
    RDFS.subClassOf,
    RDFS.subPropertyOf,
    RDFS.domain,
    RDFS.range,
    OWL.equivalentClass,
    OWL.equivalentProperty,
    OWL.disjointWith,
    OWL.complementOf,
    OWL.inverseOf,
    OWL.propertyDisjointWith,
    OWL.sameAs,
    OWL.differentFrom,
    OWL.hasKey,
    OWL.onProperty,
    OWL.someValuesFrom,
    OWL.allValuesFrom,
    OWL.hasValue,
    OWL.minCardinality,
    OWL.maxCardinality,
    OWL.cardinality,
    OWL.minQualifiedCardinality,
    OWL.maxQualifiedCardinality,
    OWL.qualifiedCardinality,
    OWL.onClass,
    OWL.onDataRange,
    OWL.oneOf,
    OWL.unionOf,
    OWL.intersectionOf,
    OWL.members,
    OWL.distinctMembers,
    OWL.disjointUnionOf,
    OWL.propertyChainAxiom,
    OWL.sourceIndividual,
    OWL.assertionProperty,
    OWL.targetIndividual,
    OWL.targetValue,
}


def get_bnode_closure(graph, bnode, visited=None):
    """
    Get all triples that are part of a blank node's closure.
    This handles nested blank nodes (e.g., complex class expressions).
    """
    if visited is None:
        visited = set()

    if bnode in visited:
        return []

    visited.add(bnode)
    triples = []

    for s, p, o in graph.triples((bnode, None, None)):
        triples.append((s, p, o))
        if isinstance(o, BNode):
            triples.extend(get_bnode_closure(graph, o, visited))

    return triples


def is_declaration_triple(graph, s, p, o):
    """Check if a triple is a declaration (should be in both files)."""
    # Type declarations for classes, properties, etc.
    if p == RDF.type and o in DECLARATION_TYPES:
        return True

    # Common annotation/metadata predicates
    if p in DECLARATION_PREDICATES:
        return True

    # Ontology-level metadata
    if (s, RDF.type, OWL.Ontology) in graph:
        return True

    return False


def is_axiom_triple(graph, s, p, o):
    """Check if a triple represents an axiom (should be split)."""
    # Direct axiom predicates
    if p in AXIOM_PREDICATES:
        return True

    # rdf:type with a complex class expression (blank node) is an axiom
    if p == RDF.type and isinstance(o, BNode):
        return True

    return False


def collect_list_items(graph, list_node):
    """Collect all triples that make up an RDF list."""
    triples = []
    current = list_node

    while current and current != RDF.nil:
        # Get first and rest
        for s, p, o in graph.triples((current, None, None)):
            triples.append((s, p, o))
            if isinstance(o, BNode) and p != RDF.rest:
                triples.extend(get_bnode_closure(graph, o))

        # Move to next
        rest = graph.value(current, RDF.rest)
        current = rest

    return triples


def find_ontology_iri(graph):
    """Find the ontology IRI from the graph."""
    for s, p, o in graph.triples((None, RDF.type, OWL.Ontology)):
        if isinstance(s, URIRef):
            return s
    return None


def rename_ontology_iri(graph, old_iri, new_iri):
    """
    Rename the ontology IRI in a graph.
    Updates both the subject and the versionIRI if present.
    """
    if old_iri is None:
        return
    
    # Collect triples to modify (can't modify during iteration)
    triples_to_remove = []
    triples_to_add = []
    
    for s, p, o in graph:
        new_s = new_iri if s == old_iri else s
        new_o = o
        
        # Update versionIRI if it contains the old IRI
        if p == OWL.versionIRI and isinstance(o, URIRef):
            old_version_str = str(o)
            old_iri_str = str(old_iri)
            if old_iri_str in old_version_str:
                new_version_str = old_version_str.replace(old_iri_str, str(new_iri))
                new_o = URIRef(new_version_str)
            else:
                # Just append a suffix to distinguish
                new_o = URIRef(str(o) + "-valid")
        
        if new_s != s or new_o != o:
            triples_to_remove.append((s, p, o))
            triples_to_add.append((new_s, p, new_o))
    
    for triple in triples_to_remove:
        graph.remove(triple)
    
    for triple in triples_to_add:
        graph.add(triple)


def find_all_referenced_uris(graph):
    """
    Find all URIs referenced in axioms that might need declarations.
    This includes classes in SubClassOf, restrictions, equivalentClass, etc.
    """
    referenced_classes = set()
    referenced_properties = set()
    
    # Properties that typically reference classes
    class_referencing_predicates = {
        RDFS.subClassOf,
        OWL.equivalentClass,
        OWL.disjointWith,
        OWL.complementOf,
        OWL.someValuesFrom,
        OWL.allValuesFrom,
        OWL.onClass,
        OWL.hasValue,
    }
    
    # Properties that reference object properties
    property_referencing_predicates = {
        OWL.onProperty,
        OWL.inverseOf,
        RDFS.subPropertyOf,
        OWL.equivalentProperty,
        OWL.propertyDisjointWith,
    }
    
    for s, p, o in graph:
        # Check subject - if it's a named class in an axiom
        if isinstance(s, URIRef) and p in class_referencing_predicates:
            referenced_classes.add(s)
        
        # Check object
        if isinstance(o, URIRef):
            if p in class_referencing_predicates:
                referenced_classes.add(o)
            elif p in property_referencing_predicates:
                referenced_properties.add(o)
            elif p == RDF.type and o not in DECLARATION_TYPES:
                # Individual typed with a class
                referenced_classes.add(o)
    
    # Also check blank node closures for restriction classes
    for s, p, o in graph:
        if isinstance(o, BNode):
            for bs, bp, bo in get_bnode_closure(graph, o):
                if isinstance(bo, URIRef):
                    if bp in class_referencing_predicates:
                        referenced_classes.add(bo)
                    elif bp in property_referencing_predicates:
                        referenced_properties.add(bo)
    
    return referenced_classes, referenced_properties


def get_declared_entities(graph):
    """Get all entities that already have type declarations."""
    declared_classes = set()
    declared_properties = set()
    
    # Find declared classes
    for s, p, o in graph.triples((None, RDF.type, OWL.Class)):
        if isinstance(s, URIRef):
            declared_classes.add(s)
    for s, p, o in graph.triples((None, RDF.type, RDFS.Class)):
        if isinstance(s, URIRef):
            declared_classes.add(s)
    
    # Find declared properties
    for prop_type in [OWL.ObjectProperty, OWL.DatatypeProperty, 
                      OWL.AnnotationProperty, RDF.Property]:
        for s, p, o in graph.triples((None, RDF.type, prop_type)):
            if isinstance(s, URIRef):
                declared_properties.add(s)
    
    return declared_classes, declared_properties


def add_missing_declarations(graph):
    """
    Add owl:Class and owl:ObjectProperty declarations for all referenced 
    entities that don't have declarations.
    
    This is necessary for mOWL because it builds class/property indexes
    from declarations, and external references (BFO, imports) need to be declared.
    """
    referenced_classes, referenced_properties = find_all_referenced_uris(graph)
    declared_classes, declared_properties = get_declared_entities(graph)
    
    # Find undeclared entities
    undeclared_classes = referenced_classes - declared_classes
    undeclared_properties = referenced_properties - declared_properties
    
    # Add declarations
    added_classes = 0
    for cls in undeclared_classes:
        graph.add((cls, RDF.type, OWL.Class))
        added_classes += 1
    
    added_properties = 0
    for prop in undeclared_properties:
        graph.add((prop, RDF.type, OWL.ObjectProperty))
        added_properties += 1
    
    return added_classes, added_properties


def remove_imports(graph):
    """
    Remove owl:imports statements from the graph.
    mOWL cannot resolve remote imports, and they cause issues.
    """
    imports_to_remove = list(graph.triples((None, OWL.imports, None)))
    removed_count = len(imports_to_remove)
    
    for triple in imports_to_remove:
        graph.remove(triple)
    
    return removed_count


def split_ontology(input_path, train_ratio=0.8, output_dir="src", seed=42):
    """
    Split ontology into train and validation sets.

    Args:
        input_path: Path to input TTL file
        train_ratio: Fraction of axioms for training (default 0.8)
        output_dir: Output directory
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print(f"Loading ontology from {input_path}...")
    g = Graph()
    g.parse(input_path, format="turtle")

    print(f"Loaded {len(g)} triples")
    
    # =========================================================================
    # PREPROCESSING: Handle external references and imports
    # =========================================================================
    
    # Remove owl:imports (mOWL cannot resolve remote imports)
    removed_imports = remove_imports(g)
    if removed_imports > 0:
        print(f"Removed {removed_imports} owl:imports statement(s)")
    
    # Add declarations for all referenced but undeclared entities
    # This is critical for mOWL which builds indexes from declarations
    added_classes, added_props = add_missing_declarations(g)
    if added_classes > 0 or added_props > 0:
        print(f"Added declarations for {added_classes} external classes and {added_props} external properties")
    
    print(f"Graph now has {len(g)} triples after preprocessing")
    
    # =========================================================================
    
    # Find the original ontology IRI
    original_ontology_iri = find_ontology_iri(g)
    if original_ontology_iri:
        print(f"Found ontology IRI: {original_ontology_iri}")

    # Collect declarations and axioms
    declarations = []  # Go in both files
    axiom_groups = []  # Groups of related triples to be split
    processed = set()

    # First pass: identify all triples and their types
    for s, p, o in g:
        triple_id = (s, p, o)
        if triple_id in processed:
            continue

        # Handle blank nodes specially - they often represent complex axioms
        if isinstance(s, BNode):
            # These will be collected as part of their parent axiom
            continue

        if is_declaration_triple(g, s, p, o):
            declarations.append((s, p, o))
            processed.add(triple_id)
            # Also collect any blank node closures for declarations
            if isinstance(o, BNode):
                for t in get_bnode_closure(g, o):
                    if t not in processed:
                        declarations.append(t)
                        processed.add(t)

        elif is_axiom_triple(g, s, p, o):
            axiom_triples = [(s, p, o)]
            processed.add(triple_id)

            # Collect all related blank node triples
            if isinstance(o, BNode):
                bnode_triples = get_bnode_closure(g, o)
                for t in bnode_triples:
                    if t not in processed:
                        axiom_triples.append(t)
                        processed.add(t)

            axiom_groups.append(axiom_triples)

    # Handle remaining triples (treat as declarations to be safe)
    for s, p, o in g:
        triple_id = (s, p, o)
        if triple_id not in processed:
            declarations.append((s, p, o))
            processed.add(triple_id)

    print(f"Found {len(declarations)} declaration triples")
    print(f"Found {len(axiom_groups)} axiom groups")

    # Shuffle and split axiom groups
    random.shuffle(axiom_groups)
    split_idx = int(len(axiom_groups) * train_ratio)

    train_axiom_groups = axiom_groups[:split_idx]
    valid_axiom_groups = axiom_groups[split_idx:]

    train_axioms = [t for group in train_axiom_groups for t in group]
    valid_axioms = [t for group in valid_axiom_groups for t in group]

    print(f"Train axiom groups: {len(train_axiom_groups)} ({len(train_axioms)} triples)")
    print(f"Valid axiom groups: {len(valid_axiom_groups)} ({len(valid_axioms)} triples)")

    # Create output graphs
    train_graph = Graph()
    valid_graph = Graph()

    # Copy namespace bindings
    for prefix, namespace in g.namespaces():
        train_graph.bind(prefix, namespace)
        valid_graph.bind(prefix, namespace)

    # Add declarations to both graphs
    for triple in declarations:
        train_graph.add(triple)
        valid_graph.add(triple)

    # Add axioms to respective graphs
    for triple in train_axioms:
        train_graph.add(triple)

    for triple in valid_axioms:
        valid_graph.add(triple)

    # =========================================================================
    # CRITICAL: Rename ontology IRIs to be unique for each file
    # This prevents OWLOntologyAlreadyExistsException when loading both
    # =========================================================================
    if original_ontology_iri:
        # Create unique IRIs for train and valid
        train_ontology_iri = URIRef(str(original_ontology_iri) + "-train")
        valid_ontology_iri = URIRef(str(original_ontology_iri) + "-valid")
        
        print(f"\nRenaming ontology IRIs to avoid conflicts:")
        print(f"  Train IRI: {train_ontology_iri}")
        print(f"  Valid IRI: {valid_ontology_iri}")
        
        rename_ontology_iri(train_graph, original_ontology_iri, train_ontology_iri)
        rename_ontology_iri(valid_graph, original_ontology_iri, valid_ontology_iri)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save files
    train_path = output_path / "train.ttl"
    valid_path = output_path / "valid.ttl"

    print(f"\nSaving train.ttl ({len(train_graph)} triples)...")
    train_graph.serialize(destination=str(train_path), format="turtle")

    print(f"Saving valid.ttl ({len(valid_graph)} triples)...")
    valid_graph.serialize(destination=str(valid_path), format="turtle")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Input file: {input_path}")
    print(f"Total triples in original: {len(g)}")
    print(f"Declaration triples (in both): {len(declarations)}")
    print(f"Axiom groups total: {len(axiom_groups)}")
    print(f"  - Train axiom groups: {len(train_axiom_groups)} ({100*train_ratio:.0f}%)")
    print(f"  - Valid axiom groups: {len(valid_axiom_groups)} ({100*(1-train_ratio):.0f}%)")
    print(f"\nOutput files:")
    print(f"  - {train_path}: {len(train_graph)} triples")
    print(f"  - {valid_path}: {len(valid_graph)} triples")

    return train_path, valid_path


def verify_split(train_path, valid_path):
    """Verify that both files have the same classes and properties."""
    print("\n" + "=" * 60)
    print("VERIFICATION (Named entities only, excluding blank nodes)")
    print("=" * 60)

    train_g = Graph()
    train_g.parse(train_path, format="turtle")

    valid_g = Graph()
    valid_g.parse(valid_path, format="turtle")
    
    # Verify unique ontology IRIs
    train_ont_iri = find_ontology_iri(train_g)
    valid_ont_iri = find_ontology_iri(valid_g)
    
    print(f"\nOntology IRIs:")
    print(f"  Train: {train_ont_iri}")
    print(f"  Valid: {valid_ont_iri}")
    print(f"  IRIs are unique: {train_ont_iri != valid_ont_iri}")
    
    if train_ont_iri == valid_ont_iri:
        print("  ⚠ WARNING: Ontology IRIs are identical! This will cause OWL API errors.")

    # Get named classes from both (exclude blank nodes - anonymous classes)
    train_classes = set(
        s for s, p, o in train_g.triples((None, RDF.type, OWL.Class))
        if not isinstance(s, BNode)
    )
    valid_classes = set(
        s for s, p, o in valid_g.triples((None, RDF.type, OWL.Class))
        if not isinstance(s, BNode)
    )

    # Get properties from both
    train_obj_props = set(
        s for s, p, o in train_g.triples((None, RDF.type, OWL.ObjectProperty))
        if not isinstance(s, BNode)
    )
    valid_obj_props = set(
        s for s, p, o in valid_g.triples((None, RDF.type, OWL.ObjectProperty))
        if not isinstance(s, BNode)
    )

    train_data_props = set(
        s for s, p, o in train_g.triples((None, RDF.type, OWL.DatatypeProperty))
        if not isinstance(s, BNode)
    )
    valid_data_props = set(
        s for s, p, o in valid_g.triples((None, RDF.type, OWL.DatatypeProperty))
        if not isinstance(s, BNode)
    )

    # Get entities with labels (for additional verification)
    train_labels = set(
        s for s, p, o in train_g.triples((None, RDFS.label, None))
        if not isinstance(s, BNode)
    )
    valid_labels = set(
        s for s, p, o in valid_g.triples((None, RDFS.label, None))
        if not isinstance(s, BNode)
    )

    print(f"\nNamed classes in train.ttl: {len(train_classes)}")
    print(f"Named classes in valid.ttl: {len(valid_classes)}")
    classes_match = train_classes == valid_classes
    print(f"Named classes match: {classes_match}")

    if not classes_match:
        missing_in_valid = train_classes - valid_classes
        missing_in_train = valid_classes - train_classes
        if missing_in_valid:
            print(f"  ⚠ Missing in valid: {len(missing_in_valid)}")
        if missing_in_train:
            print(f"  ⚠ Missing in train: {len(missing_in_train)}")

    print(f"\nEntities with rdfs:label in train.ttl: {len(train_labels)}")
    print(f"Entities with rdfs:label in valid.ttl: {len(valid_labels)}")
    print(f"Labels match: {train_labels == valid_labels}")

    if train_obj_props or valid_obj_props:
        print(f"\nObject properties in train.ttl: {len(train_obj_props)}")
        print(f"Object properties in valid.ttl: {len(valid_obj_props)}")
        print(f"Object properties match: {train_obj_props == valid_obj_props}")

    if train_data_props or valid_data_props:
        print(f"\nData properties in train.ttl: {len(train_data_props)}")
        print(f"Data properties in valid.ttl: {len(valid_data_props)}")
        print(f"Data properties match: {train_data_props == valid_data_props}")

    # Count axioms (excluding blank node subjects for accurate counting)
    train_subclass = len([
        t for t in train_g.triples((None, RDFS.subClassOf, None))
        if not isinstance(t[0], BNode)
    ])
    valid_subclass = len([
        t for t in valid_g.triples((None, RDFS.subClassOf, None))
        if not isinstance(t[0], BNode)
    ])
    train_equiv = len(list(train_g.triples((None, OWL.equivalentClass, None))))
    valid_equiv = len(list(valid_g.triples((None, OWL.equivalentClass, None))))
    train_disjoint = len(list(train_g.triples((None, OWL.disjointWith, None))))
    valid_disjoint = len(list(valid_g.triples((None, OWL.disjointWith, None))))

    print(f"\n{'Axiom Type':<25} {'Train':<10} {'Valid':<10} {'Total':<10}")
    print("-" * 55)
    
    total_subclass = train_subclass + valid_subclass
    if total_subclass > 0:
        print(f"{'rdfs:subClassOf':<25} {train_subclass:<10} {valid_subclass:<10} {total_subclass:<10} ({100*train_subclass/total_subclass:.1f}%/{100*valid_subclass/total_subclass:.1f}%)")
    
    total_equiv = train_equiv + valid_equiv
    if total_equiv > 0:
        print(f"{'owl:equivalentClass':<25} {train_equiv:<10} {valid_equiv:<10} {total_equiv:<10}")
    
    total_disjoint = train_disjoint + valid_disjoint
    if total_disjoint > 0:
        print(f"{'owl:disjointWith':<25} {train_disjoint:<10} {valid_disjoint:<10} {total_disjoint:<10}")

    return classes_match


if __name__ == "__main__":
    print("=" * 60)
    print("Ontology Splitter for mowl Embeddings")
    print("=" * 60)
    print(f"Input file:    {INPUT_FILE}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Train ratio:   {TRAIN_RATIO}")
    print(f"Random seed:   {RANDOM_SEED}")
    print("=" * 60)

    train_path, valid_path = split_ontology(
        input_path=INPUT_FILE,
        train_ratio=TRAIN_RATIO,
        output_dir=OUTPUT_DIR,
        seed=RANDOM_SEED,
    )

    verify_split(train_path, valid_path)

    print("\n✓ Done! Files ready for mowl embeddings.")