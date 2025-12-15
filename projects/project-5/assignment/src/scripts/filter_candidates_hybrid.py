"""
scripts/filter_candidates_enhanced.py

Enhanced hybrid filtering that handles BOTH:
1. Simple subclass axioms (A ⊑ B)
2. Restriction axioms (A ⊑ ∃R.B)

Uses Anthropic Claude API for semantic validation of both types.
"""

import json
import logging
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import argparse
import time
import os

# Configure logging to suppress urllib3 debug messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress HTTP debug logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

try:
    import numpy as np
except ImportError as e:
    logger.error(f"Missing numpy package: {e}")
    logger.error("Install with: pip install numpy")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    logger.error("anthropic package not installed!")
    logger.error("Install with: pip install anthropic")
    sys.exit(1)


@dataclass
class SimpleAxiom:
    """A simple subclass axiom: A ⊑ B"""
    subclass: str
    superclass: str
    
    def __str__(self):
        return f"{self.subclass} ⊑ {self.superclass}"


@dataclass
class RestrictionAxiom:
    """A restriction axiom: A ⊑ ∃R.B or A ⊑ ∀R.B"""
    subclass: str
    property_iri: str
    restriction_type: str  # "someValuesFrom" or "allValuesFrom"
    filler: str
    
    def __str__(self):
        symbol = "∃" if self.restriction_type == "someValuesFrom" else "∀"
        return f"{self.subclass} ⊑ {symbol}{self.property_iri}.{self.filler}"


# Known OBO property IRIs with their natural language descriptions
PROPERTY_LABELS = {
    "http://purl.obolibrary.org/obo/BFO_0000196": "bearer of",
    "http://purl.obolibrary.org/obo/BFO_0000197": "inheres in",
    "http://purl.obolibrary.org/obo/BFO_0000054": "realized by",
    "http://purl.obolibrary.org/obo/BFO_0000055": "realizes",
    "http://purl.obolibrary.org/obo/BFO_0000057": "has participant",
    "http://purl.obolibrary.org/obo/BFO_0000056": "participates in",
    "http://purl.obolibrary.org/obo/RO_0000053": "bearer of",
    "http://purl.obolibrary.org/obo/RO_0000052": "inheres in",
}


def resolve_paths(candidates_file, metrics_file, output_file):
    """Resolve file paths relative to script location."""
    script_dir = Path(__file__).resolve().parent
    
    if script_dir.name == 'scripts':
        if script_dir.parent.name == 'src':
            project_root = script_dir.parent.parent
            src_dir = script_dir.parent
        else:
            project_root = script_dir.parent
            src_dir = project_root / 'src'
    else:
        project_root = script_dir
        src_dir = project_root / 'src'
    
    def resolve_path(path_str):
        p = Path(path_str)
        if p.is_absolute():
            return p.resolve()
        else:
            rel_to_script = (script_dir / p).resolve()
            if rel_to_script.exists():
                return rel_to_script
            rel_to_root = (project_root / p).resolve()
            return rel_to_root
    
    return resolve_path(candidates_file), resolve_path(metrics_file), resolve_path(output_file)


def to_full_uri(ref: str) -> str:
    """Convert prefixed or angle-bracketed URI to full URI."""
    ref = ref.strip()
    if ref.startswith('<') and ref.endswith('>'):
        return ref[1:-1]
    elif ref.startswith('cco:'):
        return f"https://www.commoncoreontologies.org/{ref.split(':', 1)[1]}"
    elif ref.startswith('obo:'):
        return f"http://purl.obolibrary.org/obo/{ref.split(':', 1)[1]}"
    elif ref.startswith(':'):
        return f"http://test.org/ontology#{ref[1:]}"
    elif ref.startswith('http://') or ref.startswith('https://'):
        return ref
    else:
        return ref


def parse_restriction_axiom(line: str) -> Optional[RestrictionAxiom]:
    """Parse a restriction axiom from a TTL line.
    
    Expected format:
    cco:X rdfs:subClassOf [ a owl:Restriction ; owl:onProperty <P> ; owl:someValuesFrom <Y> ] .
    """
    # Extract subject (before rdfs:subClassOf)
    match = re.match(r'^\s*(\S+)\s+rdfs:subClassOf\s+\[', line)
    if not match:
        return None
    
    subclass = to_full_uri(match.group(1))
    
    # Extract property
    prop_match = re.search(r'owl:onProperty\s+(\S+)', line)
    if not prop_match:
        return None
    property_iri = to_full_uri(prop_match.group(1))
    
    # Extract restriction type and filler
    some_match = re.search(r'owl:someValuesFrom\s+(\S+)', line)
    all_match = re.search(r'owl:allValuesFrom\s+(\S+)', line)
    
    if some_match:
        filler = to_full_uri(some_match.group(1).rstrip(']').rstrip('.').strip())
        return RestrictionAxiom(subclass, property_iri, "someValuesFrom", filler)
    elif all_match:
        filler = to_full_uri(all_match.group(1).rstrip(']').rstrip('.').strip())
        return RestrictionAxiom(subclass, property_iri, "allValuesFrom", filler)
    
    return None


def load_candidates(candidates_file: Path) -> Tuple[List[SimpleAxiom], List[RestrictionAxiom]]:
    """Load both simple and restriction axioms from TTL file."""
    logger.info(f"Loading candidates from: {candidates_file}")
    
    simple_axioms = []
    restriction_axioms = []
    
    with open(candidates_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip comments, empty lines, and prefix declarations
        if not line or line.startswith('#') or line.startswith('@prefix'):
            continue
        
        if 'rdfs:subClassOf' not in line:
            continue
        
        # Try to parse as restriction axiom first
        if '[' in line and 'owl:Restriction' in line:
            restriction = parse_restriction_axiom(line)
            if restriction:
                restriction_axioms.append(restriction)
                logger.debug(f"Line {line_num}: Restriction - {restriction}")
            else:
                logger.warning(f"Line {line_num}: Could not parse restriction axiom")
        else:
            # Parse as simple axiom
            line = line.rstrip('.')
            parts = line.split('rdfs:subClassOf')
            if len(parts) == 2:
                subclass_ref = parts[0].strip()
                superclass_ref = parts[1].strip()
                
                subclass_iri = to_full_uri(subclass_ref)
                superclass_iri = to_full_uri(superclass_ref)
                
                if 'http' in subclass_iri and 'http' in superclass_iri:
                    simple_axioms.append(SimpleAxiom(subclass_iri, superclass_iri))
                    logger.debug(f"Line {line_num}: Simple - {subclass_iri} → {superclass_iri}")
    
    logger.info(f"Loaded {len(simple_axioms)} simple axioms")
    logger.info(f"Loaded {len(restriction_axioms)} restriction axioms")
    
    return simple_axioms, restriction_axioms


def load_mowl_metrics(metrics_file: Path) -> Dict:
    """Load MOWL training results."""
    logger.info(f"Loading MOWL metrics from: {metrics_file}")
    with open(metrics_file, 'r') as f:
        return json.load(f)


def load_class_labels(train_file: Path) -> Dict[str, str]:
    """Extract class labels from training ontology."""
    logger.info(f"Loading class labels from: {train_file}")
    
    labels = {}
    current_class = None
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('cco:ont') and 'rdf:type owl:Class' in line:
                current_class = line.split()[0].replace('cco:', 'https://www.commoncoreontologies.org/')
            elif current_class and 'rdfs:label' in line:
                label = line.split('rdfs:label')[1].strip()
                label = label.replace('"', '').replace('@en', '').replace(';', '').replace('.', '').strip()
                labels[current_class] = label
                current_class = None
    
    logger.info(f"Loaded {len(labels)} class labels")
    return labels


def load_candidate_labels(candidates_file: Path) -> Dict[str, str]:
    """Extract labels from candidate file for entities not in training set."""
    logger.info(f"Loading labels from candidate file: {candidates_file}")
    
    labels = {}
    
    with open(candidates_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#') or line.startswith('@prefix'):
                continue
            
            if 'rdfs:label' in line:
                parts = line.split('rdfs:label')
                if len(parts) == 2:
                    iri_part = parts[0].strip()
                    label_part = parts[1].strip()
                    
                    iri = to_full_uri(iri_part)
                    label = label_part.replace('"', '').replace('@en', '').replace('.', '').replace(';', '').strip()
                    
                    if label and 'http' in iri:
                        labels[iri] = label
    
    logger.info(f"Loaded {len(labels)} labels from candidate file")
    return labels


def get_label_for_iri(iri: str, class_labels: Dict[str, str], 
                       candidate_labels: Dict[str, str]) -> str:
    """Get label for an IRI, checking both training and candidate labels."""
    if iri in class_labels:
        return class_labels[iri]
    if iri in candidate_labels:
        return candidate_labels[iri]
    
    # Check if it's a known property
    if iri in PROPERTY_LABELS:
        return PROPERTY_LABELS[iri]
    
    # Fallback: extract the ontology ID from the IRI
    iri_id = iri.split('/')[-1]
    logger.warning(f"No label found for {iri_id}, using IRI identifier as fallback")
    return iri_id


def compute_cosine_similarity(sub_iri: str, sup_iri: str, embeddings: np.ndarray, 
                               class_to_id: Dict[str, int]) -> float:
    """Compute cosine similarity between two classes using MOWL embeddings."""
    
    if sub_iri not in class_to_id or sup_iri not in class_to_id:
        return 0.0
    
    sub_id = class_to_id[sub_iri]
    sup_id = class_to_id[sup_iri]
    
    sub_emb = embeddings[sub_id]
    sup_emb = embeddings[sup_id]
    
    cos_sim = np.dot(sub_emb, sup_emb) / (
        np.linalg.norm(sub_emb) * np.linalg.norm(sup_emb) + 1e-8
    )
    
    return float(cos_sim)


def query_simple_axiom_plausibility(subclass_label: str, superclass_label: str, 
                                     client: anthropic.Anthropic) -> float:
    """Query Claude to rate semantic plausibility of simple subclass relationship."""
    
    prompt = f"""You are evaluating an ontology axiom for semantic plausibility.

Given the following proposed subclass relationship:
- Subclass: "{subclass_label}"
- Superclass: "{superclass_label}"

Question: On a scale from 0.0 to 1.0, how semantically plausible is it that "{subclass_label}" is a subclass (more specific type) of "{superclass_label}"?

Guidelines:
- 1.0 = Highly plausible, clearly makes sense (e.g., "Hospital" is a subclass of "Healthcare Facility")
- 0.7-0.9 = Plausible, reasonable relationship
- 0.4-0.6 = Questionable, might be valid but unclear
- 0.1-0.3 = Implausible, likely incorrect
- 0.0 = Completely implausible, definitely wrong

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        score = float(response_text)
        return max(0.0, min(1.0, score))
        
    except ValueError:
        logger.warning(f"Could not parse Claude response: {response_text}")
        return 0.5
    except Exception as e:
        logger.error(f"Error querying Claude API: {e}")
        return 0.5


def query_restriction_plausibility(subclass_label: str, property_label: str,
                                    restriction_type: str, filler_label: str,
                                    client: anthropic.Anthropic) -> float:
    """Query Claude to rate semantic plausibility of restriction axiom."""
    
    quantifier = "some" if restriction_type == "someValuesFrom" else "all"
    
    prompt = f"""You are evaluating an ontology axiom for semantic plausibility.

Given the following proposed axiom:
- Subject: "{subclass_label}"
- Relationship: "{property_label}" (applies to {quantifier} instances)
- Object: "{filler_label}"

This axiom states: Every instance of "{subclass_label}" has the relationship "{property_label}" with {quantifier} instance(s) of "{filler_label}".

For example:
- "Hospital" bearer of "Healthcare Function" = PLAUSIBLE (hospitals bear healthcare functions)
- "Airplane" inheres in "Building" = IMPLAUSIBLE (airplanes don't inhere in buildings)

Question: On a scale from 0.0 to 1.0, how semantically plausible is this axiom?

Guidelines:
- 1.0 = Highly plausible, clearly makes sense
- 0.7-0.9 = Plausible, reasonable relationship
- 0.4-0.6 = Questionable, might be valid but unclear
- 0.1-0.3 = Implausible, likely incorrect
- 0.0 = Completely implausible, definitely wrong

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        score = float(response_text)
        return max(0.0, min(1.0, score))
        
    except ValueError:
        logger.warning(f"Could not parse Claude response: {response_text}")
        return 0.5
    except Exception as e:
        logger.error(f"Error querying Claude API: {e}")
        return 0.5


def filter_simple_axioms(axioms: List[SimpleAxiom],
                         embeddings: np.ndarray,
                         class_to_id: Dict[str, int],
                         class_labels: Dict[str, str],
                         candidate_labels: Dict[str, str],
                         client: anthropic.Anthropic,
                         cosine_weight: float = 0.7,
                         llm_weight: float = 0.3,
                         threshold: float = 0.70) -> List[Dict]:
    """Filter simple axioms using hybrid approach."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FILTERING SIMPLE AXIOMS ({len(axioms)} total)")
    logger.info(f"{'='*60}")
    
    accepted = []
    
    for idx, axiom in enumerate(axioms, 1):
        logger.info(f"\n[{idx}/{len(axioms)}] Evaluating simple axiom:")
        
        sub_label = get_label_for_iri(axiom.subclass, class_labels, candidate_labels)
        sup_label = get_label_for_iri(axiom.superclass, class_labels, candidate_labels)
        
        logger.info(f"  {sub_label} ⊑ {sup_label}")
        
        # Compute cosine similarity
        cosine_sim = compute_cosine_similarity(axiom.subclass, axiom.superclass, embeddings, class_to_id)
        logger.info(f"  Cosine: {cosine_sim:.4f}")
        
        # Query LLM
        logger.info(f"  Querying Claude...")
        llm_score = query_simple_axiom_plausibility(sub_label, sup_label, client)
        logger.info(f"  LLM: {llm_score:.4f}")
        
        # Combined score
        combined = (cosine_weight * cosine_sim) + (llm_weight * llm_score)
        logger.info(f"  Combined: {combined:.4f}")
        
        if combined >= threshold:
            logger.info(f"  ✓ ACCEPTED")
            accepted.append({
                'type': 'simple',
                'subclass': axiom.subclass,
                'superclass': axiom.superclass,
                'subclass_label': sub_label,
                'superclass_label': sup_label,
                'cosine_similarity': cosine_sim,
                'llm_score': llm_score,
                'combined_score': combined
            })
        else:
            logger.info(f"  ✗ REJECTED")
        
        time.sleep(0.5)
    
    logger.info(f"\nSimple axioms: {len(accepted)}/{len(axioms)} accepted")
    return accepted


def filter_restriction_axioms(axioms: List[RestrictionAxiom],
                               embeddings: np.ndarray,
                               class_to_id: Dict[str, int],
                               class_labels: Dict[str, str],
                               candidate_labels: Dict[str, str],
                               client: anthropic.Anthropic,
                               embedding_weight: float = 0.4,
                               llm_weight: float = 0.6,
                               threshold: float = 0.70) -> List[Dict]:
    """Filter restriction axioms using hybrid embedding + LLM validation.
    
    For restrictions A ⊑ ∃R.B, we compute:
    - Embedding score: cosine similarity between subject A and filler B
      (captures whether the concepts are semantically related)
    - LLM score: semantic plausibility of the full restriction
    - Combined score: weighted combination
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FILTERING RESTRICTION AXIOMS ({len(axioms)} total)")
    logger.info(f"Weights: embedding={embedding_weight}, llm={llm_weight}")
    logger.info(f"{'='*60}")
    
    accepted = []
    
    for idx, axiom in enumerate(axioms, 1):
        logger.info(f"\n[{idx}/{len(axioms)}] Evaluating restriction axiom:")
        
        sub_label = get_label_for_iri(axiom.subclass, class_labels, candidate_labels)
        prop_label = get_label_for_iri(axiom.property_iri, class_labels, candidate_labels)
        filler_label = get_label_for_iri(axiom.filler, class_labels, candidate_labels)
        
        quantifier = "∃" if axiom.restriction_type == "someValuesFrom" else "∀"
        logger.info(f"  {sub_label} ⊑ {quantifier}{prop_label}.{filler_label}")
        
        # Compute subject-filler embedding similarity
        # This captures whether the subject and filler are semantically related
        embedding_sim = compute_cosine_similarity(
            axiom.subclass, axiom.filler, embeddings, class_to_id
        )
        logger.info(f"  Subject-Filler similarity: {embedding_sim:.4f}")
        
        # Query LLM for full restriction plausibility
        logger.info(f"  Querying Claude...")
        llm_score = query_restriction_plausibility(
            sub_label, prop_label, axiom.restriction_type, filler_label, client
        )
        logger.info(f"  LLM: {llm_score:.4f}")
        
        # Hybrid score: embedding similarity + LLM validation
        combined = (embedding_weight * embedding_sim) + (llm_weight * llm_score)
        logger.info(f"  Combined: {combined:.4f}")
        
        if combined >= threshold:
            logger.info(f"  ✓ ACCEPTED")
            accepted.append({
                'type': 'restriction',
                'subclass': axiom.subclass,
                'property': axiom.property_iri,
                'restriction_type': axiom.restriction_type,
                'filler': axiom.filler,
                'subclass_label': sub_label,
                'property_label': prop_label,
                'filler_label': filler_label,
                'embedding_similarity': embedding_sim,
                'llm_score': llm_score,
                'combined_score': combined
            })
        else:
            logger.info(f"  ✗ REJECTED")
        
        time.sleep(0.5)
    
    logger.info(f"\nRestriction axioms: {len(accepted)}/{len(axioms)} accepted")
    return accepted


def write_accepted_axioms(simple_accepted: List[Dict], 
                          restriction_accepted: List[Dict],
                          output_file: Path, 
                          train_file: Path):
    """Write all accepted axioms to TTL file."""
    
    total = len(simple_accepted) + len(restriction_accepted)
    logger.info(f"\nWriting {total} accepted axioms to {output_file}...")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # Prefixes
        out.write("@prefix : <https://www.commoncoreontologies.org/FacilityOntologyGenerated/> .\n")
        out.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n")
        out.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
        out.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        out.write("@prefix cco: <https://www.commoncoreontologies.org/> .\n")
        out.write("@prefix obo: <http://purl.obolibrary.org/obo/> .\n\n")
        
        # Ontology declaration
        out.write("<https://www.commoncoreontologies.org/FacilityOntologyGenerated> rdf:type owl:Ontology ;\n")
        out.write("    rdfs:label \"Generated Facility Ontology Axioms\"@en ;\n")
        out.write("    rdfs:comment \"Axioms generated by enhanced hybrid MOWL + LLM filtering\"@en .\n\n")
        
        # Simple axioms
        if simple_accepted:
            out.write("#################################################################\n")
            out.write("#    Simple Subclass Axioms\n")
            out.write("#################################################################\n\n")
            
            for axiom in simple_accepted:
                sub_id = axiom['subclass'].split('/')[-1]
                sup_id = axiom['superclass'].split('/')[-1]
                
                out.write(f"### {axiom['subclass_label']} ⊑ {axiom['superclass_label']}\n")
                out.write(f"### Cosine: {axiom['cosine_similarity']:.4f}, LLM: {axiom['llm_score']:.4f}, Combined: {axiom['combined_score']:.4f}\n")
                out.write(f"cco:{sub_id} rdfs:subClassOf cco:{sup_id} .\n\n")
        
        # Restriction axioms
        if restriction_accepted:
            out.write("#################################################################\n")
            out.write("#    Restriction Axioms\n")
            out.write("#################################################################\n\n")
            
            for axiom in restriction_accepted:
                sub_id = axiom['subclass'].split('/')[-1]
                prop = axiom['property']
                filler = axiom['filler']
                
                # Format property and filler
                if prop.startswith('http://purl.obolibrary.org/obo/'):
                    prop_ref = f"obo:{prop.split('/')[-1]}"
                else:
                    prop_ref = f"<{prop}>"
                
                if filler.startswith('https://www.commoncoreontologies.org/'):
                    filler_ref = f"cco:{filler.split('/')[-1]}"
                else:
                    filler_ref = f"<{filler}>"
                
                quantifier = "∃" if axiom['restriction_type'] == "someValuesFrom" else "∀"
                out.write(f"### {axiom['subclass_label']} ⊑ {quantifier}{axiom['property_label']}.{axiom['filler_label']}\n")
                out.write(f"### Embedding: {axiom['embedding_similarity']:.4f}, LLM: {axiom['llm_score']:.4f}, Combined: {axiom['combined_score']:.4f}\n")
                out.write(f"cco:{sub_id} rdfs:subClassOf [ a owl:Restriction ;\n")
                out.write(f"    owl:onProperty {prop_ref} ;\n")
                out.write(f"    owl:{axiom['restriction_type']} {filler_ref} ] .\n\n")
    
    logger.info(f"✓ Wrote {output_file}")
    logger.info(f"  - {len(simple_accepted)} simple axioms")
    logger.info(f"  - {len(restriction_accepted)} restriction axioms")


def load_embeddings_and_mappings(metrics: Dict) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load saved embeddings and class mappings."""
    
    if 'embeddings_file' in metrics and 'mappings_file' in metrics:
        embeddings_file = Path(metrics['embeddings_file'])
        mappings_file = Path(metrics['mappings_file'])
        
        if embeddings_file.exists() and mappings_file.exists():
            logger.info(f"Loading embeddings from: {embeddings_file}")
            embeddings = np.load(embeddings_file)
            
            logger.info(f"Loading mappings from: {mappings_file}")
            import pickle
            with open(mappings_file, 'rb') as f:
                mappings_data = pickle.load(f)
            
            class_to_id = mappings_data['class_to_id']
            
            logger.info(f"Loaded embeddings: shape {embeddings.shape}")
            logger.info(f"Loaded {len(class_to_id)} class mappings")
            
            return embeddings, class_to_id
    
    logger.warning("Could not load saved embeddings/mappings. Creating synthetic data.")
    
    n_classes = metrics.get('n_classes', 100)
    embed_dim = metrics.get('hyperparameters', {}).get('embedding_dim', 200)
    
    embeddings = np.random.randn(n_classes, embed_dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    class_to_id = {}
    
    return embeddings, class_to_id


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced hybrid filtering: handles simple + restriction axioms'
    )
    
    script_dir = Path(__file__).resolve().parent
    reports_in_scripts = script_dir / 'reports'
    
    if script_dir.name == 'scripts':
        if script_dir.parent.name == 'src':
            src_dir = script_dir.parent
            project_root = src_dir.parent
        else:
            project_root = script_dir.parent
            src_dir = project_root / 'src'
    else:
        project_root = script_dir
        src_dir = project_root / 'src'
    
    default_generated = src_dir / 'generated'
    if not default_generated.exists():
        default_generated = project_root / 'generated'
    
    if reports_in_scripts.exists():
        default_reports = reports_in_scripts
    elif (project_root / 'reports').exists():
        default_reports = project_root / 'reports'
    elif (src_dir / 'reports').exists():
        default_reports = src_dir / 'reports'
    else:
        default_reports = reports_in_scripts
    
    parser.add_argument('--candidates', 
                        default=str(default_generated / 'candidate_el.ttl'),
                        help='Input candidate axioms file')
    parser.add_argument('--metrics', 
                        default=str(default_reports / 'mowl_metrics.json'),
                        help='MOWL training metrics file')
    parser.add_argument('--train', 
                        default=str(src_dir / 'train.ttl'),
                        help='Training ontology (for class labels)')
    parser.add_argument('--output', 
                        default=str(default_generated / 'accepted_el.ttl'),
                        help='Output accepted axioms file')
    parser.add_argument('--cosine-weight', type=float, default=0.7,
                        help='Weight for cosine similarity (simple axioms only, default: 0.7)')
    parser.add_argument('--llm-weight', type=float, default=0.3,
                        help='Weight for LLM score (simple axioms only, default: 0.3)')
    parser.add_argument('--restriction-embedding-weight', type=float, default=0.4,
                        help='Weight for embedding similarity (restriction axioms, default: 0.4)')
    parser.add_argument('--restriction-llm-weight', type=float, default=0.6,
                        help='Weight for LLM score (restriction axioms, default: 0.6)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score threshold for acceptance (default: read from metrics file, fallback 0.70)')
    parser.add_argument('--skip-simple', action='store_true', default=True,
                        help='Skip simple subclass axioms (A ⊑ B), only process restrictions (default: True)')
    parser.add_argument('--include-simple', action='store_true', default=False,
                        help='Include simple subclass axioms (overrides --skip-simple)')
    parser.add_argument('--api-key', default=None,
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Resolve paths
    candidates_path, metrics_path, output_path = resolve_paths(
        args.candidates, args.metrics, args.output
    )
    
    train_path = Path(args.train).resolve()
    
    # Check files exist
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    # Get API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        logger.error("\n" + "="*60)
        logger.error("ERROR: Anthropic API key required!")
        logger.error("="*60)
        logger.error("Set environment variable:")
        logger.error("  PowerShell: $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
        logger.error("  Bash: export ANTHROPIC_API_KEY='sk-ant-...'")
        logger.error("\nOr pass as argument:")
        logger.error("  python filter_candidates_enhanced.py --api-key sk-ant-...")
        logger.error("="*60)
        sys.exit(1)
    
    # Initialize Anthropic client
    logger.info("Initializing Anthropic API client...")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("✓ Anthropic client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")
        sys.exit(1)
    
    # Load data
    simple_axioms, restriction_axioms = load_candidates(candidates_path)
    metrics = load_mowl_metrics(metrics_path)
    class_labels = load_class_labels(train_path)
    candidate_labels = load_candidate_labels(candidates_path)
    
    # =========================================================================
    # Determine threshold: command line > metrics file > fallback
    # =========================================================================
    FALLBACK_THRESHOLD = 0.70
    
    if args.threshold is not None:
        # User explicitly provided threshold on command line
        threshold = args.threshold
        threshold_source = "command line"
    else:
        # Try to read from metrics file
        optimal_from_metrics = metrics.get('validation_metrics', {}).get('optimal_threshold')
        
        if optimal_from_metrics is not None:
            threshold = optimal_from_metrics
            threshold_source = "metrics file (optimal_threshold)"
        else:
            threshold = FALLBACK_THRESHOLD
            threshold_source = "fallback default"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"THRESHOLD CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"Using threshold: {threshold:.4f} (from {threshold_source})")
    
    # Also log embedding quality context from metrics
    cosine_stats = metrics.get('validation_metrics', {}).get('cosine_similarities', {})
    if cosine_stats:
        logger.info(f"Embedding quality (from training):")
        logger.info(f"  Mean cosine: {cosine_stats.get('mean', 'N/A'):.4f}" if cosine_stats.get('mean') else "  Mean cosine: N/A")
        logger.info(f"  Max cosine: {cosine_stats.get('max', 'N/A'):.4f}" if cosine_stats.get('max') else "  Max cosine: N/A")
    logger.info(f"{'='*60}\n")
    
    # Load embeddings
    embeddings, class_to_id = load_embeddings_and_mappings(metrics)
    
    if not class_to_id:
        logger.info("Building class_to_id mapping from candidates...")
        unique_classes = set()
        for axiom in simple_axioms:
            unique_classes.add(axiom.subclass)
            unique_classes.add(axiom.superclass)
        for axiom in restriction_axioms:
            unique_classes.add(axiom.subclass)
            unique_classes.add(axiom.filler)
        
        for idx, cls in enumerate(sorted(unique_classes)):
            class_to_id[cls] = idx
        
        embeddings = np.random.randn(len(class_to_id), 
                                     metrics.get('hyperparameters', {}).get('embedding_dim', 200))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
    
    # =========================================================================
    # Determine whether to process simple axioms
    # =========================================================================
    # Default: skip simple axioms (they're likely already in the ontology)
    # Use --include-simple to override and process them
    process_simple = args.include_simple
    
    logger.info(f"\n{'='*60}")
    logger.info("AXIOM PROCESSING CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"Simple subclass axioms (A ⊑ B): {len(simple_axioms)} candidates")
    logger.info(f"Restriction axioms (A ⊑ ∃R.B): {len(restriction_axioms)} candidates")
    
    if process_simple:
        logger.info(f"Processing: BOTH simple and restriction axioms (--include-simple)")
    else:
        logger.info(f"Processing: ONLY restriction axioms (simple axioms skipped)")
        logger.info(f"  Reason: Simple subclass axioms are likely already in the ontology")
        logger.info(f"  Use --include-simple to process them")
    logger.info(f"{'='*60}\n")
    
    # Filter axioms based on configuration
    if process_simple:
        simple_accepted = filter_simple_axioms(
            simple_axioms, embeddings, class_to_id, class_labels, candidate_labels,
            client, args.cosine_weight, args.llm_weight, threshold
        )
    else:
        simple_accepted = []  # Skip simple axioms
    
    restriction_accepted = filter_restriction_axioms(
        restriction_axioms, embeddings, class_to_id, class_labels, candidate_labels, 
        client, 
        embedding_weight=args.restriction_embedding_weight, 
        llm_weight=args.restriction_llm_weight, 
        threshold=threshold
    )
    
    # Write results
    write_accepted_axioms(simple_accepted, restriction_accepted, output_path, train_path)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FILTERING COMPLETE")
    logger.info("="*60)
    logger.info(f"Threshold used: {threshold:.4f} (from {threshold_source})")
    if process_simple:
        logger.info(f"Simple axioms: {len(simple_accepted)}/{len(simple_axioms)} accepted")
    else:
        logger.info(f"Simple axioms: SKIPPED ({len(simple_axioms)} not processed)")
    logger.info(f"Restriction axioms: {len(restriction_accepted)}/{len(restriction_axioms)} accepted")
    logger.info(f"Total accepted: {len(simple_accepted) + len(restriction_accepted)} axioms")
    logger.info(f"Output file: {output_path}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())