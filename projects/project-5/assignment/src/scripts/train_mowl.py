#!/usr/bin/env python3
"""
Train an ELEmbeddings model on ontology data and evaluate on validation set.
Finds optimal threshold for subclass-superclass relationships based on cosine similarity.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Initialize JVM before importing mOWL
import jpype
import jpype.imports

if not jpype.isJVMStarted():
    # Start JVM with mOWL's classpath (4GB memory allocation)
    import mowl
    mowl.init_jvm("4g")

from mowl.datasets import PathDataset
from mowl.models import ELEmbeddings

import torch
from sklearn.metrics.pairwise import cosine_similarity


def load_ontology_dataset(train_path: str, valid_path: str) -> PathDataset:
    """Load ontology files into mOWL dataset."""
    print(f"Loading ontology from {train_path} and {valid_path}")
    dataset = PathDataset(train_path, validation_path=valid_path)
    return dataset


def extract_subclass_pairs(dataset: PathDataset, split: str = 'validation') -> List[Tuple[str, str]]:
    """Extract subclass-superclass pairs from the validation ontology."""
    ontology = dataset.validation if split == 'validation' else dataset.ontology
    pairs = []
    
    for axiom in ontology.getAxioms():
        axiom_type = axiom.getAxiomType().getName()
        
        if axiom_type == "SubClassOf":
            subclass = axiom.getSubClass()
            superclass = axiom.getSuperClass()
            
            # Only consider named classes (not complex expressions)
            if not subclass.isAnonymous() and not superclass.isAnonymous():
                sub_iri = str(subclass.asOWLClass().getIRI())
                super_iri = str(superclass.asOWLClass().getIRI())
                pairs.append((sub_iri, super_iri))
    
    print(f"Extracted {len(pairs)} subclass-superclass pairs from {split} set")
    return pairs


def train_elembeddings(dataset: PathDataset, 
                       embedding_dim: int = 50,
                       epochs: int = 100,
                       learning_rate: float = 0.001,
                       batch_size: int = 128,
                       margin: float = 0.1) -> ELEmbeddings:
    """Train ELEmbeddings model on the dataset."""
    print(f"\nTraining ELEmbeddings model...")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Margin: {margin}")
    
    model = ELEmbeddings(
        dataset,
        embed_dim=embedding_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        margin=margin
    )
    
    model.train()
    print("Training completed!")
    
    return model


def get_class_embedding(model: ELEmbeddings, class_iri: str) -> Tuple:
    """Get embedding vector and radius for a class IRI.
    
    Returns:
        tuple: (center_embedding, radius) or (None, None) if not found
    """
    try:
        # ELEmbeddings stores class centers
        center = model.class_embeddings[class_iri]
        if torch.is_tensor(center):
            center = center.detach().cpu().numpy()
        
        # ELEmbeddings also stores radii
        radius = None
        if hasattr(model, 'class_rad') and class_iri in model.class_rad:
            radius = model.class_rad[class_iri]
            if torch.is_tensor(radius):
                radius = float(radius.detach().cpu().numpy())
        
        return center, radius
    except (KeyError, AttributeError):
        return None, None


def compute_cosine_similarities(model: ELEmbeddings, 
                                pairs: List[Tuple[str, str]]) -> List[float]:
    """Compute cosine similarities for subclass-superclass pairs.
    
    Uses only cosine similarity between class center embeddings.
    """
    similarities = []
    missing_count = 0
    
    for sub_iri, super_iri in pairs:
        sub_center, _ = get_class_embedding(model, sub_iri)
        super_center, _ = get_class_embedding(model, super_iri)
        
        if sub_center is not None and super_center is not None:
            # Reshape for sklearn
            sub_vec = sub_center.reshape(1, -1)
            super_vec = super_center.reshape(1, -1)
            
            # Compute cosine similarity between centers
            center_sim = cosine_similarity(sub_vec, super_vec)[0, 0]
            similarities.append(float(center_sim))
        else:
            missing_count += 1
    
    if missing_count > 0:
        print(f"Warning: {missing_count} pairs had missing embeddings")
    
    return similarities


def find_optimal_threshold(similarities: List[float], 
                           threshold_range: np.ndarray,
                           target_mean_cos: float = 0.70) -> Tuple[float, Dict]:
    """Find smallest threshold achieving target mean cosine similarity."""
    print(f"\nSearching for optimal threshold (target mean_cos >= {target_mean_cos})")
    
    results = {}
    optimal_threshold = None
    
    for tau in threshold_range:
        # Filter similarities >= threshold
        filtered = [s for s in similarities if s >= tau]
        
        if len(filtered) > 0:
            mean_cos = np.mean(filtered)
            results[float(tau)] = {
                'mean_cos': float(mean_cos),
                'num_pairs': len(filtered),
                'percentage': len(filtered) / len(similarities) * 100
            }
            
            print(f"  τ = {tau:.2f}: mean_cos = {mean_cos:.4f}, "
                  f"pairs = {len(filtered)}/{len(similarities)} ({results[float(tau)]['percentage']:.1f}%)")
            
            # Check if this threshold meets criterion
            if mean_cos >= target_mean_cos and optimal_threshold is None:
                optimal_threshold = float(tau)
        else:
            results[float(tau)] = {
                'mean_cos': None,
                'num_pairs': 0,
                'percentage': 0.0
            }
            print(f"  τ = {tau:.2f}: No pairs meet threshold")
    
    return optimal_threshold, results


def save_results(output_path: str, 
                 similarities: List[float],
                 optimal_threshold: float,
                 threshold_results: Dict,
                 model_params: Dict):
    """Save evaluation metrics to JSON file."""
    
    metrics = {
        'model': 'ELEmbeddings',
        'model_parameters': model_params,
        'validation_metrics': {
            'total_pairs': len(similarities),
            'all_similarities': {
                'mean': float(np.mean(similarities)),
                'median': float(np.median(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities))
            },
            'optimal_threshold': optimal_threshold,
            'threshold_analysis': threshold_results
        }
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    # Configuration
    TRAIN_PATH = "../train.ttl"
    VALID_PATH = "../valid.ttl"
    OUTPUT_PATH = "reports/mowl_metrics.json"
    
    # Model hyperparameters - Start with original values
    EMBEDDING_DIM = 50
    EPOCHS = 100
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    MARGIN = 0.1  # margin parameter
    
    # Threshold search parameters
    THRESHOLD_MIN = 0.60
    THRESHOLD_MAX = 0.80
    THRESHOLD_STEP = 0.01
    TARGET_MEAN_COS = 0.70
    
    print("=" * 60)
    print("mOWL ELEmbeddings Training Script")
    print("=" * 60)
    
    # Load dataset
    dataset = load_ontology_dataset(TRAIN_PATH, VALID_PATH)
    
    # Extract validation pairs
    validation_pairs = extract_subclass_pairs(dataset, split='validation')
    
    if len(validation_pairs) == 0:
        print("Error: No subclass-superclass pairs found in validation set!")
        return
    
    # Train model
    model_params = {
        'embedding_dim': EMBEDDING_DIM,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'margin': MARGIN
    }
    
    model = train_elembeddings(
        dataset,
        embedding_dim=EMBEDDING_DIM,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        margin=MARGIN
    )
    
    # Compute cosine similarities
    print("\nComputing cosine similarities for validation pairs...")
    similarities = compute_cosine_similarities(model, validation_pairs)
    
    if len(similarities) == 0:
        print("Error: No similarities could be computed!")
        return
    
    print(f"Computed {len(similarities)} similarities")
    print(f"  Mean: {np.mean(similarities):.4f}")
    print(f"  Median: {np.median(similarities):.4f}")
    print(f"  Std: {np.std(similarities):.4f}")
    
    # Find optimal threshold
    threshold_range = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    optimal_threshold, threshold_results = find_optimal_threshold(
        similarities,
        threshold_range,
        TARGET_MEAN_COS
    )
    
    if optimal_threshold is not None:
        print(f"\n✓ Optimal threshold found: τ = {optimal_threshold:.2f}")
        print(f"  Mean cosine similarity: {threshold_results[optimal_threshold]['mean_cos']:.4f}")
        print(f"  Pairs included: {threshold_results[optimal_threshold]['num_pairs']}")
    else:
        print(f"\n✗ No threshold in range [{THRESHOLD_MIN}, {THRESHOLD_MAX}] "
              f"achieves mean_cos >= {TARGET_MEAN_COS}")
        # Use the best threshold available
        valid_thresholds = {k: v for k, v in threshold_results.items() 
                           if v['mean_cos'] is not None}
        if valid_thresholds:
            best_tau = max(valid_thresholds.keys(), 
                          key=lambda k: threshold_results[k]['mean_cos'])
            optimal_threshold = best_tau
            print(f"  Using best available threshold: τ = {optimal_threshold:.2f}")
    
    # Save results
    save_results(OUTPUT_PATH, similarities, optimal_threshold, 
                threshold_results, model_params)
    
    print("\n" + "=" * 60)
    print("Training and evaluation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()