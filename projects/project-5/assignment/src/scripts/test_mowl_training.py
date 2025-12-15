import mowl
mowl.init_jvm("5g")
from mowl.datasets import PathDataset
from mowl.projection import TaxonomyProjector, OWL2VecStarProjector
import torch as th
import numpy as np
import json

class SubsumptionDataset(PathDataset):
    @property
    def evaluation_classes(self):
        return self.classes, self.classes

def main():
    dataset = SubsumptionDataset(
        ontology_path="../train.ttl",
        validation_path="../valid.ttl",
        testing_path="../valid.ttl"
    )

    # Train OWL2Vec* model
    model = OWL2VecStarProjector(dataset)
    model.train()

    # Get embeddings (flat vectors - cosine similarity works!)
    embeddings = model.get_embeddings()
    
    # Convert to torch tensors
    emb = {k: th.tensor(v) for k, v in embeddings.items()}
    cos = th.nn.functional.cosine_similarity

    # Extract held-out subsumption pairs from validation ontology
    projector = TaxonomyProjector()
    edges = projector.project(dataset.validation)

    # Compute cosine similarities
    similarities = []
    for edge in edges:
        subclass = edge.src
        superclass = edge.dst
        
        if subclass in emb and superclass in emb:
            sim = float(cos(emb[subclass], emb[superclass], dim=0))
            similarities.append(sim)
            print(f"sim({subclass}, {superclass}): {sim:.4f}")

    mean_cos = np.mean(similarities)
    print(f"\nMean cosine similarity: {mean_cos:.4f}")

    # Find smallest threshold τ ∈ {0.60 – 0.80} achieving mean_cos ≥ 0.70
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    best_threshold = None
    
    for tau in thresholds:
        above = [s for s in similarities if s >= tau]
        if len(above) > 0:
            mean_above = np.mean(above)
            coverage = len(above) / len(similarities)
            print(f"τ={tau:.2f}: mean={mean_above:.4f}, coverage={coverage:.1%}")
            
            if mean_above >= 0.70 and best_threshold is None:
                best_threshold = tau

    print(f"\nSmallest threshold achieving mean_cos ≥ 0.70: τ = {best_threshold}")

    # Save results
    results = {
        "mean_cosine_similarity": float(mean_cos),
        "num_pairs": len(similarities),
        "threshold": best_threshold
    }
    
    with open("reports/mowl_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
