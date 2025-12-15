
import torch
import numpy as np
import pickle
import json
from pathlib import Path

# Input checkpoint
checkpoint_path = ("models/mowl_best.pt")

# Output files
embeddings_out = Path("embeddings.npy")
mappings_out = Path("mappings.pkl")
metrics_out = Path("mowl_metrics.json")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Extract class embeddings
class_embeddings = checkpoint['class_embed.weight'].cpu().numpy()
n_classes, embedding_dim = class_embeddings.shape

# Save embeddings
np.save(embeddings_out, class_embeddings)

# Build synthetic mapping (replace with real IRIs if available)
class_to_id = {f"http://example.org/class/{i}": i for i in range(n_classes)}
with open(mappings_out, 'wb') as f:
    pickle.dump({'class_to_id': class_to_id}, f)

# Create metrics JSON
metrics = {
    "embeddings_file": str(embeddings_out.resolve()),
    "mappings_file": str(mappings_out.resolve()),
    "hyperparameters": {"embedding_dim": embedding_dim},
    "n_classes": n_classes
}
with open(metrics_out, 'w') as f:
    json.dump(metrics, f, indent=4)

print("Export complete:")
print(f"- Embeddings: {embeddings_out}")
print(f"- Mappings: {mappings_out}")
print(f"- Metrics: {metrics_out}")
print(f"Embedding shape: {class_embeddings.shape}")
