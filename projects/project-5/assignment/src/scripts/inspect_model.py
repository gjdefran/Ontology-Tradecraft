import torch
from pathlib import Path

# Load the model
model_path = Path(__file__).parent / 'models' / 'mowl_best.pt'
print(f"Loading: {model_path}")

checkpoint = torch.load(model_path, map_location='cpu')

print("\n" + "="*60)
print("CHECKPOINT STRUCTURE")
print("="*60)

if isinstance(checkpoint, dict):
    print("\nTop-level keys:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if hasattr(value, 'shape'):
            print(f"  {key}: {type(value).__name__} with shape {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} items")
            # Show first few keys if it's a mapping
            if len(value) <= 10:
                for k in list(value.keys())[:5]:
                    print(f"    - {k}: {type(value[k]).__name__}")
            else:
                print(f"    - First 3 keys: {list(value.keys())[:3]}")
        elif isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
else:
    print(f"\nCheckpoint is a {type(checkpoint).__name__}")
    if hasattr(checkpoint, 'shape'):
        print(f"Shape: {checkpoint.shape}")

print("\n" + "="*60)

# Look for embeddings specifically
print("\nLOOKING FOR EMBEDDINGS:")
print("="*60)

if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        if 'embed' in key.lower() or 'weight' in key.lower():
            value = checkpoint[key]
            print(f"\n'{key}':")
            print(f"  Type: {type(value).__name__}")
            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
            elif isinstance(value, dict):
                print(f"  Dict with {len(value)} entries")
                print(f"  Sample keys: {list(value.keys())[:3]}")

# Look for class mappings
print("\n" + "="*60)
print("LOOKING FOR CLASS MAPPINGS:")
print("="*60)

if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        if 'class' in key.lower() or 'map' in key.lower() or 'id' in key.lower():
            value = checkpoint[key]
            print(f"\n'{key}':")
            print(f"  Type: {type(value).__name__}")
            if isinstance(value, dict):
                print(f"  Dict with {len(value)} entries")
                print(f"  Sample entries:")
                for k, v in list(value.items())[:3]:
                    print(f"    {k}: {v}")