"""
Check what MOWL is actually loading from your ontology.
This will tell us if the problem is in MOWL's data loading.
"""

import jpype
import jpype.imports

if not jpype.isJVMStarted():
    import mowl
    mowl.init_jvm("4g")

from mowl.datasets import PathDataset

TRAIN_PATH = "../train.ttl"
VALID_PATH = "../valid.ttl"

print("="*70)
print("CHECKING WHAT MOWL ACTUALLY LOADS")
print("="*70)

# Load dataset
dataset = PathDataset(TRAIN_PATH, validation_path=VALID_PATH)

# Check training ontology
train_ont = dataset.ontology
print(f"\nTRAINING ONTOLOGY:")
print(f"  Total axioms: {train_ont.getAxiomCount()}")

# Count axiom types
axiom_types = {}
for axiom in train_ont.getAxioms():
    axiom_type = axiom.getAxiomType().getName()
    axiom_types[axiom_type] = axiom_types.get(axiom_type, 0) + 1

print(f"\n  Axiom breakdown:")
for axiom_type, count in sorted(axiom_types.items(), key=lambda x: -x[1]):
    print(f"    {axiom_type}: {count}")

# Count simple subclass axioms specifically
simple_count = 0
restriction_count = 0

for axiom in train_ont.getAxioms():
    if axiom.getAxiomType().getName() == "SubClassOf":
        subclass = axiom.getSubClass()
        superclass = axiom.getSuperClass()
        
        if not subclass.isAnonymous() and not superclass.isAnonymous():
            simple_count += 1
        elif not subclass.isAnonymous() and superclass.isAnonymous():
            restriction_count += 1

print(f"\n  SubClassOf axioms breakdown:")
print(f"    Simple (A ⊑ B): {simple_count}")
print(f"    Restrictions (A ⊑ ∃R.C): {restriction_count}")

# Check classes
classes = set()
for cls in train_ont.getClassesInSignature():
    classes.add(str(cls.getIRI()))

print(f"\n  Classes in signature: {len(classes)}")

# Check properties
properties = set()
for prop in train_ont.getObjectPropertiesInSignature():
    properties.add(str(prop.getIRI()))

print(f"  Object properties: {len(properties)}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if simple_count == 0:
    print("\n❌ CRITICAL: MOWL found ZERO simple subclass axioms!")
    print("   This explains why embeddings are random (0.33 similarity).")
    print("   The model has no hierarchical structure to learn from!")
    print("\n   POSSIBLE CAUSES:")
    print("   1. All subClassOf axioms are restrictions (A ⊑ ∃R.C)")
    print("   2. Axioms are using complex class expressions")
    print("   3. File format issue preventing MOWL from parsing")
    
elif simple_count < 100:
    print(f"\n⚠️  WARNING: Only {simple_count} simple subclass axioms!")
    print(f"   This is very sparse for {len(classes)} classes.")
    print(f"   Density: {simple_count/len(classes):.2f} axioms per class")
    print("\n   RECOMMENDATION:")
    print("   Need at least 2-3 axioms per class for good embeddings.")
    
else:
    print(f"\n✓ Training data looks OK:")
    print(f"   {simple_count} simple axioms for {len(classes)} classes")
    print(f"   Density: {simple_count/len(classes):.2f} axioms per class")
    print("\n   The problem is likely elsewhere (hyperparameters or training bug)")

# Sample some axioms
print("\n" + "="*70)
print("SAMPLE AXIOMS (first 10):")
print("="*70)

count = 0
for axiom in train_ont.getAxioms():
    if axiom.getAxiomType().getName() == "SubClassOf":
        subclass = axiom.getSubClass()
        superclass = axiom.getSuperClass()
        
        if not subclass.isAnonymous() and not superclass.isAnonymous():
            sub_iri = str(subclass.asOWLClass().getIRI())
            sup_iri = str(superclass.asOWLClass().getIRI())
            print(f"{count+1}. {sub_iri} ⊑ {sup_iri}")
            count += 1
            if count >= 10:
                break

if count == 0:
    print("(No simple subclass axioms found)")

print("\n" + "="*70)