import os, mowl, torch
mowl.init_jvm("2g")
from mowl.datasets import PathDataset
from mowl.models import ELEmbeddings



ds =PathDataset("toy_train.ttl")

os.makedirs("checkpoints", exist_ok=True)

model = ELEmbeddings(ds, embed_dim=16, epochs=6)
model.save_path = "checkpoints"
model.model_filepath = "checkpoints/elembeddings_best.pt"

model.train(epochs=6)

emb = model.get_embeddings()
cos = torch.nn.functional.cosine_similarity

dog = "http://example.org#Dog"
mammal="http://example.org#Mammal"
animal="http://example.org#Animal"

print("sim(Dog, Mammal):", float(cos(emb[dog], emb[mammal],dim=0)))
print("sim(Mammal, Animal):", float(cos(emb[mammal], emb[animal], dim=0)))
print("sim(Dog, Animal):", float(cos(emb[dog], emb[animal], dim=0)))

print("\n[TEST] Held-out axiom: Dog = Animal")
print("Plausibility (cosine):", float(cos(emb[dog], emb[animal], dim=0)))