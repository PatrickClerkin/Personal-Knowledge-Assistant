from sentence_transformers import SentenceTransformer
import numpy as np

print("=" * 70)
print("TESTING EMBEDDINGS")
print("=" * 70)

# Load a small, fast model
print("\nLoading model: all-MiniLM-L6-v2")
print("(This will download ~90MB on first run)")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test sentences - some similar, some different
sentences = [
    "Object-oriented programming uses composition and inheritance",
    "OOP relies on classes and objects to structure code",
    "Machine learning models require training data",
    "Neural networks learn patterns from examples",
    "The weather is nice today",
    "I enjoy sunny days"
]

print("\nEmbedding sentences...")
embeddings = model.encode(sentences)

print(f"\n✅ Embeddings generated!")
print(f"Shape: {embeddings.shape}")
print(f"Each sentence → {embeddings.shape[1]}-dimensional vector")

# Calculate similarities
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n" + "=" * 70)
print("SIMILARITY MATRIX")
print("=" * 70)
print("(1.0 = identical, 0.0 = unrelated, higher = more similar)\n")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        similarity = cosine_similarity(embeddings[i], embeddings[j])
        print(f"[{i}] vs [{j}]: {similarity:.3f}")
        print(f"  '{sentences[i][:50]}...'")
        print(f"  '{sentences[j][:50]}...'")
        print()

print("=" * 70)
print("EXPECTED RESULTS:")
print("=" * 70)
print("• Sentences 0-1 (both about OOP): HIGH similarity (~0.5-0.7)")
print("• Sentences 2-3 (both about ML): HIGH similarity (~0.5-0.7)")
print("• Sentences 4-5 (both about weather): HIGH similarity (~0.4-0.6)")
print("• Sentences 0-4 (OOP vs weather): LOW similarity (~0.0-0.2)")
print("=" * 70)