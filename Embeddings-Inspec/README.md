# Embeddings Inspector - Semantic Similarity Explorer

An **interactive learning tool** built with **LangChain** and **GitHub Models API** that demonstrates how embeddings work and how to measure semantic similarity between text using cosine similarity.

This application is a **hands-on exploration tool** that helps you understand the foundation of modern RAG systems: **text embeddings and semantic meaning**.

## 🎯 Overview

EmbdApp generates embeddings for text sentences and calculates **cosine similarity** - a metric that measures how semantically similar two pieces of text are.

### What It Does

- **Text Embedding**: Converts sentences into high-dimensional vectors (1536 dimensions)
- **Cosine Similarity**: Measures the angle between vectors (0 = opposite, 1 = identical meaning)
- **Interactive Exploration**: 6 experiments + custom input mode
- **Visual Comparisons**: See how different phrases relate to each other
- **Semantic Understanding**: Discover what embeddings actually measure

### Why This Matters

Embeddings are the **foundation** of RAG systems:

```
Text Document
    ↓
Embedding Model (text-embedding-3-small)
    ↓
1536-Dimensional Vector
    ↓
Vector Database (searchable)
```

Understanding embeddings helps you design better RAG systems!

## 🚀 Features

### Pre-Built Experiments

1. **Standard Lab Sentences** - Baseline semantic similarity
   - "The canine barked loudly"
   - "The dog made a noise"
   - "The electron spins rapidly"

2. **Same Sentence, Different Languages** - Cross-lingual similarity
   - English, Spanish, and French versions of "The dog barked"

3. **Change One Word** - Impact of vocabulary
   - Modify "cat" to "dog" or "mat" to "hat"

4. **Add Details** - Effect of elaboration
   - "I like pizza" vs detailed pizza description

5. **Opposite Meanings** - Semantic opposition
   - "Excellent and entertaining" vs "Terrible and boring"

6. **Technical vs. Casual Language** - Register variations
   - Formal weather statement vs casual phrasing

7. **Different Topics** - Low similarity baseline
   - Pizza, programming, and the ocean

### Interactive Features

- **Custom Sentences**: Enter your own text to test
- **Real-Time Calculation**: Instant similarity results
- **Error Handling**: Graceful handling of edge cases
- **Repeatable Testing**: Menu-driven interface

## 📋 Prerequisites

- Python 3.8+
- GitHub personal access token (with access to GitHub Models)
- 2MB+ of disk space for dependencies

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install langchain-openai langchain python-dotenv
```

### 2. Configure Environment Variables

Create a `.env` file in the `Embeddings-Inspec/` directory:

```env
API_KEY=your_github_token_here
API_BASE_URL=https://models.inference.ai.azure.com
```

Or use the parent directory's `.env`:
```env
GITHUB_TOKEN=your_github_token_here
API_BASE_URL=https://models.inference.ai.azure.com
```

## 🖥️ Usage

### Run the Application

```bash
python3 EmbdApp.py
```

### Interactive Menu

```text
🤖 Python LangChain Agent Starting...

==============================================
          EMBEDDING INSPECTOR MENU            
==============================================
1. Run Standard Lab Sentences
2. Run Experiment 1: Same Sentence in Different Languages
3. Run Experiment 2: Change Just One Word
4. Run Experiment 3: Add Adjectives and Details
5. Run Experiment 4: Opposite Meanings
6. Run Experiment 5: Technical vs. Casual Language
7. Run Experiment 6: Different Topics
8. Enter Custom Sentences
9. Exit
==============================================

Select an option (1-9): 
```

### Example: Running Standard Lab

```text
Select an option (1-9): 1

--- Running: Standard Lab Sentences ---

Generating embeddings for three sentences...
Sentence 1: "The canine barked loudly."
Sentence 2: "The dog made a noise."
Sentence 3: "The electron spins rapidly."

=== Cosine Similarities ===
Cosine similarity between Sentence 1 and Sentence 2: 0.8234
Cosine similarity between Sentence 2 and Sentence 3: 0.2145
Cosine similarity between Sentence 3 and Sentence 1: 0.1987
```

### Interpreting Results

| Similarity | Meaning | Example |
|-----------|---------|---------|
| 0.95-1.0 | Nearly identical | "dog" vs "dog" |
| 0.85-0.95 | Very similar | "canine" vs "dog" |
| 0.70-0.85 | Similar | "I like pizza" vs detailed pizza description |
| 0.50-0.70 | Somewhat related | "good weather" vs "bad weather" |
| 0.30-0.50 | Loosely related | "pizza" vs "programming" |
| 0.0-0.30 | Very different | "pizza" vs "ocean" |

## 🏗️ Architecture

### How Embeddings Work

```
Text Input
    ↓
┌────────────────────────────────┐
│ Embedding Model                │
│ (text-embedding-3-small)       │
│                                │
│ • Tokenizes text               │
│ • Processes through neural net │
│ • Outputs 1536-D vector        │
└────────┬───────────────────────┘
         ↓
    [0.23, -0.15, 0.89, ..., 0.12]
    (1536 dimensions)
         ↓
    ┌────────────────────┐
    │  Vector Database   │
    │  (Searchable)      │
    └────────────────────┘
```

### Cosine Similarity Calculation

Cosine similarity measures the angle between two vectors:

```
Cosine Similarity = (v1 · v2) / (||v1|| × ||v2||)

Where:
  • v1 · v2 = dot product (how aligned are they)
  • ||v1|| = magnitude of v1
  • ||v2|| = magnitude of v2

Range: -1 to 1
  • 1 = same direction (same meaning)
  • 0 = perpendicular (unrelated)
  • -1 = opposite direction (opposite meaning)
```

### Application Flow

```
┌─────────────────┐
│  Load .env      │
│  Get API Key    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Initialize Embedding Model      │
│ (text-embedding-3-small)        │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Show Menu                       │
│ Get User Choice                 │
└────────┬────────────────────────┘
         │
         ├──→ Experiment 1-7
         │    • Get sentences
         │    • Embed each
         │    • Calculate similarity
         │    • Display results
         │
         ├──→ Custom Input (8)
         │    • Prompt for 3 sentences
         │    • Run same logic
         │
         └──→ Exit (9)
```

## 🔍 Key Functions

### `cosine_similarity(v1, v2)`

Calculates cosine similarity between two embedding vectors.

```python
similarity = cosine_similarity(embedding1, embedding2)
# Returns float between 0 and 1
```

**Parameters**:
- `v1`: First embedding vector
- `v2`: Second embedding vector

**Returns**: Similarity score (0 = unrelated, 1 = identical)

**Implementation**:
- Computes dot product of vectors
- Normalizes by magnitude
- Handles zero-magnitude edge case

---

### `run_inspector(embeddings, sentences)`

Generates embeddings and calculates all pairwise similarities.

```python
run_inspector(embeddings, [
    "The dog barked.",
    "The canine made noise.",
    "The weather is nice."
])
```

**Parameters**:
- `embeddings`: OpenAIEmbeddings instance
- `sentences`: List of 3 strings to compare

**Output**:
- Confirmation of each sentence
- All pairwise cosine similarities
- Formatted for readability

---

### `main()`

Interactive menu system for experimenting with embeddings.

**Flow**:
1. Loads environment variables
2. Initializes embedding model
3. Displays menu
4. Routes user choice to appropriate experiment or custom input
5. Repeats until user exits

## 📚 Understanding Embeddings

### What Are Embeddings?

Embeddings are **numerical representations** of text that capture semantic meaning:

```
"The dog barked" → [0.23, -0.15, 0.89, ..., 0.12]
"The canine barked" → [0.24, -0.16, 0.88, ..., 0.11]
                       ↑ Similar but slightly different
```

### Why Are They Useful?

1. **Semantic Search**: Find similar documents without keywords
2. **Clustering**: Group similar texts together
3. **Classification**: Determine text categories
4. **Recommendation**: Suggest related content
5. **RAG Foundation**: Core technology for modern AI systems

### How Do Models Learn Meanings?

Modern embedding models (like text-embedding-3-small):

1. **Train on billions** of text examples
2. **Learn patterns** of word meanings
3. **Capture relationships** (king - man + woman ≈ queen)
4. **Encode semantics** into 1536 dimensions
5. **Produce consistent** representations

## 🎓 Experiment Insights

### What You'll Discover

**Experiment 1: Same Meaning, Different Words**
- High similarity (0.82+) despite different words
- "canine" and "dog" understood as equivalent
- Demonstrates semantic understanding

**Experiment 2: Cross-Lingual Similarity**
- Same meaning in different languages shows high similarity
- Model learns that concepts translate
- Useful for multilingual RAG systems

**Experiment 3: Single Word Changes**
- Changing "cat" to "dog" increases similarity to dog-related sentences
- Changing "mat" to "hat" shows sensitivity to object context
- Demonstrates attention to meaning, not just words

**Experiment 4: Elaboration Effects**
- Detailed description = very high similarity to simple version
- Model captures core meaning despite added details
- Robustness to paraphrasing

**Experiment 5: Opposite Meanings**
- "Excellent" and "terrible" show low similarity
- Model understands semantic opposition
- Important for sentiment-aware RAG

**Experiment 6: Register Variations**
- Formal, casual, and technical versions cluster together
- Meaning preserved across language styles
- Shows robustness to phrasing variations

**Experiment 7: Unrelated Topics**
- Pizza, programming, ocean show low mutual similarity
- Baseline for comparison
- Confirms model distinguishes topics

## 🚀 Common Questions

### Q: Why 1536 dimensions?

**A**: The text-embedding-3-small model outputs 1536-dimensional vectors. This is chosen by OpenAI to balance:
- Information capacity (more dimensions = more detail)
- Computational efficiency (fewer dimensions = faster)
- Performance quality (1536 is empirically optimal)

---

### Q: What's a "good" similarity score?

**A**: Depends on your use case:
- **Search**: Typically use scores > 0.7 as relevant
- **Clustering**: Group scores > 0.8 together
- **Deduplication**: Merge texts with > 0.95 similarity
- **Recommendations**: Show scores > 0.6 as related

---

### Q: Why cosine and not Euclidean distance?

**A**: Cosine similarity has advantages:
- **Normalized**: Range from -1 to 1 (interpretable)
- **Angle-based**: Captures direction, not magnitude
- **Independent of scale**: Works with any embedding size
- **Fast computation**: Efficient for high dimensions

---

### Q: Can I use this for production?

**A**: This is an **explorer/learning tool**. For production RAG:
- Use persistent vector databases (Weaviate, Pinecone, etc.)
- Add filtering, metadata, and hybrid search
- Implement caching and optimization
- Add monitoring and quality metrics

## 🔧 Extending the Application

### Add More Experiments

```python
experiments["8"] = ("My Experiment", [
    "Custom sentence 1",
    "Custom sentence 2",
    "Custom sentence 3"
])
```

### Calculate Vector Magnitude

```python
def vector_magnitude(v):
    return math.sqrt(sum(a * a for a in v))

# Then display in results:
mag1 = vector_magnitude(vector1)
print(f"Magnitude of vector 1: {mag1:.4f}")
```

### Visualize Vectors (Advanced)

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce 1536D vectors to 2D for visualization
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
plt.show()
```

### Batch Processing

```python
# Process multiple groups of sentences:
sentence_groups = [
    ["dog", "canine", "puppy"],
    ["happy", "joyful", "sad"],
]

for group in sentence_groups:
    run_inspector(embeddings, group)
```

## 📊 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| First embedding call | ~1-2s | Model initialization |
| Subsequent embeddings | ~200-300ms | Per sentence |
| Cosine similarity calc | <1ms | Simple math |
| Full experiment | ~1-2s | 3 embeddings + 3 similarities |

## ⚠️ Common Issues

### Issue: API Key Not Found

**Error**: `API_KEY or GITHUB_TOKEN not found in environment variables.`

**Solution**:
1. Create `.env` file in this directory or parent
2. Add your GitHub token
3. Restart the application

---

### Issue: Embeddings Take Long Time

**Cause**: First call includes model initialization

**Expected**: Subsequent calls are much faster (~200ms)

---

### Issue: Different Results Each Time

**Note**: Embeddings are deterministic, not random. Same input = same output every time.

## 📚 Learning Path

EmbdApp is **Lab 1: Embeddings** in the RAG fundamentals course.

### Course Progression

```
Lab 1: Embeddings (THIS LAB)
  └─ Understand vectors and semantic similarity
     └─ Lab 2: Vector Storage
        └─ Store and retrieve embeddings
           └─ Lab 3: Semantic Search
              └─ Search across documents
                 └─ Lab 4: Chunking Strategies
                    └─ Handle large documents
                       └─ Lab 5: RAG with LLM
                          └─ Complete end-to-end RAG
```

## 📖 Documentation Links

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [LangChain Embeddings](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
- [GitHub Models API](https://docs.github.com/en/github-models/prototyping-with-ai-models)

## 💡 Key Concepts

✅ **Embeddings** capture semantic meaning in vectors

✅ **Cosine similarity** measures semantic relatedness (0-1)

✅ **Similar meanings** produce high similarity scores

✅ **Different topics** produce low similarity scores

✅ **Language variations** (formal/casual) preserve similarity

✅ **This foundation** enables modern RAG systems

## 🎯 Next Steps

After exploring embeddings with EmbdApp:

1. **Store Embeddings**: See [../Document-Storage/README.md](../Document-Storage/README.md)
2. **Search with Embeddings**: Implement semantic search
3. **Handle Large Documents**: See [../Chunking/README.md](../Chunking/README.md)
4. **Build RAG Systems**: Create retrieval-augmented generation agents
5. **Deploy Professionally**: Scale to production workloads

---

**Happy exploring!** 🚀 

Try the experiments and notice how meaning is preserved across language variations, elaborations, and different expressions. That's the power of embeddings!

**Questions?** Review the inline code comments in EmbdApp.py for implementation details.
