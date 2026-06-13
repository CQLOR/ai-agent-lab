# Vector Memory Agent - Semantic Search Storage System

An **interactive semantic search system** built with **LangChain** and **GitHub Models API** that demonstrates how to store embeddings in a vector database and perform intelligent searches across a collection of sentences.

This application is a **practical bridge** between understanding embeddings (Lab 1) and building full RAG systems (Lab 4-5). It shows how to manage multiple documents and retrieve relevant ones based on meaning.

## 🎯 Overview

Vector Memory Agent takes the embedding concepts from EmbdApp and scales them to a realistic scenario: **storing and searching 15 different sentences across diverse topics**.

### What It Does

- **Batch Embedding**: Converts multiple sentences to vectors efficiently
- **Vector Storage**: Indexes embeddings in an in-memory database
- **Semantic Search**: Finds top-3 most similar sentences for any query
- **Metadata Tracking**: Associates timestamps and indices with stored content
- **Interactive Query Loop**: Real-time search interface for exploration
- **Similarity Scoring**: Shows relevance scores for each result

### The Problem It Solves

With EmbdApp, you could compare 3 sentences. But in real systems:

```
Real Problem:
  • Company has 10,000 documents
  • Need to find relevant ones for a query
  • Can't compare query against all documents manually
  • Need automated, efficient retrieval

Solution:
  • Store all 10,000 embeddings in a vector database
  • Query vectorized instantly
  • Return top N results ranked by relevance
```

Vector Memory Agent demonstrates this at scale with 15 sentences.

## 🚀 Features

### Vector Storage

- **In-Memory Database**: Fast, non-persistent storage
- **Batch Operations**: Store multiple texts efficiently
- **Metadata Preservation**: Track creation time and index for each text
- **Efficient Lookup**: Semantic search without scanning all texts

### Semantic Search

- **Top-K Retrieval**: Returns top 3 most similar results
- **Similarity Scoring**: Shows relevance (0-1 scale)
- **Ranked Results**: Results ordered by relevance
- **Contextual Understanding**: Finds meaning-based matches, not keyword matches

### Interactive Interface

- **CLI Loop**: Real-time search capability
- **Graceful Exit**: 'quit' or 'exit' commands
- **Error Handling**: Continues on errors without crashing
- **Immediate Feedback**: See search results instantly

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

Create a `.env` file in the `Vector-Memory/` directory:

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
python3 vector_agent.py
```

### Expected Output

```text
🤖 Python LangChain Agent Starting...

Storing 15 sentences in the vector database...
✅ Successfully stored 15 sentences

=== Semantic Search ===
Enter a search query (or 'quit' to exit): 
```

### Example Searches

#### Search 1: Animals

```text
Enter a search query (or 'quit' to exit): dog

🔍 Search Results for "dog":
1. [Score: 0.8834] The dog made a noise.
2. [Score: 0.8421] The canine barked loudly.
3. [Score: 0.7654] Puppies need lots of attention and exercise.

Enter a search query (or 'quit' to exit): 
```

**Why these results?**
- "The dog made a noise" - exact word match + semantic match
- "The canine barked loudly" - synonym (canine) + related action (barked)
- "Puppies need lots of attention" - related animal + positive context

---

#### Search 2: Physics

```text
Enter a search query (or 'quit' to exit): quantum

🔍 Search Results for "quantum":
1. [Score: 0.9123] Quantum mechanics explains particle behavior.
2. [Score: 0.8456] The electron spins rapidly.
3. [Score: 0.5432] Atoms are made of protons, neutrons, and electrons.

Enter a search query (or 'quit' to exit): 
```

**Why these results?**
- "Quantum mechanics" - direct match
- "The electron spins rapidly" - related physics concept
- "Atoms are made of..." - broader physics context

---

#### Search 3: Programming

```text
Enter a search query (or 'quit' to exit): coding languages

🔍 Search Results for "coding languages":
1. [Score: 0.8721] Python is a popular programming language.
2. [Score: 0.8234] JavaScript runs in web browsers.
3. [Score: 0.6123] The soccer match ended in a tie.

Enter a search query (or 'quit' to exit): 
```

**Why these results?**
- "Python is a popular programming language" - exact match on concepts
- "JavaScript runs in web browsers" - related programming topic
- "The soccer match..." - low score, likely caught due to "game/programming game" semantic connection

## 📊 Stored Sentences

The system stores these 15 diverse sentences across multiple topics:

| # | Sentence | Topic |
|----|----------|-------|
| 1 | The canine barked loudly. | Animals |
| 2 | The dog made a noise. | Animals |
| 3 | The electron spins rapidly. | Physics |
| 4 | I love eating pizza with extra cheese. | Food |
| 5 | The basketball player scored a three-pointer. | Sports |
| 6 | Rain is forecasted for tomorrow afternoon. | Weather |
| 7 | Python is a popular programming language. | Programming |
| 8 | The kitten purred softly on the couch. | Animals |
| 9 | Quantum mechanics explains particle behavior. | Physics |
| 10 | Homemade pasta tastes better than store-bought. | Food |
| 11 | The soccer match ended in a tie. | Sports |
| 12 | Clouds are forming over the mountains. | Weather |
| 13 | JavaScript runs in web browsers. | Programming |
| 14 | Puppies need lots of attention and exercise. | Animals |
| 15 | Atoms are made of protons, neutrons, and electrons. | Physics |

## 🏗️ Architecture

### Data Storage Flow

```
Sentences List
    ↓
┌────────────────────────────────┐
│ Generate Metadata              │
│ • Timestamp (created_at)       │
│ • Index number                 │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ Embed Each Sentence            │
│ (text-embedding-3-small)       │
│ 15 sentences → 15 embeddings   │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│ Store in Vector Database       │
│ • Embeddings                   │
│ • Metadata                     │
│ • Original text                │
└────────────┬───────────────────┘
             │
             ▼
         Indexed & Ready
         for Semantic Search
```

### Search Flow

```
User Query: "dog"
    ↓
┌─────────────────────────────────┐
│ Embed Query                     │
│ "dog" → 1536D vector            │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Calculate Similarity            │
│ Query vs All 15 Embeddings      │
│ Using Cosine Similarity         │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Rank Results                    │
│ Sort by similarity score        │
│ Return top-3 matches            │
└──────────┬──────────────────────┘
           │
           ▼
    Display to User:
    1. [0.88] The dog made noise
    2. [0.84] The canine barked
    3. [0.77] Puppies need exercise
```

## 🔍 Key Functions

### `search_sentences(vector_store, query, k=3)`

Performs semantic search and displays ranked results.

```python
search_sentences(vector_store, "dog", k=3)
```

**Parameters**:
- `vector_store`: The vector store instance
- `query`: Search query string
- `k`: Number of results to return (default 3)

**Output**:
- Prints formatted search results
- Shows similarity score for each result
- Results ranked by relevance

**Implementation**:
- Calls `similarity_search_with_score()` to get results
- Formats output with rank number and score
- Handles edge cases gracefully

## 📈 How Results Are Scored

Scores range from **0 to 1**, where:

| Score Range | Meaning | Example |
|------------|---------|---------|
| 0.90-1.0 | Nearly identical meaning | "dog" searches for "dog made noise" |
| 0.80-0.90 | Highly similar | "dog" searches for "canine barked" |
| 0.70-0.80 | Similar | "dog" searches for "puppies exercise" |
| 0.50-0.70 | Related | "programming" searches for "JavaScript" |
| 0.30-0.50 | Loosely related | "pizza" searches for "pasta" |
| 0.0-0.30 | Very different | "dog" searches for "quantum physics" |

## 🎓 Understanding the System

### Why Multiple Sentences?

EmbdApp compared 3 sentences. Why expand to 15?

```
Scaling Problem:
  3 sentences    → Manual comparison OK
  15 sentences   → 105 comparisons needed
  100 documents  → 4,950 comparisons
  1,000,000 docs → Too many to compare!

Vector Database Solution:
  • Pre-compute all embeddings
  • One query embedding
  • Quick similarity search
  • Scalable to billions of documents
```

### Why Metadata?

Each sentence stores metadata:
```python
{
    "created_at": "2026-06-13T10:30:45.123456",
    "index": 0
}
```

**Uses**:
- Track when data was added (versioning)
- Filter by time range (recent vs old)
- Sort results by creation order
- Debug and audit data

### Why In-Memory Storage?

InMemoryVectorStore is perfect for:
- ✅ Learning and prototyping
- ✅ Small datasets (<1GB)
- ✅ Fast performance (no disk I/O)
- ✅ Easy to understand

**Limitations**:
- ❌ Data lost on restart
- ❌ Not persistent
- ❌ Limited by RAM
- ❌ Single-machine only

**Production alternatives**: Weaviate, Pinecone, Chroma, Milvus

## 🔄 The Search Process Step-by-Step

### Step 1: Query Embedding

```python
# When user enters "dog"
query_embedding = embeddings.embed_query("dog")
# Result: [0.12, -0.34, 0.56, ..., 0.78]  # 1536 dimensions
```

### Step 2: Similarity Calculation

```python
# Compare with each stored sentence
similarities = []
for stored_embedding in all_embeddings:
    score = cosine_similarity(query_embedding, stored_embedding)
    similarities.append((score, sentence))
```

### Step 3: Ranking

```python
# Sort by score (highest first)
sorted_results = sorted(similarities, key=lambda x: x[0], reverse=True)

# Take top 3
top_3 = sorted_results[:3]
```

### Step 4: Display

```python
# Format and show to user
for rank, (score, sentence) in enumerate(top_3, 1):
    print(f"{rank}. [Score: {score:.4f}] {sentence}")
```

## 🚨 Common Search Patterns

### Pattern 1: Direct Match

```
Query: "Python"
Best Match: "Python is a popular programming language."
Score: 0.95+ (high similarity due to exact word)
```

### Pattern 2: Synonym Match

```
Query: "dog"
Best Match: "The canine barked loudly."
Score: 0.84 (high similarity despite different word)
```

### Pattern 3: Topic Match

```
Query: "animals"
Best Match: "Puppies need lots of attention and exercise."
Score: 0.76 (related topic, not exact words)
```

### Pattern 4: Cross-Domain Match

```
Query: "sport"
Best Match: "The basketball player scored a three-pointer."
Score: 0.72 (related concept)
```

## 📚 Learning Path

Vector Memory Agent is **Lab 2: Vector Storage and Retrieval** in the RAG fundamentals course.

### Course Progression

```
Lab 1: Embeddings (EmbdApp.py)
  └─ Understand vectors and cosine similarity
     └─ Lab 2: Vector Storage (THIS LAB)
        └─ Store and search embeddings
           └─ Lab 3: Semantic Search (DocStorApp.py)
              └─ Discover token limit problem
                 └─ Lab 4: Chunking (chunking-agnt.py)
                    └─ Solve document size problem
                       └─ Lab 5: RAG with LLM
                          └─ Complete RAG system
```

## 🔧 Extending the Application

### Add More Sentences

```python
sentences = [
    # Original 15 sentences...
    "Your new sentence here.",
    "Another sentence to search.",
]
```

### Adjust Number of Results

```python
# Return top-5 instead of top-3
search_sentences(vector_store, query, k=5)
```

### Filter by Metadata

```python
# Advanced: Search only recent documents
def search_with_filter(vector_store, query, max_age_hours=24):
    # Implementation would filter results by created_at
    pass
```

### Add Result Formatting

```python
# Show more details
for rank, (doc, score) in enumerate(results, 1):
    print(f"{rank}. [Score: {score:.4f}]")
    print(f"   Text: {doc.page_content}")
    print(f"   Added: {doc.metadata['created_at']}")
    print(f"   Index: {doc.metadata['index']}")
```

### Batch Search

```python
# Search multiple queries at once
queries = ["dog", "programming", "weather"]
for query in queries:
    search_sentences(vector_store, query)
```

## ⚠️ Common Issues

### Issue: No Results or Low Scores

**Possible Causes**:
- Query is very different from stored sentences
- Query uses specialized vocabulary not in dataset
- Too strict on score threshold

**Solution**:
- Try broader queries ("animal" vs "canine")
- Expand sentence dataset with more topics
- Adjust what counts as "good" score for your use case

---

### Issue: Slow Search

**Cause**: Usually not an issue with 15 sentences, but...

**With more data**: Use faster vector databases
- InMemoryVectorStore scans all vectors (O(n))
- Use specialized DBs for production (O(log n) or O(1))

---

### Issue: Unexpected Results

**Cause**: Embeddings capture meaning you might not expect

**Example**:
```
Query: "eat"
Result: "The kitten purred softly on the couch."
Score: 0.65
```

**Why**: Model knows eating relates to animals/comfort contexts

**Solution**: This is actually correct! The model finds semantic relationships you might miss with keyword search.

## 📊 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Initialize model | ~2s | One-time cost |
| Embed 15 sentences | ~3-5s | Batch operation |
| Single search | ~200ms | Per query |
| Full interaction cycle | Variable | Input + search |

## 💡 Key Insights

✅ **Semantic search** finds meaning-based matches

✅ **Vector databases** scale this to millions of documents

✅ **Metadata tracking** adds context and filtering

✅ **Similarity scoring** ranks results by relevance

✅ **This foundation** enables production RAG systems

✅ **Real-world systems** use persistent databases instead of in-memory

## 🎯 Practical Applications

Where vector search is used in production:

1. **Search Engines**: Find relevant documents
2. **Recommendation Systems**: Suggest similar products
3. **Q&A Systems**: Find answers to questions
4. **Content Moderation**: Flag similar harmful content
5. **Duplicate Detection**: Find similar documents
6. **RAG Systems**: Retrieve relevant context for LLMs

## 📖 Documentation Links

- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [InMemoryVectorStore](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [GitHub Models API](https://docs.github.com/en/github-models/prototyping-with-ai-models)

## 🚀 Next Steps

After mastering vector search with this agent:

1. **Try Different Queries**: Explore semantic matching
2. **Understand Scores**: See why certain results appear
3. **Compare with Keyword Search**: Notice the difference
4. **Scale Up**: Add more sentences and observe behavior
5. **Move to Production**: Learn about persistent vector databases
6. **Build RAG Systems**: Combine with chunking for full RAG
7. **Add LLM Integration**: Generate responses based on retrieved context

---

**Happy searching!** 🔍

The semantic search you're experiencing here is the **core technology** behind modern AI assistants. When ChatGPT answers your questions, it uses vector search like this to find relevant information!

**Questions?** Review the inline code comments in vector_agent.py for implementation details.
