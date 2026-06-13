# Document Storage Application

A **foundational document storage system** built with **LangChain** and **GitHub Models API** that demonstrates the core RAG challenge: embedding documents for semantic search.

This application is a **learning project** that shows why document chunking is necessary and sets the stage for more advanced RAG systems.

## 🎯 Overview

DocStorApp loads company documents into a vector store, creating searchable embeddings. However, it reveals a critical limitation: **documents that exceed the embedding model's token limit cannot be embedded as single units**.

### What It Does

- **Document Loading**: Reads markdown files into memory
- **Vector Embeddings**: Converts full documents to vectors via text-embedding-3-small
- **Vector Storage**: Stores embeddings in an in-memory vector database
- **Metadata Tracking**: Associates documents with timestamps and filenames
- **Error Detection**: Identifies when documents are too large to process

### The Problem It Reveals

```
Document Size: 28,000 characters
                    ↓
Embedding Model Limit: 8,191 tokens
                    ↓
Result: ❌ TOKEN LIMIT EXCEEDED
```

This is why **chunking** is essential! See the [Chunking README](../Chunking/README.md) for the solution.

## 🚀 Features

### Document Management

- **Simple Loading**: Load documents from markdown files
- **Full-Document Embedding**: Attempts to embed entire documents
- **Metadata Preservation**: Tracks filename and creation timestamp
- **Error Handling**: Graceful failures with helpful error messages

### Vector Storage

- **In-Memory Vector DB**: Fast, non-persistent storage
- **Semantic Indexing**: Documents indexed by meaning, not keywords
- **Metadata Storage**: Information about each document's source

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

Create a `.env` file in the `Document-Storage/` directory:

```env
API_KEY=your_github_token_here
API_BASE_URL=https://models.inference.ai.azure.com
```

Or use the parent directory's `.env`:
```env
GITHUB_TOKEN=your_github_token_here
API_BASE_URL=https://models.inference.ai.azure.com
```

### 3. Prepare Document Files

Ensure these files exist in the `Document-Storage/` directory:
- `EmployeeHandbook.md` - Company policies and procedures
- `HealthInsuranceBrochure.md` - Benefits and coverage information

## 🖥️ Usage

### Run the Application

```bash
python3 DocStorApp.py
```

### Expected Output

```text
🤖 Python LangChain Agent Starting...

=== Loading Documents into Vector Database ===

Loading HealthInsuranceBrochure.md...
✅ Successfully loaded HealthInsuranceBrochure.md (12,450 characters)

Loading EmployeeHandbook.md...
❌ Error loading EmployeeHandbook.md
⚠️ This document is too large to embed as a single chunk.
Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.
Solution: The document needs to be split into smaller chunks.
```

### What Happened?

1. ✅ **HealthInsuranceBrochure.md** - Small enough to embed (12,450 chars ≈ 3,000 tokens)
2. ❌ **EmployeeHandbook.md** - Too large to embed (28,000 chars ≈ 7,000+ tokens)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Raw Documents                       │
│  • EmployeeHandbook.md (28,000 chars)       │
│  • HealthInsuranceBrochure.md (12,450 chars)│
└──────────────┬──────────────────────────────┘
               │
               ▼
        ┌──────────────┐
        │  Load File   │
        └──────┬───────┘
               │
               ▼
    ┌──────────────────────┐
    │  Create Document     │
    │  with metadata       │
    └──────┬───────────────┘
           │
           ▼
    ┌──────────────────────┐
    │ Count Tokens         │
    └──────┬───────────────┘
           │
           ├─ Small (< 8,191 tokens)?
           │  └──→ ✅ Embed & Store
           │
           └─ Large (> 8,191 tokens)?
              └──→ ❌ Error: Token Limit Exceeded
```

## 🔍 Key Functions

### `load_document(vector_store, file_path)`

Loads a single document into the vector store.

```python
doc_id = load_document(vector_store, "EmployeeHandbook.md")
```

**Parameters**:
- `vector_store`: The vector store instance
- `file_path`: Path to the markdown file

**Returns**: Document ID if successful, None if failed

**What It Does**:
1. Reads the entire file into memory
2. Creates a Document object with metadata
3. Attempts to embed and store the document
4. Returns success/failure with error details

**Error Handling**:
- Catches token limit exceeded errors
- Provides helpful guidance for large documents
- Returns None on failure (doesn't crash)

## 📊 Token Counting

Understanding the token limit is crucial:

| Document | Size | Est. Tokens | Result |
|----------|------|-------------|--------|
| HealthInsuranceBrochure.md | 12,450 chars | ~3,100 | ✅ Fits |
| EmployeeHandbook.md | 28,000 chars | ~7,000 | ❌ Exceeds |
| Embedding Model Limit | - | 8,191 | Max Capacity |

**Token Formula** (approximate): `characters ÷ 4 ≈ tokens`

## 🚨 The Token Limit Problem

### Why This Matters

The embedding model can only process **8,191 tokens** at once:

```
If your document has 28,000 characters:
  • ~7,000 tokens
  • Approaches the 8,191 limit
  • Any increase = failure

If your document has 50,000 characters:
  • ~12,500 tokens
  • Far exceeds the limit
  • Will always fail
```

### Solutions

1. **Chunking** (Recommended)
   - Split documents into ~1000 character pieces
   - Embed each chunk separately
   - Search across all chunks
   - See: [Chunking README](../Chunking/README.md)

2. **Summarization**
   - Summarize large documents before embedding
   - Trade detail for ability to embed

3. **Manual Selection**
   - Manually select relevant sections to embed
   - Not scalable for large document collections

4. **Different Embedding Model**
   - Use models with higher token limits
   - Trade cost/latency for capacity

## 📚 Learning Path

This application is **Lab 3: Semantic Search** in the RAG fundamentals course.

### Course Progression

```
Lab 1: Embeddings
  └─ Understand vectors and embeddings
     └─ Lab 2: Vector Storage
        └─ Store and retrieve embeddings
           └─ Lab 3: Semantic Search (THIS LAB)
              └─ 🚨 Discover the token limit problem
                 └─ Lab 4: Chunking Strategies
                    └─ Solve the problem with chunking
                       └─ Lab 5: RAG with LLM
                          └─ Complete end-to-end RAG
```

## 🔧 Extending the Application

### Add More Documents

```python
# In main(), after existing loads:
load_document(vector_store, "CompensationPolicy.md")
load_document(vector_store, "TrainingProgram.md")
```

### Use Persistent Storage

```python
from langchain_community.vectorstores import Chroma

# Replace InMemoryVectorStore with:
vector_store = Chroma(embedding_function=embeddings)
```

### Add Semantic Search

```python
# After loading documents:
query = "What are health insurance benefits?"
results = vector_store.similarity_search(query, k=3)

for result in results:
    print(f"Document: {result.metadata['fileName']}")
    print(f"Content: {result.page_content[:200]}...")
```

### Add Search Capability

Modify to add a search loop:

```python
print("\n=== Semantic Search ===")
while True:
    query = input("Search query (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    
    results = vector_store.similarity_search(query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata['fileName']}")
        print(f"   {doc.page_content[:150]}...")
```

## ⚠️ Common Issues

### Issue: Document Won't Load

**Error**: `Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.`

**Cause**: Document is too large (>25,000 characters)

**Solution**: 
- Use the chunking approach (see [Chunking README](../Chunking/README.md))
- Or split the document manually before loading

---

### Issue: API Key Not Found

**Error**: `API_KEY or GITHUB_TOKEN not found in environment variables.`

**Solution**:
1. Create `.env` file in this directory or parent
2. Add your GitHub token: `GITHUB_TOKEN=ghp_xxxxx`
3. Ensure the file is readable

---

### Issue: Slow Embedding

**Cause**: First embedding takes longer due to model initialization

**Expected**: Subsequent calls are faster

---

## 📈 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Load small file | ~100ms | File I/O |
| Embed small doc | ~1-2s | First call slower |
| Embed large doc | ❌ | Fails with token error |
| Vector lookup | <1ms | In-memory |

## 🎓 Why This Matters

DocStorApp demonstrates a **real-world constraint** in AI systems:

- **LLMs and Embeddings** have token limits
- **Real documents** often exceed these limits
- **Naive approaches** fail (trying to embed the whole document)
- **Chunking is essential** for scalable RAG systems
- **Solution design** requires understanding constraints

## 🚀 Next Steps

After understanding this application's limitations:

1. **Learn Chunking**: See [Chunking README](../Chunking/README.md)
2. **Implement Solutions**: Use chunking with `chunking-agnt.py`
3. **Add Search**: Implement semantic search over chunks
4. **Build Agents**: Create AI agents that reason over retrieved context
5. **Deploy**: Make your RAG system production-ready

## 📖 Documentation Links

- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Text Embeddings](https://python.langchain.com/docs/integrations/text_embedding/)
- [GitHub Models API](https://docs.github.com/en/github-models/prototyping-with-ai-models)
- [Token Counting](https://platform.openai.com/tokenizer)

## 💡 Key Takeaways

✅ **Full document embedding** works for small documents (<8,191 tokens)

⚠️ **Large documents** exceed embedding model limits

❌ **Naive approaches fail** when documents are too large

✨ **Chunking is the solution** for handling arbitrary document sizes

🚀 **This challenge drives the design** of production RAG systems

---

**Ready to solve the chunking problem?** Check out [../Chunking/README.md](../Chunking/README.md)!

**Questions?** Review the inline code comments in DocStorApp.py for implementation details.
