# Chunking Agent - Document RAG System

A **Retrieval-Augmented Generation (RAG)** agent built with **LangChain** and **GitHub Models API** that intelligently chunks large documents, stores them in a vector database, and answers questions by retrieving relevant information from company documents.

## 🎯 Overview

This agent solves a critical problem in RAG systems: **handling large documents that exceed embedding model token limits**. It implements multiple document chunking strategies and uses semantic search to find relevant information before generating responses.

### What It Does

- **Document Chunking**: Splits large documents (EmployeeHandbook.md, HealthInsuranceBrochure.md) into manageable chunks
- **Vector Storage**: Embeds and stores chunks in an in-memory vector database
- **Semantic Search**: Finds the top 3 most relevant document chunks for any query
- **RAG-Enhanced Responses**: Generates natural language answers based on retrieved document context
- **Conversation Memory**: Maintains chat history for multi-turn conversations

## 🚀 Features

### Document Processing

- **Fixed-Size Character Chunking**: Splits documents into 1000-character chunks with space boundaries
- **Metadata Tracking**: Each chunk includes filename, timestamp, and chunk index
- **Automatic Batching**: Processes chunks incrementally for large documents
- **Error Handling**: Gracefully handles failed chunk processing

### RAG Agent Capabilities

- **Smart Tool Use**: Automatically searches documents when needed to answer questions
- **Source Citation**: Identifies which document chunks were used in responses
- **Semantic Matching**: Uses vector embeddings to find contextually relevant content
- **Confidence Scoring**: Returns similarity scores for retrieved results

### Interactive Chat Interface

- **Multi-turn Conversations**: Maintains chat history across multiple questions
- **Verbose Mode**: Shows agent reasoning and tool invocation
- **Graceful Exit**: Supports "quit" or "exit" commands
- **Error Recovery**: Handles errors without stopping the conversation

## 📋 Prerequisites

- Python 3.8+
- GitHub personal access token (with access to GitHub Models)
- 4MB+ of disk space for dependencies

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install langchain-openai langchain langchain-text-splitters python-dotenv
```

### 2. Configure Environment Variables

Create a `.env` file in the `Chunking/` directory:

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

Ensure these files exist in the `Chunking/` directory:
- `EmployeeHandbook.md` - Company policies and procedures
- `HealthInsuranceBrochure.md` - Benefits and coverage information

## 🖥️ Usage

### Run the Agent

```bash
python3 chunking-agnt.py
```

### Example Conversation

```text
🤖 Welcome! I'm your company assistant.
I can help you find information about:
  • Company policies
  • Benefits and procedures
  • Health insurance coverage

Type 'quit' or 'exit' to end the conversation.

==================================================

You: What are the main health insurance benefits?

[Agent searches documents and responds with relevant information from chunks]

Agent: Based on the company documentation, the main health insurance benefits include... 
[Response with sources cited]

You: How do I enroll?

Agent: To enroll in health insurance benefits, you need to... 
[Uses conversation history to maintain context]

You: quit

👋 Thank you for using the company assistant. Goodbye!
```

## 🏗️ Architecture

### Components

```
┌─────────────────────────────────────────────┐
│         Raw Documents                       │
│  • EmployeeHandbook.md                      │
│  • HealthInsuranceBrochure.md               │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│    Document Chunking (CharacterTextSplitter)│
│  • Chunk Size: 1000 characters              │
│  • Strategy: Split on spaces                │
│  • Preserve word boundaries                 │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│    Embeddings (text-embedding-3-small)      │
│  • Convert chunks to vectors                │
│  • Embed metadata                           │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│    Vector Store (InMemoryVectorStore)       │
│  • Index all chunk embeddings               │
│  • Store metadata for retrieval             │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│    Search Tool (search_documents)           │
│  • Similarity search: k=3 results           │
│  • Return chunks + scores                   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│    Chat Model (gpt-4o via GitHub Models)    │
│  • Reason about retrieved context           │
│  • Generate natural language responses      │
│  • Cite sources                             │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│    Agent (create_agent with tool calling)   │
│  • Orchestrates search → reason → respond   │
│  • Maintains conversation history           │
│  • Handles multi-turn interactions          │
└─────────────────────────────────────────────┘
```

### Data Flow

1. **Documents Loaded** → Chunked into 1000-char pieces
2. **Chunks Embedded** → Converted to vectors via text-embedding-3-small
3. **Vectors Stored** → Indexed in InMemoryVectorStore with metadata
4. **User Query** → Input to agent
5. **Tool Invoked** → search_documents(query) called
6. **Top 3 Results** → Retrieved with similarity scores
7. **Context Used** → Passed to chat model for reasoning
8. **Response Generated** → Natural language answer with sources
9. **History Updated** → Message added to chat_history for context

## 📊 Chunking Statistics

When you run the agent, you'll see statistics like:

```
📄 Loading EmployeeHandbook.md with fixed-size chunking...
  Processing chunk 1/28...
  Processing chunk 2/28...
  ...

📊 Fixed-Size Chunking Statistics:
  Total chunks created: 28
  Average chunk size: 1,247 characters
  Min chunk size: 456 characters
  Max chunk size: 1,000 characters
```

### Why These Numbers Matter

- **28 chunks**: The 28,000 character document becomes manageable 1KB pieces
- **1,247 avg**: Good balance between context and precision
- **1,000 max**: Respects the embedding model's requirements
- **456 min**: Even small chunks are retained (e.g., short sections)

## 🔍 Key Functions

### `load_document_with_chunks(vector_store, file_path, chunks)`

Loads pre-chunked documents into the vector store with metadata tracking.

```python
stored_count = load_document_with_chunks(
    vector_store, 
    "EmployeeHandbook.md",
    document_chunks
)
```

**Returns**: Total number of chunks successfully stored

---

### `load_with_fixed_size_chunking(vector_store, file_path)`

Implements fixed-size character-based document chunking strategy.

```python
chunk_count = load_with_fixed_size_chunking(
    vector_store,
    "EmployeeHandbook.md"
)
```

**Features**:
- Chunk size: 1000 characters
- No overlap between chunks
- Splits on space boundaries to preserve words
- Prints processing statistics

---

### `create_search_tool(vector_store)`

Creates a LangChain tool that the agent can automatically invoke for document search.

```python
search_tool = create_search_tool(vector_store)
```

**Returns**: A @tool decorated function that:
- Searches the vector store with similarity scoring
- Returns top 3 results with scores
- Formats results for LLM consumption

## 🎓 Understanding the RAG Pattern

### Why Chunking?

Large documents exceed embedding model limits (8,191 tokens). Chunking solves this by:

1. **Breaking documents into ~1KB pieces** → Each fits within token limits
2. **Embedding each chunk separately** → All content becomes searchable
3. **Storing chunks with metadata** → Know where information comes from
4. **Searching semantically** → Find relevant content, not exact matches
5. **Using context for generation** → LLM generates better answers with relevant chunks

### Example Flow

```
User: "What's the vacation policy?"
       ↓
Agent searches: "vacation policy"
       ↓
Vector DB returns:
  Result 1 (Score: 0.8923): "Annual Vacation: Employees receive..."
  Result 2 (Score: 0.7452): "Time off policies are managed by..."
  Result 3 (Score: 0.6234): "Holidays and vacation differ as..."
       ↓
Agent reasons: "These chunks clearly answer the question"
       ↓
Agent responds: "Based on the Employee Handbook, vacation 
policy allows employees to receive..."
```

## 🚨 Troubleshooting

### Issue: `ImportError: cannot import name 'create_react_agent'`

**Solution**: You're likely using LangChain 1.3.4+. The code already uses the modern `create_agent` API. Update your installation:

```bash
pip install --upgrade langchain
```

### Issue: Vector store is empty / No results returned

**Solution**: Verify documents exist:
```bash
ls -la EmployeeHandbook.md HealthInsuranceBrochure.md
```

### Issue: Agent responses are generic/not citing sources

**Solution**: Check that verbose mode is enabled in `AgentExecutor` to see what's being searched:

```python
agent_graph = create_agent(
    model=chat_model,
    tools=[search_tool],
    system_prompt=system_prompt,
    debug=True  # Enable debug output
)
```

## 📈 Performance Considerations

| Metric | Value | Impact |
|--------|-------|--------|
| Chunk Size | 1000 chars | Balances context vs precision |
| Top-K Results | 3 chunks | Provides context without noise |
| Vector DB | In-Memory | Fast but non-persistent |
| Model | gpt-4o | High quality, fast reasoning |
| Temperature | 0 | Consistent, factual responses |

## 🔧 Extending the System

### Add More Documents

```python
# In main(), add after loading existing documents:
load_with_fixed_size_chunking(vector_store, "CompensationPolicy.md")
load_with_fixed_size_chunking(vector_store, "PerformanceReview.md")
```

### Try Different Chunking Strategies

```python
# Paragraph-based chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
)

# Markdown-aware chunking
from langchain_text_splitters import MarkdownHeaderTextSplitter
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
)
```

### Use Persistent Vector Store

Replace `InMemoryVectorStore` with a persistent option:

```python
from langchain_community.vectorstores import Chroma
vector_store = Chroma(embedding_function=embeddings)
```

## 📚 Related Lab Exercises

This implementation completes **Lab 4: Implementing Chunking Strategies with LangChain** from the RAG fundamentals course. Additional labs:

- **Lab 1**: Embeddings and Vector Representations
- **Lab 2**: Vector Storage and Retrieval
- **Lab 3**: Semantic Search
- **Lab 4**: Chunking Strategies (THIS PROJECT)
- **Lab 5**: RAG with Language Models (THIS PROJECT, Step 6)

## 📖 Documentation Links

- [LangChain Agents](https://python.langchain.com/docs/how_to/#agents)
- [Text Splitters](https://python.langchain.com/docs/how_to/#text-splitters)
- [LangChain Tools](https://python.langchain.com/docs/how_to/#tools)
- [GitHub Models API](https://docs.github.com/en/github-models/prototyping-with-ai-models)

## 📝 License

MIT License - See parent repository for details

## 🤝 Contributing

Improvements and extensions welcome! Consider:

- Additional chunking strategies (hierarchical, semantic-aware)
- Persistent vector store integration
- Multi-document comparison
- Document source tracking in responses
- Performance benchmarking

---

**Happy RAGing! 🚀** Questions? Check the inline code comments for implementation details.
