import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables. First look for .env in the parent directory (root folder),
# then the current directory.
if not os.path.exists(".env") and os.path.exists("../.env"):
    load_dotenv("../.env")
else:
    load_dotenv()

def load_document(vector_store, file_path):
    filename = os.path.basename(file_path)
    print(f"\nLoading {filename}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(content)
        
        # Create Document objects for each chunk
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    'fileName': filename,
                    'createdAt': datetime.now().isoformat(),
                    'chunkIndex': idx
                }
            )
            for idx, chunk in enumerate(chunks)
        ]
        
        ids = vector_store.add_documents(documents)
        print(f"✅ Successfully loaded {filename} as {len(chunks)} chunks")
        return ids
    except Exception as e:
        print(f"❌ Error loading {filename}")
        err_msg = str(e)
        if "maximum context length" in err_msg or "token" in err_msg:
            print("⚠️ This document is too large to embed as a single chunk.")
            print("Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("Solution: The document has been split into smaller chunks.")
        else:
            print(f"Detail: {err_msg}")
        return None

def load_document_with_chunks(vector_store, file_path, chunks):
    """
    Load pre-chunked documents into the vector store with metadata.
    
    Args:
        vector_store: The vector store to add documents to
        file_path: Path to the original document file
        chunks: List of LangChain Document objects (chunks)
    
    Returns:
        Total number of chunks stored
    """
    filename = os.path.basename(file_path)
    total_chunks = len(chunks)
    stored_count = 0
    
    try:
        for idx, chunk in enumerate(chunks, 1):
            try:
                # Update chunk metadata
                chunk.metadata['fileName'] = f"{filename} (Chunk {idx}/{total_chunks})"
                chunk.metadata['createdAt'] = datetime.now().isoformat()
                chunk.metadata['chunkIndex'] = idx - 1
                
                # Add chunk to vector store
                vector_store.add_documents([chunk])
                stored_count += 1
                
                # Print progress
                print(f"  Processing chunk {idx}/{total_chunks}...")
                
            except Exception as chunk_error:
                print(f"  ❌ Error processing chunk {idx}/{total_chunks}: {str(chunk_error)}")
                continue
        
        print(f"✅ Successfully loaded {stored_count}/{total_chunks} chunks from {filename}")
        return stored_count
        
    except Exception as e:
        print(f"❌ Error loading chunks from {filename}: {str(e)}")
        return 0

def load_with_fixed_size_chunking(vector_store, file_path):
    """
    Load a document using fixed-size character chunking.
    
    Args:
        vector_store: The vector store to add documents to
        file_path: Path to the document file
    
    Returns:
        Total number of chunks stored
    """
    filename = os.path.basename(file_path)
    print(f"\n📄 Loading {filename} with fixed-size chunking...")
    
    try:
        # Read the document
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split using fixed-size character chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separator=" "
        )
        chunks = text_splitter.split_text(content)
        
        # Create Document objects
        documents = text_splitter.create_documents([content])
        
        # Load chunks into vector store
        stored_count = load_document_with_chunks(vector_store, file_path, documents)
        
        # Print statistics
        if chunks:
            avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            min_size = min(len(chunk) for chunk in chunks)
            max_size = max(len(chunk) for chunk in chunks)
            print(f"\n📊 Fixed-Size Chunking Statistics:")
            print(f"  Total chunks created: {len(chunks)}")
            print(f"  Average chunk size: {avg_size:.0f} characters")
            print(f"  Min chunk size: {min_size} characters")
            print(f"  Max chunk size: {max_size} characters")
        
        return stored_count
        
    except Exception as e:
        print(f"❌ Error loading {filename} with fixed-size chunking: {str(e)}")
        return 0

def create_search_tool(vector_store):
    """
    Create a LangChain tool for searching documents in the vector store.
    
    Args:
        vector_store: The vector store to search
    
    Returns:
        A LangChain Tool that can be used by agents
    """
    @tool
    def search_documents(query: str) -> str:
        """Searches the company document repository for relevant information based on the given query. Use this to find information about company policies, benefits, and procedures."""
        try:
            # Search vector store for top 3 results
            results = vector_store.similarity_search_with_score(query, k=3)
            
            if not results:
                return "No results found for the query."
            
            # Format results
            formatted_results = []
            for idx, (doc, score) in enumerate(results, 1):
                formatted_results.append(f"Result {idx} (Score: {score:.4f}): {doc.page_content}")
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    return search_documents

def main():
    print("🤖 Python LangChain Agent Starting...\n")
    
    # API key check
    api_key = os.getenv("API_KEY") or os.getenv("GITHUB_TOKEN")
    if not api_key:
        print("❌ Error: API_KEY or GITHUB_TOKEN not found in environment variables.")
        print("Please check your .env file.")
        return  
    # Initialize Embedding Model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url=os.getenv("API_BASE_URL") or "https://models.inference.ai.azure.com",
        api_key=api_key,
        check_embedding_ctx_length=False
    )
    # Initialize Vector Store
    vector_store = InMemoryVectorStore(embeddings)
    
    # Initialize Chat Model
    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url=os.getenv("API_BASE_URL") or "https://models.inference.ai.azure.com",
        api_key=api_key
    )
    
    # Create search tool for the agent
    search_tool = create_search_tool(vector_store)
    
    # Create the agent using the modern LangChain API
    # The new create_agent returns a compiled state graph
    system_prompt = """You are a helpful assistant that answers questions about company policies, benefits, and procedures. 
Use the search_documents tool to find relevant information before answering. 
Always cite which document chunks you used in your answer."""
    
    agent_graph = create_agent(
        model=chat_model,
        tools=[search_tool],
        system_prompt=system_prompt
    )

    print("=== Loading Documents into Vector Database ===")
    
    # Load health brochure
    load_document(vector_store, "HealthInsuranceBrochure.md")
    
    # Load employee handbook with fixed-size chunking
    load_with_fixed_size_chunking(vector_store, "EmployeeHandbook.md")
    
    # Initialize chat history
    chat_history = []
    
    # Print welcome message
    print("\n=== Agent Chat Interface ===")
    print("🤖 Welcome! I'm your company assistant.")
    print("I can help you find information about:")
    print("  • Company policies")
    print("  • Benefits and procedures")
    print("  • Health insurance coverage")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("\n" + "="*50 + "\n")
    
    # Interactive chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit"]:
                print("\n👋 Thank you for using the company assistant. Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Invoke the agent with the new message format
            # The modern create_agent expects {"messages": [...]}
            result = agent_graph.invoke({
                "messages": chat_history + [HumanMessage(content=user_input)]
            })
            
            # Extract the agent response from the messages
            # The result contains all messages, we want the last one
            response_messages = result.get("messages", [])
            if response_messages:
                agent_response = response_messages[-1].content
            else:
                agent_response = "No response generated"
            
            print(f"\nAgent: {agent_response}\n")
            
            # Add to chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=agent_response))
            
        except KeyboardInterrupt:
            print("\n\n👋 Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()