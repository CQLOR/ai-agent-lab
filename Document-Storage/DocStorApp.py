import os
import datetime
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

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
        
        doc = Document(
            page_content=content,
            metadata={
                'fileName': filename,
                'createdAt': datetime.datetime.now().isoformat()
            }
        )
        
        ids = vector_store.add_documents([doc])
        print(f"✅ Successfully loaded {filename} ({len(content)} characters)")
        return ids[0] if ids else None
    except Exception as e:
        print(f"❌ Error loading {filename}")
        err_msg = str(e)
        if "maximum context length" in err_msg or "token" in err_msg:
            print("⚠️ This document is too large to embed as a single chunk.")
            print("Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("Solution: The document needs to be split into smaller chunks.")
        else:
            print(f"Detail: {err_msg}")
        return None

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

    print("=== Loading Documents into Vector Database ===")
    
    # Load health brochure
    load_document(vector_store, "HealthInsuranceBrochure.md")
    
    # Load employee handbook
    load_document(vector_store, "EmployeeHandbook.md")

if __name__ == "__main__":
    main()