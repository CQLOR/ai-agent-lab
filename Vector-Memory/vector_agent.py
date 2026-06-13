import os
import datetime
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables. First look for .env in the parent directory (root folder),
# then the current directory.
if not os.path.exists(".env") and os.path.exists("../.env"):
    load_dotenv("../.env")
else:
    load_dotenv()

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

    # Expanded Dataset (Step 7)
    sentences = [
        "The canine barked loudly.",
        "The dog made a noise.",
        "The electron spins rapidly.",
        "I love eating pizza with extra cheese.",
        "The basketball player scored a three-pointer.",
        "Rain is forecasted for tomorrow afternoon.",
        "Python is a popular programming language.",
        "The kitten purred softly on the couch.",
        "Quantum mechanics explains particle behavior.",
        "Homemade pasta tastes better than store-bought.",
        "The soccer match ended in a tie.",
        "Clouds are forming over the mountains.",
        "JavaScript runs in web browsers.",
        "Puppies need lots of attention and exercise.",
        "Atoms are made of protons, neutrons, and electrons."
    ]

    print(f"Storing {len(sentences)} sentences in the vector database...")
    
    # Create metadata (Step 3)
    metadatas = [
        {
            "created_at": datetime.datetime.now().isoformat(),
            "index": idx
        }
        for idx, _ in enumerate(sentences)
    ]

    # Store sentences in the vector database
    vector_store.add_texts(texts=sentences, metadatas=metadatas)
    print(f"✅ Successfully stored {len(sentences)} sentences\n")

    # Define the search function (Step 4)
    def search_sentences(vector_store, query, k=3):
        results = vector_store.similarity_search_with_score(query, k=k)
        print(f"\n🔍 Search Results for \"{query}\":")
        for rank, (doc, score) in enumerate(results, 1):
            print(f"{rank}. [Score: {score:.4f}] {doc.page_content}")

    # Interactive search CLI loop (Step 5)
    print("=== Semantic Search ===")
    while True:
        try:
            query = input("Enter a search query (or 'quit' to exit): ").strip()
            if query.lower() in ["quit", "exit"]:
                break
            if not query:
                continue
            search_sentences(vector_store, query)
            print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during search: {e}")

    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()