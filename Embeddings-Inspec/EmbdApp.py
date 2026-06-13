import os
import math
from dotenv import load_dotenv

# Load environment variables. First look for .env in the parent directory (root folder),
# then the current directory.
if not os.path.exists(".env") and os.path.exists("../.env"):
    load_dotenv("../.env")
else:
    load_dotenv()

from langchain_openai import OpenAIEmbeddings

def cosine_similarity(v1, v2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def run_inspector(embeddings, sentences):
    print("\nGenerating embeddings for three sentences...")
    try:
        vectors = []
        for i, sentence in enumerate(sentences, 1):
            vector = embeddings.embed_query(sentence)
            vectors.append(vector)
            print(f"Sentence {i}: \"{sentence}\"")

        print("\n=== Cosine Similarities ===")
        sim1_2 = cosine_similarity(vectors[0], vectors[1])
        sim2_3 = cosine_similarity(vectors[1], vectors[2])
        sim3_1 = cosine_similarity(vectors[2], vectors[0])

        print(f"Cosine similarity between Sentence 1 and Sentence 2: {sim1_2:.4f}")
        print(f"Cosine similarity between Sentence 2 and Sentence 3: {sim2_3:.4f}")
        print(f"Cosine similarity between Sentence 3 and Sentence 1: {sim3_1:.4f}")
    except Exception as e:
        print(f"❌ Error generating embeddings: {str(e)}")

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

    experiments = {
        "1": ("Standard Lab Sentences", [
            "The canine barked loudly.",
            "The dog made a noise.",
            "The electron spins rapidly."
        ]),
        "2": ("Experiment 1: Same Sentence in Different Languages", [
            "The dog barked loudly.",
            "El perro ladró fuerte.",
            "Le chien a aboyé fort."
        ]),
        "3": ("Experiment 2: Change Just One Word", [
            "The cat sat on the mat.",
            "The dog sat on the mat.",
            "The cat sat on the hat."
        ]),
        "4": ("Experiment 3: Add Adjectives and Details", [
            "I like pizza.",
            "I really enjoy eating delicious, hot pizza.",
            "Pizza is good."
        ]),
        "5": ("Experiment 4: Opposite Meanings", [
            "The movie was excellent and entertaining.",
            "The movie was terrible and boring.",
            "I enjoyed watching the film."
        ]),
        "6": ("Experiment 5: Technical vs. Casual Language", [
            "The precipitation is forecasted to commence shortly.",
            "It's going to rain soon.",
            "The weather forecast indicates imminent rainfall."
        ]),
        "7": ("Experiment 6: Different Topics", [
            "I enjoy programming in Python.",
            "Chocolate cake is delicious.",
            "The ocean is vast and deep."
        ])
    }

    while True:
        print("\n==============================================")
        print("          EMBEDDING INSPECTOR MENU            ")
        print("==============================================")
        print("1. Run Standard Lab Sentences")
        print("2. Run Experiment 1: Same Sentence in Different Languages")
        print("3. Run Experiment 2: Change Just One Word")
        print("4. Run Experiment 3: Add Adjectives and Details")
        print("5. Run Experiment 4: Opposite Meanings")
        print("6. Run Experiment 5: Technical vs. Casual Language")
        print("7. Run Experiment 6: Different Topics")
        print("8. Enter Custom Sentences")
        print("9. Exit")
        print("==============================================")
        
        choice = input("Select an option (1-9): ").strip()
        
        if choice in experiments:
            name, sentences = experiments[choice]
            print(f"\n--- Running: {name} ---")
            run_inspector(embeddings, sentences)
        elif choice == "8":
            print("\nEnter 3 custom sentences:")
            s1 = input("Sentence 1: ").strip()
            s2 = input("Sentence 2: ").strip()
            s3 = input("Sentence 3: ").strip()
            if not s1 or not s2 or not s3:
                print("❌ Error: All 3 sentences are required.")
                continue
            print(f"\n--- Running: Custom Sentences ---")
            run_inspector(embeddings, [s1, s2, s3])
        elif choice == "9" or choice.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please select 1-9.")

if __name__ == "__main__":
    main()
