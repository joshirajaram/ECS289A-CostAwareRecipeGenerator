import argparse
import pandas as pd
import time
import ollama
import chromadb
from chromadb.utils import embedding_functions

parser = argparse.ArgumentParser(description="Generate embeddings and store in Chroma DB")
parser.add_argument("--max-rows", type=int, default=None, help="If set, only process the first N rows (useful for testing)")
args = parser.parse_args()

df = pd.read_csv("../recipes_data.csv", nrows=args.max_rows)  # Make sure your CSV has 'ingredients', 'instructions', 'total_cost', 'nutrition_profile'
df["text"] = df["title"].astype(str) + " " + df["NER"].astype(str) + " " + df["directions"].astype(str)

# Add cost and nutrition metadata here if needed
# df["metadata"] = df.apply(lambda row: {
#     "cost": row["total_cost"],
#     "nutrition": row["nutrition_profile"]
# }, axis=1)

# client = chromadb.PersistentClient(chromadb.config.Settings(
#     persist_directory="~/chroma_db",  # local storage
#     anonymized_telemetry=False
# ))
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("recipes")

# Prefer the built-in Chroma Ollama wrapper (matches installed chromadb API).
# This implementation expects an Ollama server at localhost:11434 by default.
ef = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")

batch_size = 50  # adjust depending on RAM
start_time = time.time()
for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch_texts = df["text"].iloc[start:end].tolist()
    batch_ids = [str(i) for i in range(start, end)]
    # batch_metadata = df["metadata"].iloc[start:end].tolist()

    # Compute embeddings using the Chroma-provided Ollama wrapper
    try:
        batch_embeddings = ef(batch_texts)
    except Exception as e:
        print(f"Error computing embeddings for batch {start}-{end}: {e}")
        raise

    collection.add(
        documents=batch_texts,
        ids=batch_ids,
        embeddings=batch_embeddings,
        # metadatas=batch_metadata,
    )

    print(f"Processed recipes {start} to {end}")
    time.sleep(0.2)  # prevent overloading local server
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")

query_text = "Spaghetti with tomato sauce and basil"
query_embedding = ef([query_text])
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
)

print("\nTop 5 similar recipes:")
for doc in results["documents"][0]:
    print(f"\nRecipe: {doc}")