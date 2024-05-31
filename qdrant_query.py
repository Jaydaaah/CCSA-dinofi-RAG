from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

Client = QdrantClient(":memory:")

# Prepare your documents, metadata, and IDs

with open("school data.md", "r") as textfile:
    docs = [text.strip() for text in textfile.read().split("\n\n") if text != ""]
    ids = [x for x in range(1, len(docs) + 1)]

# If you want to change the model:
# client.set_model("sentence-transformers/all-MiniLM-L6-v2")
# List of supported models: https://qdrant.github.io/fastembed/examples/Supported_Models

# Use the new add() instead of upsert()
# This internally calls embed() of the configured embedding model
Client.create_collection(
    collection_name="ChatBot_Database",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)

Client.upload_points(
    collection_name="ChatBot_Database",
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(doc).tolist(), payload={"description": doc}
        )
        for idx, doc in enumerate(docs)
    ]
)

def Query(query_str: str, limit = 5):
    return Client.search(
        collection_name="ChatBot_Database",
        query_vector=encoder.encode(query_str).tolist(),
        limit=limit
    )

if __name__ == "__main__":
    search_result = Client.query(
        collection_name="ChatBot_Database",
        query_text="Who created you?",
        limit=1
    )
    print(search_result[0].document)