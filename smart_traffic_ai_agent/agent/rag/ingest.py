from pathlib import Path
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

AGENT_ROOT = Path(__file__).resolve().parents[2]

if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agent.rag.knowledge import load_knowledge_base
from agent.rag.retriever import (
    get_qdrant_client,
    get_embeddings,
    verify_qdrant_collection,
)
from config.settings import (
    QDRANT_COLLECTION,
    EMBEDDING_DIMENSION,
    QDRANT_DISTANCE,
)


# =========================================================
# Chunk Configuration
# =========================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# =========================================================
# Document Splitting
# =========================================================

def split_documents(documents):
    """
    Split documents into smaller chunks
    suitable for embedding and retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)


# =========================================================
# Qdrant Collection Check
# =========================================================

def check_collection(client):
    """
    Check whether the configured Qdrant collection exists and is
    compatible with the embedding configuration.
    """

    verify_qdrant_collection(client)
    return True


# =========================================================
# Knowledge Base Ingestion
# =========================================================

def ingest_knowledge_base():
    """
    Load documents, split them into chunks,
    create embeddings, and store them in Qdrant.
    """

    print("=" * 60)
    print("STARTING KNOWLEDGE BASE INGESTION")
    print("=" * 60)

    # -----------------------------------------------------
    # Load documents
    # -----------------------------------------------------

    print("\n[1] Loading knowledge base...")

    documents = load_knowledge_base()

    if not documents:
        print("[ERROR] No knowledge documents found.")
        return

    print(
        f"[OK] Loaded {len(documents)} documents."
    )

    # -----------------------------------------------------
    # Split documents
    # -----------------------------------------------------

    print("\n[2] Splitting documents...")

    chunks = split_documents(documents)

    if not chunks:
        print("[ERROR] No chunks were created.")
        return

    print(
        f"[OK] Created {len(chunks)} chunks."
    )

    # -----------------------------------------------------
    # Connect to Qdrant
    # -----------------------------------------------------

    print("\n[3] Connecting to Qdrant...")

    client = get_qdrant_client()

    print(
        f"[INFO] Collection: {QDRANT_COLLECTION}"
    )

    # -----------------------------------------------------
    # Check collection
    # -----------------------------------------------------

    try:
        check_collection(client)
    except RuntimeError as exc:

        print(
            f"\n[ERROR] {exc}"
        )

        print(
            "\nCreate or update the collection in Qdrant with:"
        )

        print(
            f"Dimension: {EMBEDDING_DIMENSION}"
        )

        print(
            f"Distance: {QDRANT_DISTANCE}"
        )

        return

    print(
        f"[OK] Collection '{QDRANT_COLLECTION}' found."
    )

    # -----------------------------------------------------
    # Create embeddings
    # -----------------------------------------------------

    print("\n[4] Loading embedding model...")

    embeddings = get_embeddings()

    print(
        "[OK] Embedding model loaded."
    )

    # -----------------------------------------------------
    # Create vector store
    # -----------------------------------------------------

    print("\n[5] Connecting to vector store...")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    # -----------------------------------------------------
    # Add documents
    # -----------------------------------------------------

    print("\n[6] Adding documents to Qdrant...")

    vector_store.add_documents(chunks)

    print(
        f"[OK] Added {len(chunks)} chunks."
    )

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE INGESTION COMPLETED")
    print("=" * 60)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    ingest_knowledge_base()
