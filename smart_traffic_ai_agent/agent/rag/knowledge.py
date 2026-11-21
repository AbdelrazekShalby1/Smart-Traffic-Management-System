from pathlib import Path
from typing import List

from langchain_core.documents import Document


# ============================================================
# Knowledge Base Directory
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DOCS_DIR = BASE_DIR / "docs"


# ============================================================
# Supported File Types
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
}


# ============================================================
# Get Knowledge Base Files
# ============================================================

def get_knowledge_files() -> List[Path]:
    """
    Return all supported knowledge-base files inside docs/.

    Supported formats:
        - .txt
        - .md
    """

    if not DOCS_DIR.exists():
        return []

    files = []

    for path in DOCS_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(path)

    return sorted(files)


# ============================================================
# Load One Document
# ============================================================

def load_document(
    file_path: Path,
) -> Document:
    """
    Load a single knowledge-base file and convert it into
    a LangChain Document.
    """

    try:

        text = file_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        text = file_path.read_text(
            encoding="utf-8-sig"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    relative_path = file_path.relative_to(
        DOCS_DIR
    )

    metadata = {
        "source": str(relative_path),
        "file_name": file_path.name,
        "file_type": file_path.suffix.lower(),
    }

    return Document(
        page_content=text,
        metadata=metadata,
    )


# ============================================================
# Load Knowledge Base
# ============================================================

def load_knowledge_base() -> List[Document]:
    """
    Load all supported documents from the knowledge base.
    """

    files = get_knowledge_files()

    documents = []

    for file_path in files:

        try:

            document = load_document(
                file_path
            )

            # Ignore completely empty files
            if not document.page_content.strip():
                continue

            documents.append(
                document
            )

        except Exception as e:

            print(
                f"[WARNING] Could not load "
                f"{file_path}: {e}"
            )

    return documents


# ============================================================
# Get Knowledge Base Statistics
# ============================================================

def get_knowledge_base_info():
    """
    Return basic information about the knowledge base.
    """

    files = get_knowledge_files()

    documents = load_knowledge_base()

    return {
        "directory": str(DOCS_DIR),
        "file_count": len(files),
        "document_count": len(documents),
        "files": [
            str(
                file.relative_to(DOCS_DIR)
            )
            for file in files
        ],
    }