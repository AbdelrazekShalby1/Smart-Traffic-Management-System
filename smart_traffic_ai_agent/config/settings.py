from pathlib import Path
import os

from dotenv import load_dotenv


# =========================================================
# Project Root
# =========================================================

def _find_project_root() -> Path:
    """
    Find the Smart Traffic project root from this settings file.

    The project root is the directory containing main.py, DataBase/,
    and smart_traffic_ai_agent/.
    """

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (
            (parent / "main.py").exists()
            and (parent / "DataBase").is_dir()
            and (parent / "smart_traffic_ai_agent").is_dir()
        ):
            return parent

    raise RuntimeError(
        "Could not determine project root. Expected a parent directory "
        "containing main.py, DataBase/, and smart_traffic_ai_agent/."
    )


PROJECT_ROOT = _find_project_root()
AGENT_ROOT = PROJECT_ROOT / "smart_traffic_ai_agent"

# Backwards-compatible name used by existing modules.
BASE_DIR = PROJECT_ROOT


# =========================================================
# Environment Variables
# =========================================================

ENV_FILE = PROJECT_ROOT / ".env"
LEGACY_ENV_FILE = AGENT_ROOT / ".env"

# Root .env is the canonical location. The legacy fallback keeps older
# local setups running without exposing or copying secrets.
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)
elif LEGACY_ENV_FILE.exists():
    load_dotenv(LEGACY_ENV_FILE, override=False)


def _resolve_project_path(value: str) -> str:
    """
    Resolve a path from .env relative to the project root.
    """

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return str(path.resolve())


# =========================================================
# LLM Configuration
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-120b"
)


# =========================================================
# Embedding Configuration
# =========================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"
)

EMBEDDING_DIMENSION = int(
    os.getenv(
        "EMBEDDING_DIMENSION",
        "768"
    )
)

QDRANT_DISTANCE = os.getenv(
    "QDRANT_DISTANCE",
    "Cosine"
)


# =========================================================
# Qdrant Configuration
# =========================================================

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "traffic_knowledge"
)


# =========================================================
# Database Configuration
# =========================================================

TRAFFIC_DB_PATH = os.getenv(
    "TRAFFIC_DB_PATH",
    str(PROJECT_ROOT / "DataBase" / "traffic_system.db")
)

TRAFFIC_DB_PATH = _resolve_project_path(TRAFFIC_DB_PATH)

# Alias for existing code that imports DB_PATH
DB_PATH = TRAFFIC_DB_PATH


# =========================================================
# Agent Configuration
# =========================================================

MAX_MEMORY_MESSAGES = int(
    os.getenv(
        "MAX_MEMORY_MESSAGES",
        "20"
    )
)

RAG_TOP_K = int(
    os.getenv(
        "RAG_TOP_K",
        "4"
    )
)


# =========================================================
# Validation
# =========================================================

def validate_settings(require_qdrant: bool = True):
    """
    Validate the required configuration.
    """

    missing = []

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if require_qdrant:
        if not QDRANT_URL:
            missing.append("QDRANT_URL")

        if not QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")

        if not QDRANT_COLLECTION:
            missing.append("QDRANT_COLLECTION")

        if not EMBEDDING_MODEL:
            missing.append("EMBEDDING_MODEL")

    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + f". Check project_root/.env at {ENV_FILE}"
        )

    if not Path(TRAFFIC_DB_PATH).exists():
        raise ValueError(
            "Traffic database not found. Check TRAFFIC_DB_PATH in "
            f"project_root/.env. Resolved path: {TRAFFIC_DB_PATH}"
        )


def validate_agent_settings():
    """
    Validate only the settings required to start the LLM agent.
    """

    validate_settings(require_qdrant=False)


# =========================================================
# Debug / Test
# =========================================================

if __name__ == "__main__":

    validate_settings()

    print("=" * 60)
    print("SETTINGS LOADED SUCCESSFULLY")
    print("=" * 60)

    print(f"LLM Model       : {LLM_MODEL}")
    print(f"Embedding Model : {EMBEDDING_MODEL}")
    print(f"Qdrant URL      : {QDRANT_URL}")
    print(f"Qdrant Collection: {QDRANT_COLLECTION}")
    print(f"Database        : {TRAFFIC_DB_PATH}")

    print("=" * 60)
