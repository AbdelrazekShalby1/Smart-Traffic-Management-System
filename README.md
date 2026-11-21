# Smart Traffic Management System

A Python computer vision prototype for traffic monitoring, paired with a
LangGraph AI assistant, retrieval augmented generation, a Streamlit dashboard,
and a separate React dashboard frontend.

## Capabilities

- YOLO vehicle and emergency vehicle detection, with ByteTrack tracking.
- Vehicle counting, region based traffic density, and line based speed estimation.
- Red light violation detection and license plate detection with EasyOCR.
- SQLite violation records and evidence image generation.
- Groq powered assistant with traffic tools, conversation memory, and Qdrant retrieval.
- React / TypeScript / Vite frontend for dashboard, monitoring, violations, and chat.

Speed and violation outputs depend on camera calibration, configured regions,
frame rate, and model quality. No production accuracy claim is made here.

## Architecture

```text
Video -> YOLO -> ByteTrack -> Vehicle state -> Traffic analysis -> SQLite / evidence
                                 |                                |
                                 +-> LangGraph tools              +-> Streamlit dashboard
                                          |
                                  Groq + Qdrant knowledge retrieval

React frontend -> expected HTTP API (adapter not included)
```

## Quick start (Windows PowerShell)

Python 3.11 or 3.12 is a sensible starting point; the dependency set is not locked.
Use a virtual environment and compatible PyTorch packages for your machine.

```powershell
git clone https://github.com/AbdelrazekShalby1/Smart-Traffic-Management-System.git
cd Smart-Traffic-Management-System
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`. For knowledge retrieval, configure your Qdrant URL,
API key if required, and collection. Never commit the populated `.env` file.
The included `Models/` directory contains the supplied detection weights.
Place your video in `Utils/` or update `VIDEO_PATH` in `main.py`; calibrate the
speed lines, distance, stop line, and regions for that camera.

Run commands from the repository root:

```powershell
python main.py
```

The OpenCV window displays the video while a terminal thread accepts assistant
questions. Type `exit` to stop the chat. The pipeline creates the local violations
database. No private records or source videos are included in this repository.

For the database dashboard, use a second terminal:

```powershell
python -m streamlit run Dashboard/dashboard.py
```

## Knowledge retrieval

Add traffic guidance as `.md` or `.txt` files under `smart_traffic_ai_agent/docs/`.
With Qdrant and the environment configured:

```powershell
python smart_traffic_ai_agent/agent/rag/ingest.py
```

The default embedding model is `sentence-transformers/all-mpnet-base-v2`, using
768 dimensions. The Qdrant collection must match the embedding model.

## React frontend

```powershell
cd smart_traffic_frontend
npm ci
npm run dev
```

The frontend source is supplied, but the matching HTTP endpoints and camera
stream adapter are not included. It will require an API backend before live data
and chat work. See [frontend setup and API contract](smart_traffic_frontend/README.md).
Use `npm run build` to create a production frontend bundle.

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Video processing and terminal assistant integration |
| `core/` | Detection, tracking, and vehicle state |
| `features/` | Counting, speed, density, violations, and OCR |
| `Models/` | Supplied detection weights and provenance notes |
| `DataBase/` | Database utilities; runtime records are excluded |
| `Dashboard/` | Streamlit violation dashboard |
| `smart_traffic_ai_agent/` | LangGraph tools, memory, RAG, and configuration |
| `smart_traffic_frontend/` | React dashboard and API client |
| `Emergency_Detection.ipynb` | Supplied training notebook, without saved outputs |

## Current limits

- This is a local research and portfolio prototype.
- API services, model downloads, and end to end detection require local setup.
- The React frontend needs a backend adapter; it is not automatically connected to `main.py`.
- API keys, database records, evidence, cached files, and footage are excluded.
- Dependency versions and model evaluation are not locked or independently reproduced.
- Model and dataset provenance should be documented before wider distribution or deployment.

See [AI assistant notes](smart_traffic_ai_agent/README.md) for the supplied agent documentation.
