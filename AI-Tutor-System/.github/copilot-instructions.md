# Copilot Instructions for AI-Tutor-System

## Project Overview
- This is a modular tutoring system for educational research, combining intent classification (BERT), LLM-based response generation (Ollama), metrics tracking, and virtual user simulation.
- Main components:
  - `intent_api.py`: FastAPI service for intent detection (port 9000)
  - `llm_api.py`: FastAPI service for LLM completions (port 8000), routes requests to Ollama
  - `simulator.py`: Runs virtual user personas to test the system end-to-end
  - `models.py`: SQLAlchemy ORM models (mainly `Interaction`)
  - `database.py`: DB session/context management, stats, and CLI
  - `metrics.py`: Metrics computation and CSV export

## Data Flow & Architecture
- User or simulator sends a prompt → `llm_api.py` → (calls `intent_api.py` for intent) → generates LLM response → logs to SQLite via `models.py`/`database.py`.
- All interactions and metrics are stored in `data/tutoring_metrics.db` (SQLite).
- Reports and exports are saved in `reports/`.
- Virtual users simulate 4 personas (lazy, curious, persistent, strategic) for robust testing.

## Developer Workflows
- **Setup:**
  - Install Python deps: `pip install -r requirements.txt`
  - Ensure Ollama is installed and running with a model (see README)
- **Run APIs:**
  - `python intent_api.py` (port 9000)
  - `python llm_api.py` (port 8000)
- **Simulate Users:**
  - `python simulator.py` (generates reports in `reports/`)
- **DB Management:**
  - `python database.py` (init/stats)
  - Use `get_db_context()` for direct SQLAlchemy access
  - Export metrics: see `metrics.py` and README for code snippets
- **Testing APIs:**
  - Use `curl` commands in README for manual API checks

## Project Conventions & Patterns
- All database access uses SQLAlchemy ORM (`models.py`), with context/session helpers in `database.py`.
- FastAPI is used for all API services, with automatic OpenAPI docs at `/docs`.
- Persona logic and simulation are centralized in `simulator.py`.
- Model/LLM config is set in each API file (see comments in `intent_api.py` and `llm_api.py`).
- All persistent data (DB, reports) is stored in `data/` and `reports/`.
- Use `to_dict()` on ORM models for JSON serialization.

## Integration Points
- Ollama must be running for LLM completions (`llm_api.py`).
- All API endpoints are local (localhost, different ports).
- SQLite DB is file-based; can be browsed with DB Browser for SQLite or VS Code extensions.

## Examples
- See README for API usage, persona simulation, and metrics export code.
- Example: Export metrics to CSV:
  ```python
  from database import get_db_context
  from metrics import export_metrics_csv
  with get_db_context() as db:
      export_metrics_csv(db, "reports/metrics_export.csv")
  ```

---

For new features, follow the modular structure and reuse context/session helpers. See README for more details and code snippets.
