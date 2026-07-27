# Project Memory

## Last Completed Task
Created Artifact Service microservice for saving, retrieving, listing, and serving model artifacts and system files, with Docker/Docker Compose support and cross-service client integrations across `train`, `predict`, `pressure`, `embed`, `jepa`, and `retrieval`.

## Architecture Notes
- Artifact Service runs as a standalone FastAPI service (`services/artifact/main.py`) on port `8006`.
- Centralized Python helper client created under `cryptotrading.client.artifact.service` (exposing `upload_artifact`, `download_artifact`, and `list_category_files`).
- Models in `train`, `predict`, `pressure`, `embed`, and `jepa` auto-sync to the Artifact Service on save and attempt automatic download on service startup if missing locally.

## Environment / Config
- `ARTIFACT_SERVICE_URL`: Base URL of the Artifact Service (default: `http://artifact:8006`).
- `ARTIFACT_STORAGE_DIR`: Local path inside the artifact container (default: `/app/checkpoints`).

## Dependencies Added
- `python-multipart` added to `services/artifact/pyproject.toml`.

## Known Blockers / Next Steps
- None.
