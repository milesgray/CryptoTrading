# Artifact Service

FastAPI service for storing, serving, listing, and managing model artifacts and system files.

## Endpoints

- `POST /upload`: Upload file/artifact to a specified category/namespace.
- `GET /download/{category}/{filename}`: Download/serve a specific artifact file.
- `GET /files/{category}`: List all files within a category.
- `GET /artifacts`: List all categories and summary of artifacts.
- `DELETE /files/{category}/{filename}`: Delete an artifact file.
- `GET /health`: Service health check.
