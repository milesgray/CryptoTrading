import os
import shutil
import logging
import tempfile
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("artifact_service")

app = FastAPI(
    title="CryptoTrading Artifact Service",
    description="Service for saving, retrieving, listing, and managing model artifacts and system files.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_STORAGE_DIR = os.environ.get(
    "ARTIFACT_STORAGE_DIR",
    os.path.join(PROJECT_ROOT, "checkpoints")
)
TEMP_DIR = os.path.abspath(tempfile.gettempdir())

os.makedirs(DEFAULT_STORAGE_DIR, exist_ok=True)


def validate_safe_path(category: str, filename: Optional[str] = None) -> str:
    """Ensure path traversal attacks are prevented and target paths stay within storage directory."""
    # Sanitize category and filename
    clean_category = os.path.normpath(category).lstrip("/").lstrip("\\")
    base_dir = os.path.abspath(os.path.join(DEFAULT_STORAGE_DIR, clean_category))
    
    if not base_dir.startswith(os.path.abspath(DEFAULT_STORAGE_DIR)):
        raise HTTPException(status_code=400, detail="Invalid path: Path traversal detected in category.")
        
    if filename:
        clean_filename = os.path.basename(filename)
        target_path = os.path.abspath(os.path.join(base_dir, clean_filename))
        if not target_path.startswith(base_dir):
            raise HTTPException(status_code=400, detail="Invalid path: Path traversal detected in filename.")
        return target_path
        
    return base_dir


@app.get("/health")
def health_check():
    return {"status": "ok", "storage_dir": DEFAULT_STORAGE_DIR}


@app.post("/upload")
async def upload_artifact(
    file: UploadFile = File(...),
    category: str = Form("models", description="Category or namespace folder (e.g., 'models', 'checkpoints', 'reports')"),
    overwrite: bool = Form(False, description="Whether to overwrite if file already exists")
):
    """Save/upload an artifact file into the repository."""
    category_dir = validate_safe_path(category)
    os.makedirs(category_dir, exist_ok=True)
    
    target_path = validate_safe_path(category, file.filename)
    
    if os.path.exists(target_path) and not overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"File '{file.filename}' already exists in category '{category}'. Set overwrite=true to replace."
        )
        
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(target_path)
        logger.info(f"Artifact saved: {target_path} ({file_size} bytes)")
        return {
            "success": True,
            "filename": file.filename,
            "category": category,
            "path": os.path.relative_path(target_path, DEFAULT_STORAGE_DIR) if hasattr(os.path, 'relative_path') else target_path,
            "size_bytes": file_size
        }
    except Exception as e:
        logger.error(f"Error saving artifact {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save artifact: {str(e)}")


@app.get("/download/{category}/{filename:path}")
def download_artifact(category: str, filename: str):
    """Serve/download a specific artifact file."""
    file_path = validate_safe_path(category, filename)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Artifact '{filename}' not found in category '{category}'.")
        
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/octet-stream"
    )


@app.get("/files/{category}")
def list_category_files(category: str):
    """List all artifact files in a specific category directory."""
    category_dir = validate_safe_path(category)
    
    if not os.path.exists(category_dir):
        return {"category": category, "files": []}
        
    file_list = []
    for root, _, files in os.walk(category_dir):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, category_dir)
            stat = os.stat(full_p)
            file_list.append({
                "filename": rel_p,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime
            })
            
    return {"category": category, "files": file_list}


@app.get("/artifacts")
def list_all_artifacts():
    """List all categories and artifact files across the repository."""
    if not os.path.exists(DEFAULT_STORAGE_DIR):
        return {"categories": []}
        
    categories = {}
    for entry in os.listdir(DEFAULT_STORAGE_DIR):
        cat_path = os.path.join(DEFAULT_STORAGE_DIR, entry)
        if os.path.isdir(cat_path):
            files = []
            for root, _, filenames in os.walk(cat_path):
                for f in filenames:
                    fp = os.path.join(root, f)
                    stat = os.stat(fp)
                    files.append({
                        "filename": os.path.relpath(fp, cat_path),
                        "size_bytes": stat.st_size,
                        "modified_at": stat.st_mtime
                    })
            categories[entry] = files
            
    return {"categories": categories}


@app.delete("/files/{category}/{filename:path}")
def delete_artifact(category: str, filename: str):
    """Delete an artifact file from the repository."""
    file_path = validate_safe_path(category, filename)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Artifact '{filename}' not found in category '{category}'.")
        
    try:
        os.remove(file_path)
        logger.info(f"Artifact deleted: {file_path}")
        return {"success": True, "deleted": filename, "category": category}
    except Exception as e:
        logger.error(f"Error deleting artifact {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete artifact: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
