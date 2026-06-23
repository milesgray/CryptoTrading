# Task Log: Poetry to uv Migration

## Task Information
- **Date**: 2026-06-22
- **Time Started**: 21:21
- **Time Completed**: 21:28
- **Files Modified**:
  - `src/pyproject.toml`
  - `services/serve/pyproject.toml`
  - `src/cryptotrading/rollbit/prices/serve/pyproject.toml`
  - `Dockerfile.serve`
  - `Dockerfile.record`
  - `services/serve/Dockerfile`
  - `src/cryptotrading/rollbit/prices/serve/Dockerfile`
  - `src/dev.sh`
  - `src/record.sh`
  - `src/serve.sh`
  - `services/pressure/README.md`
  - `services/train/README.md`
  - `services/price/README.md`
  - `services/README.md`
  - `services/predict/README.md`
  - `.agent/core/techContext.md`
  - `.agent/core/activeContext.md`
  - `.agent/core/progress.md`

## Task Details
- **Goal**: Switch the Python dependency manager from Poetry to `uv` everywhere in the codebase.
- **Implementation**:
  - Converted three `pyproject.toml` files to use PEP 621 metadata standard and Hatchling build backend.
  - Configured non-package microservices using the `[tool.uv]` package-less config to skip unnecessary wheel packaging.
  - Deleted legacy `poetry.lock` files and generated new `uv.lock` files.
  - Updated four Dockerfiles to install `uv` via fast binaries and use `uv sync --frozen --no-install-project` for cached dependency builds.
  - Added `build-essential` and `g++` compilers to slim Docker containers to build C++ extensions (e.g. `annoy`).
  - Rewrote run orchestration scripts (`dev.sh`, `record.sh`, `serve.sh`) to use `uv run`.
  - Re-documented execution and setups across README files, active contexts, and progress files.
- **Challenges**:
  - Encountered an issue compiling `annoy` because python-slim Docker images lack compiler tools by default. Fixed by installing `build-essential` and `g++` in the Dockerfiles.
  - Encountered project wheel build errors due to missing package directories and missing `README.md` in Docker layers. Fixed by using `--no-install-project` flags during `uv sync` and copying `README.md` early in Docker steps.
- **Decisions**:
  - Configured `uv sync` using `--no-install-project` to prevent unnecessary editable wheel packaging errors in production containers.

## Performance Evaluation
- **Score**: 22/23
- **Strengths**: High execution speed, comprehensive migration coverage of files, and prompt handling/debugging of Docker build failures.
- **Areas for Improvement**: Could have anticipated that C++ extensions like `annoy` would need compilers in standard python-slim containers.

## Next Steps
- Continue implementing Phase 4 goals: Integrate `pgvector` HNSW queries into the REST API.
