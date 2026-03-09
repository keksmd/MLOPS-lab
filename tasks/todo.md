# CI Re-Adaptation

## Tasks
- [x] Reconcile CI.yml with current repository branch model
- [x] Rebuild CI around the actual monorepo layout (backend uv package + frontend bun workspace)
- [x] Adapt image publish for backend/frontend Dockerfiles with explicit `/dada/` registry namespace
- [x] Adapt Argo update for backend and frontend repository/tag values
- [x] Validate workflow syntax

## Review
- `backend_test` now runs from `backend/` with `uv` and a Postgres service, matching how the backend package and `.env` are structured.
- `frontend_build` now installs Bun dependencies, generates `frontend/openapi.json` from the backend app, regenerates the client, and builds the frontend.
- `build_publish` now publishes two images from `backend/Dockerfile` and `frontend/Dockerfile` to `nexus.dada-tuda.ru/dada/...`.
- `update_argo_repo` now updates both `repository` and `tag` fields for backend and frontend values files.
- YAML parse validation passed (`YAML_OK`).
