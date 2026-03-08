# CI Re-Adaptation

## Tasks
- [x] Reconcile CI.yml with current repository branch model
- [x] Adapt build pipeline for two applications (backend/frontend) without root Dockerfile
- [x] Adapt Argo update for backend and frontend values files
- [x] Validate workflow syntax

## Review
- `build_publish` now builds/pushes two images: `backend/Dockerfile` and `frontend/Dockerfile`.
- Image repositories are split with vars:
  - `NEXUS_DOCKER_REPOSITORY_BACKEND`
  - `NEXUS_DOCKER_REPOSITORY_FRONTEND`
- Argo update now updates both values files (backend + frontend) with the same image tag.
- YAML parse validation passed (`YAML_OK`).
