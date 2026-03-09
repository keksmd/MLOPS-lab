# Lessons

## 2026-03-09
- Pattern: Continued from stale assumptions (single app + root Dockerfile) after repo changed to backend/frontend layout.
- Correction: Re-scan project structure before changing CI and map workflow to actual build contexts.
- Rule: For this repository, CI must target `backend/Dockerfile` and `frontend/Dockerfile`, not a root Dockerfile.
- Pattern: Treated the repo root as the Python application root and ignored that the actual package/tests live under `backend/`.
- Correction: Run backend dependency install and tests from `backend/`, and treat root only as the uv workspace / Docker build context.
- Rule: In this repo, root is orchestration; backend execution context is `backend/`, frontend execution context is Bun workspace `frontend/`.
