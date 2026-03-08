# Lessons

## 2026-03-09
- Pattern: Continued from stale assumptions (single app + root Dockerfile) after repo changed to backend/frontend layout.
- Correction: Re-scan project structure before changing CI and map workflow to actual build contexts.
- Rule: For this repository, CI must target `backend/Dockerfile` and `frontend/Dockerfile`, not a root Dockerfile.
