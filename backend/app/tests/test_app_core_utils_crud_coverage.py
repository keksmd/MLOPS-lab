
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from jwt.exceptions import InvalidTokenError

import app.backend_pre_start as backend_pre_start
import app.core.db as core_db
import app.core.security as security
import app.crud as crud
import app.initial_data as initial_data
import app.tests_pre_start as tests_pre_start
import app.utils as utils
from app.api import deps
from app.models import ItemCreate, User, UserCreate, UserUpdate


class _ExecResult:
    def __init__(
        self,
        *,
        first: object | None = None,
        one: object | None = None,
        all_values: list[object] | None = None,
    ) -> None:
        self._first = first
        self._one = one
        self._all = all_values or []

    def first(self) -> object | None:
        return self._first

    def one(self) -> object | None:
        return self._one

    def all(self) -> list[object]:
        return self._all


class _FakeSession:
    def __init__(self, exec_results: list[_ExecResult] | None = None) -> None:
        self.exec_results = exec_results or []
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.commits = 0
        self.refreshed: list[object] = []
        self.got: dict[tuple[object, object], object | None] = {}

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        self.commits += 1

    def refresh(self, obj: object) -> None:
        self.refreshed.append(obj)

    def exec(self, statement: object) -> _ExecResult:
        if self.exec_results:
            return self.exec_results.pop(0)
        return _ExecResult()

    def get(self, model: object, key: object) -> object | None:
        return self.got.get((model, key))

    def delete(self, obj: object) -> None:
        self.deleted.append(obj)


def _make_user(
    *,
    email: str = "user@example.com",
    is_superuser: bool = False,
    is_active: bool = True,
) -> User:
    return User(
        id=uuid4(),
        email=email,
        hashed_password="hashed",
        full_name="Test User",
        is_superuser=is_superuser,
        is_active=is_active,
    )


def test_crud_create_update_get_and_create_item(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(exec_results=[_ExecResult(first=None)])
    monkeypatch.setattr(crud, "get_password_hash", lambda password: f"hashed:{password}")

    created = crud.create_user(
        session=session,
        user_create=UserCreate(
            email="new@example.com",
            password="password123",
            full_name="New User",
        ),
    )
    assert created.hashed_password == "hashed:password123"
    assert session.commits == 1
    assert session.refreshed[-1] is created

    updated = crud.update_user(
        session=session,
        db_user=created,
        user_in=UserUpdate(full_name="Updated", password="newpassword123"),
    )
    assert updated.full_name == "Updated"
    assert updated.hashed_password == "hashed:newpassword123"

    monkeypatch.setattr(
        session,
        "exec",
        lambda statement: _ExecResult(first=created),
    )
    assert crud.get_user_by_email(session=session, email="new@example.com") is created

    item = crud.create_item(
        session=session,
        item_in=ItemCreate(title="Item title", description="desc"),
        owner_id=created.id,
    )
    assert item.owner_id == created.id
    assert session.refreshed[-1] is item


def test_crud_authenticate_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    user = _make_user()
    session = _FakeSession()

    monkeypatch.setattr(crud, "get_user_by_email", lambda session, email: None)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        crud,
        "verify_password",
        lambda password, hashed: calls.append((password, hashed)) or (False, None),
    )
    assert crud.authenticate(session=session, email="none@example.com", password="pw") is None
    assert calls and calls[0][1] == crud.DUMMY_HASH

    monkeypatch.setattr(crud, "get_user_by_email", lambda session, email: user)
    monkeypatch.setattr(crud, "verify_password", lambda password, hashed: (False, None))
    assert crud.authenticate(session=session, email="user@example.com", password="pw") is None

    monkeypatch.setattr(
        crud,
        "verify_password",
        lambda password, hashed: (True, "updated-hash"),
    )
    authenticated = crud.authenticate(
        session=session,
        email="user@example.com",
        password="pw",
    )
    assert authenticated is user
    assert user.hashed_password == "updated-hash"
    assert session.commits == 1


def test_security_helpers_and_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(security.settings, "SECRET_KEY", "secret", raising=False)

    token = security.create_access_token("subject", timedelta(minutes=5))
    assert isinstance(token, str)

    hashed = security.get_password_hash("password123")
    verified, maybe_updated = security.verify_password("password123", hashed)
    assert verified is True
    assert maybe_updated is None


def test_utils_email_helpers_and_password_reset_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils.settings, "PROJECT_NAME", "Demo", raising=False)
    monkeypatch.setattr(utils.settings, "FRONTEND_HOST", "https://frontend", raising=False)
    monkeypatch.setattr(utils.settings, "EMAIL_RESET_TOKEN_EXPIRE_HOURS", 2, raising=False)
    monkeypatch.setattr(utils.settings, "SECRET_KEY", "secret", raising=False)
    monkeypatch.setattr(utils.settings, "EMAILS_FROM_NAME", "Demo", raising=False)
    monkeypatch.setattr(utils.settings, "EMAILS_FROM_EMAIL", "demo@example.com", raising=False)
    monkeypatch.setattr(utils.settings, "SMTP_HOST", "smtp.example.com", raising=False)
    monkeypatch.setattr(utils.settings, "SMTP_PORT", 2525, raising=False)
    monkeypatch.setattr(utils.settings, "SMTP_TLS", True, raising=False)
    monkeypatch.setattr(utils.settings, "SMTP_SSL", False, raising=False)
    monkeypatch.setattr(utils.settings, "SMTP_USER", "user", raising=False)
    monkeypatch.setattr(utils.settings, "SMTP_PASSWORD", "pass", raising=False)

    monkeypatch.setattr(
        utils.Path,
        "read_text",
        lambda self: "Hello {{ project_name }} {{ email }} {{ username|default('') }}",
    )

    html = utils.render_email_template(
        template_name="dummy.html",
        context={"project_name": "Demo", "email": "user@example.com"},
    )
    assert "Hello Demo user@example.com" in html

    sent: dict[str, object] = {}

    class _FakeResponse:
        def __str__(self) -> str:
            return "ok"

    class _FakeMessage:
        def __init__(self, *, subject: str, html: str, mail_from: tuple[str, str]) -> None:
            sent["subject"] = subject
            sent["html"] = html
            sent["mail_from"] = mail_from

        def send(self, *, to: str, smtp: dict[str, object]) -> _FakeResponse:
            sent["to"] = to
            sent["smtp"] = smtp
            return _FakeResponse()

    monkeypatch.setattr(utils.settings, "emails_enabled", True, raising=False)
    monkeypatch.setattr(utils.emails, "Message", _FakeMessage)

    utils.send_email(
        email_to="user@example.com",
        subject="Subject",
        html_content="<b>Hello</b>",
    )
    assert sent["to"] == "user@example.com"
    assert sent["subject"] == "Subject"
    assert sent["smtp"] == {
        "host": "smtp.example.com",
        "port": 2525,
        "tls": True,
        "user": "user",
        "password": "pass",
    }

    monkeypatch.setattr(utils.settings, "emails_enabled", False, raising=False)
    with pytest.raises(AssertionError):
        utils.send_email(email_to="user@example.com")

    monkeypatch.setattr(utils.settings, "emails_enabled", True, raising=False)
    test_email = utils.generate_test_email("user@example.com")
    assert test_email.subject == "Demo - Test email"

    reset_email = utils.generate_reset_password_email(
        email_to="user@example.com",
        email="user@example.com",
        token="token",
    )
    assert "Password recovery" in reset_email.subject

    new_account_email = utils.generate_new_account_email(
        email_to="user@example.com",
        username="user@example.com",
        password="password123",
    )
    assert "New account" in new_account_email.subject

    token = utils.generate_password_reset_token("user@example.com")
    assert utils.verify_password_reset_token(token) == "user@example.com"
    assert utils.verify_password_reset_token("bad-token") is None


def test_deps_get_db_and_current_user(monkeypatch: pytest.MonkeyPatch) -> None:
    yielded_session = object()

    class _SessionCtx:
        def __init__(self, engine: object) -> None:
            self.engine = engine

        def __enter__(self) -> object:
            return yielded_session

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(deps, "Session", _SessionCtx)
    gen = deps.get_db()
    assert next(gen) is yielded_session
    with pytest.raises(StopIteration):
        next(gen)

    session = _FakeSession()
    monkeypatch.setattr(deps.jwt, "decode", lambda token, key, algorithms: {"sub": str(uuid4())})
    monkeypatch.setattr(session, "get", lambda model, key: None)
    with pytest.raises(Exception) as exc_info:
        deps.get_current_user(session=session, token="token")
    assert exc_info.value.status_code == 404

    inactive_user = _make_user(is_active=False)
    monkeypatch.setattr(session, "get", lambda model, key: inactive_user)
    with pytest.raises(Exception) as exc_info:
        deps.get_current_user(session=session, token="token")
    assert exc_info.value.status_code == 400

    active_user = _make_user(is_superuser=True)
    monkeypatch.setattr(session, "get", lambda model, key: active_user)
    current = deps.get_current_user(session=session, token="token")
    assert current is active_user
    assert deps.get_current_active_superuser(active_user) is active_user

    with pytest.raises(Exception) as exc_info:
        deps.get_current_active_superuser(_make_user(is_superuser=False))
    assert exc_info.value.status_code == 403

    monkeypatch.setattr(
        deps.jwt,
        "decode",
        lambda token, key, algorithms: (_ for _ in ()).throw(InvalidTokenError("bad")),
    )
    with pytest.raises(Exception) as exc_info:
        deps.get_current_user(session=session, token="bad-token")
    assert exc_info.value.status_code == 403


def test_init_db_and_initial_data(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(exec_results=[_ExecResult(first=None)])
    created: dict[str, object] = {}

    monkeypatch.setattr(
        core_db.crud,
        "create_user",
        lambda session, user_create: created.setdefault("user_create", user_create),
    )
    monkeypatch.setattr(core_db.settings, "FIRST_SUPERUSER", "admin@example.com", raising=False)
    monkeypatch.setattr(
        core_db.settings,
        "FIRST_SUPERUSER_PASSWORD",
        "adminpassword123",
        raising=False,
    )

    core_db.init_db(session)
    assert created["user_create"].email == "admin@example.com"

    session_existing = _FakeSession(exec_results=[_ExecResult(first=_make_user())])
    created.clear()
    core_db.init_db(session_existing)
    assert created == {}


def _run_prestart_success(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    class _SessionCtx:
        def __init__(self, engine: object) -> None:
            self.engine = engine

        def __enter__(self) -> _FakeSession:
            return _FakeSession()

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(module, "Session", _SessionCtx)
    module.init.__wrapped__(object())


def _run_prestart_failure(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    class _SessionCtx:
        def __init__(self, engine: object) -> None:
            self.engine = engine

        def __enter__(self) -> _FakeSession:
            raise RuntimeError("db down")

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(module, "Session", _SessionCtx)
    with pytest.raises(RuntimeError):
        module.init.__wrapped__(object())


def test_prestart_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_prestart_success(tests_pre_start, monkeypatch)
    _run_prestart_failure(tests_pre_start, monkeypatch)
    _run_prestart_success(backend_pre_start, monkeypatch)
    _run_prestart_failure(backend_pre_start, monkeypatch)

    calls: list[str] = []
    monkeypatch.setattr(tests_pre_start, "init", lambda engine: calls.append("init"))
    monkeypatch.setattr(tests_pre_start.logger, "info", lambda msg: calls.append(msg))
    tests_pre_start.main()
    assert calls == ["Initializing service", "init", "Service finished initializing"]

    calls = []
    monkeypatch.setattr(backend_pre_start, "init", lambda engine: calls.append("init"))
    monkeypatch.setattr(backend_pre_start.logger, "info", lambda msg: calls.append(msg))
    backend_pre_start.main()
    assert calls == ["Initializing service", "init", "Service finished initializing"]
