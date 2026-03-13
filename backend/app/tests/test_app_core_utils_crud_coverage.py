from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from app import crud, initial_data, utils
from app.models import ItemCreate, User, UserCreate, UserUpdate


class _FakeExecResult:
    def __init__(self, *, first_value: Any | None = None) -> None:
        self._first_value = first_value

    def first(self) -> Any:
        return self._first_value


class _FakeSession:
    def __init__(self, *, exec_results: list[_FakeExecResult] | None = None) -> None:
        self.exec_results = exec_results or []
        self.added: list[Any] = []
        self.deleted: list[Any] = []
        self.commits = 0
        self.refreshes: list[Any] = []

    def exec(self, statement: Any) -> _FakeExecResult:
        if self.exec_results:
            return self.exec_results.pop(0)
        return _FakeExecResult()

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    def delete(self, obj: Any) -> None:
        self.deleted.append(obj)

    def commit(self) -> None:
        self.commits += 1

    def refresh(self, obj: Any) -> None:
        self.refreshes.append(obj)


def _make_user(
    *,
    email: str = "user@example.com",
    is_superuser: bool = False,
    is_active: bool = True,
) -> User:
    return User(
        id=uuid4(),
        email=email,
        full_name="User",
        is_superuser=is_superuser,
        is_active=is_active,
        hashed_password="hashed",
        created_at=datetime.now(timezone.utc),
    )


def test_crud_create_update_get_and_create_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(exec_results=[_FakeExecResult(first_value=None)])
    monkeypatch.setattr(
        crud, "get_password_hash", lambda password: "test-password-hash"
    )

    created = crud.create_user(
        session=session,
        user_create=UserCreate(
            email="new@example.com",
            password="password123",
            full_name="New User",
        ),
    )
    assert created.email == "new@example.com"
    assert created.hashed_password == "test-password-hash"
    assert session.added
    assert session.commits == 1
    assert session.refreshes == [created]

    existing = _make_user(email="exists@example.com")
    session = _FakeSession(exec_results=[_FakeExecResult(first_value=existing)])
    found = crud.get_user_by_email(session=session, email="exists@example.com")
    assert found == existing

    monkeypatch.setattr(
        crud, "verify_password", lambda plain_password, hashed_password: (True, None)
    )
    monkeypatch.setattr(crud, "get_user_by_email", lambda session, email: existing)
    authenticated = crud.authenticate(
        session=session, email="exists@example.com", password="password123"
    )
    assert authenticated == existing

    monkeypatch.setattr(
        crud, "verify_password", lambda plain_password, hashed_password: (False, None)
    )
    assert (
        crud.authenticate(session=session, email="exists@example.com", password="wrong")
        is None
    )

    monkeypatch.setattr(
        crud, "get_password_hash", lambda password: "updated-password-hash"
    )
    updated = crud.update_user(
        session=session,
        db_user=existing,
        user_in=UserUpdate(full_name="Updated", password="newpass123"),
    )
    assert updated.full_name == "Updated"
    assert updated.hashed_password == "updated-password-hash"

    owner = _make_user()
    item = crud.create_item(
        session=session,
        item_in=ItemCreate(title="Item", description="Desc"),
        owner_id=owner.id,
    )
    assert item.owner_id == owner.id
    assert item.title == "Item"


def test_utils_email_helpers_and_password_reset_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils.settings, "PROJECT_NAME", "Demo")
    monkeypatch.setattr(utils.settings, "FRONTEND_HOST", "https://frontend")
    monkeypatch.setattr(utils.settings, "EMAIL_RESET_TOKEN_EXPIRE_HOURS", 2)
    monkeypatch.setattr(utils.settings, "SECRET_KEY", "secret")
    monkeypatch.setattr(utils.settings, "EMAILS_FROM_NAME", "Demo")
    monkeypatch.setattr(utils.settings, "EMAILS_FROM_EMAIL", "demo@example.com")
    monkeypatch.setattr(utils.settings, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(utils.settings, "SMTP_PORT", 2525)
    monkeypatch.setattr(utils.settings, "SMTP_TLS", True)
    monkeypatch.setattr(utils.settings, "SMTP_SSL", False)
    monkeypatch.setattr(utils.settings, "SMTP_USER", "user")
    monkeypatch.setattr(utils.settings, "SMTP_PASSWORD", "pass")

    original_read_text = Path.read_text

    def _fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "dummy.html":
            return "Hello {{ project_name }} {{ email }} {{ username|default('') }}"
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)

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
        def __init__(
            self, *, subject: str, html: str, mail_from: tuple[str, str]
        ) -> None:
            sent["subject"] = subject
            sent["html"] = html
            sent["mail_from"] = mail_from

        def send(self, *, to: str, smtp: dict[str, object]) -> _FakeResponse:
            sent["to"] = to
            sent["smtp"] = smtp
            return _FakeResponse()

    monkeypatch.setattr(utils.emails, "Message", _FakeMessage)

    utils.send_email(
        email_to="user@example.com",
        subject="Subject",
        html_content="<b>Hello</b>",
    )
    assert sent["to"] == "user@example.com"
    assert sent["subject"] == "Subject"

    reset_token = utils.generate_password_reset_token("user@example.com")
    assert utils.verify_password_reset_token(reset_token) == "user@example.com"

    new_account = utils.generate_new_account_email(
        email_to="user@example.com",
        username="user@example.com",
        password="password123",
    )
    assert "subject" in new_account.__dict__
    assert "html_content" in new_account.__dict__

    reset_email = utils.generate_reset_password_email(
        email_to="user@example.com",
        email="user@example.com",
        token="token",
    )
    assert "subject" in reset_email.__dict__
    assert "html_content" in reset_email.__dict__

    test_email = utils.generate_test_email(email_to="user@example.com")
    assert "subject" in test_email.__dict__
    assert "html_content" in test_email.__dict__


def test_initial_data_init_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    session_instance = object()
    entered: dict[str, Any] = {}

    class _SessionContext:
        def __init__(self, engine: Any) -> None:
            entered["engine"] = engine

        def __enter__(self) -> object:
            entered["entered"] = True
            return session_instance

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            entered["exited"] = True

    monkeypatch.setattr(initial_data, "Session", _SessionContext)
    monkeypatch.setattr(initial_data, "engine", object())

    called: dict[str, Any] = {}

    def _fake_init_db(session: object) -> None:
        called["session"] = session

    monkeypatch.setattr(initial_data, "init_db", _fake_init_db)

    initial_data.init()
    assert called["session"] is session_instance
    assert entered["entered"] is True
    assert entered["exited"] is True

    monkeypatch.setattr(
        initial_data, "init", lambda: called.setdefault("main_called", True)
    )
    initial_data.main()
    assert called["main_called"] is True
