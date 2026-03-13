from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from app.api.routes import items, login, private, users, utils as route_utils
from app.models import Item, ItemCreate, ItemUpdate, User, UserCreate, UserUpdate


@dataclass
class _ExecResult:
    one_value: Any | None = None
    first_value: Any | None = None
    all_values: list[Any] = field(default_factory=list)

    def one(self) -> Any:
        return self.one_value

    def first(self) -> Any:
        return self.first_value

    def all(self) -> list[Any]:
        return self.all_values


class _FakeSession:
    def __init__(
        self,
        *,
        exec_results: list[_ExecResult] | None = None,
        get_result: Any | None = None,
    ) -> None:
        self.exec_results = exec_results or []
        self.get_result = get_result
        self.added: list[Any] = []
        self.deleted: list[Any] = []
        self.commits = 0
        self.refreshes: list[Any] = []

    def exec(self, statement: Any) -> _ExecResult:
        if self.exec_results:
            return self.exec_results.pop(0)
        return _ExecResult()

    def get(self, model: type[Any], object_id: UUID) -> Any:
        return self.get_result

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
    user_id: UUID | None = None,
    email: str = "user@example.com",
    is_superuser: bool = False,
    is_active: bool = True,
) -> User:
    return User(
        id=user_id or uuid4(),
        email=email,
        full_name="User",
        is_superuser=is_superuser,
        is_active=is_active,
        hashed_password="hashed",
        created_at=datetime.now(timezone.utc),
    )


def _make_item(owner_id: UUID, *, item_id: UUID | None = None) -> Item:
    return Item(
        id=item_id or uuid4(),
        title="Item",
        description="Desc",
        owner_id=owner_id,
        created_at=datetime.now(timezone.utc),
    )


def test_items_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _make_user(is_superuser=True)
    normal = _make_user()
    own_item = _make_item(normal.id)
    other_item = _make_item(uuid4())

    session = _FakeSession(
        exec_results=[
            _ExecResult(one_value=3),
            _ExecResult(all_values=[own_item]),
        ]
    )
    result = items.read_items(session=session, current_user=admin, skip=0, limit=10)
    assert result.count == 3
    assert len(result.data) == 1
    assert result.data[0].id == own_item.id
    assert result.data[0].owner_id == own_item.owner_id

    session = _FakeSession(
        exec_results=[
            _ExecResult(one_value=1),
            _ExecResult(all_values=[own_item]),
        ]
    )
    result = items.read_items(session=session, current_user=normal, skip=0, limit=10)
    assert result.count == 1
    assert len(result.data) == 1
    assert result.data[0].id == own_item.id

    created_item = _make_item(normal.id)
    monkeypatch.setattr(items, "Item", Item)
    session = _FakeSession()
    item_in = ItemCreate(title="New", description="Item")
    result = items.create_item(
        session=session,
        current_user=normal,
        item_in=item_in,
    )
    assert session.commits == 1
    assert session.added
    assert session.refreshes
    assert result.owner_id == normal.id

    session = _FakeSession(get_result=own_item)
    result = items.read_item_by_id(
        item_id=own_item.id,
        session=session,
        current_user=normal,
    )
    assert result.id == own_item.id

    session = _FakeSession(get_result=other_item)
    with pytest.raises(HTTPException) as exc_info:
        items.read_item_by_id(
            item_id=other_item.id,
            session=session,
            current_user=normal,
        )
    assert exc_info.value.status_code == 400

    session = _FakeSession(get_result=None)
    with pytest.raises(HTTPException) as exc_info:
        items.read_item_by_id(
            item_id=uuid4(),
            session=session,
            current_user=admin,
        )
    assert exc_info.value.status_code == 404

    session = _FakeSession(get_result=own_item)
    result = items.update_item(
        item_id=own_item.id,
        session=session,
        current_user=normal,
        item_in=ItemUpdate(title="Updated", description="Updated desc"),
    )
    assert result.title == "Updated"
    assert result.description == "Updated desc"
    assert session.commits == 1

    session = _FakeSession(get_result=other_item)
    with pytest.raises(HTTPException) as exc_info:
        items.update_item(
            item_id=other_item.id,
            session=session,
            current_user=normal,
            item_in=ItemUpdate(title="Nope"),
        )
    assert exc_info.value.status_code == 400

    session = _FakeSession(get_result=own_item)
    result = items.delete_item(
        item_id=own_item.id,
        session=session,
        current_user=normal,
    )
    assert result.message
    assert session.deleted == [own_item]
    assert session.commits == 1


def test_users_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    listed_user = _make_user(email="listed@example.com")
    session = _FakeSession(
        exec_results=[
            _ExecResult(one_value=2),
            _ExecResult(all_values=[listed_user]),
        ]
    )
    read_result = users.read_users(session=session, skip=0, limit=10)
    assert read_result.count == 2
    assert len(read_result.data) == 1
    assert read_result.data[0].email == "listed@example.com"

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: _make_user())
    with pytest.raises(HTTPException) as exc_info:
        users.create_user(
            session=session,
            user_in=UserCreate(
                email="exists@example.com",
                password="password123",
                full_name="Exists",
            ),
        )
    assert exc_info.value.status_code == 400

    created_user = _make_user(email="new@example.com")
    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: None)
    monkeypatch.setattr(users.crud, "create_user", lambda session, user_create: created_user)
    monkeypatch.setattr(users.settings, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(users.settings, "EMAILS_FROM_EMAIL", "noreply@example.com")
    monkeypatch.setattr(users, "generate_new_account_email", lambda **kwargs: {"subject": "subj", "html_content": "html"})
    sent: dict[str, Any] = {}
    monkeypatch.setattr(users, "send_email", lambda **kwargs: sent.update(kwargs))

    result = users.create_user(
        session=session,
        user_in=UserCreate(
            email="new@example.com",
            password="password123",
            full_name="New",
        ),
    )
    assert result.email == "new@example.com"
    assert sent["email_to"] == "new@example.com"

    current_user = _make_user()
    session = _FakeSession(get_result=current_user)
    result = users.read_user_by_id(
        user_id=current_user.id,
        session=session,
        current_user=current_user,
    )
    assert result == current_user

    other_user = _make_user(email="other@example.com")
    session = _FakeSession(get_result=other_user)
    with pytest.raises(HTTPException) as exc_info:
        users.read_user_by_id(
            user_id=other_user.id,
            session=session,
            current_user=current_user,
        )
    assert exc_info.value.status_code == 403

    admin_user = _make_user(email="admin@example.com", is_superuser=True)
    session = _FakeSession(get_result=other_user)
    result = users.read_user_by_id(
        user_id=other_user.id,
        session=session,
        current_user=admin_user,
    )
    assert result.email == "other@example.com"

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: other_user)
    with pytest.raises(HTTPException) as exc_info:
        users.update_user_me(
            session=session,
            current_user=current_user,
            user_in=UserUpdate(email="other@example.com"),
        )
    assert exc_info.value.status_code == 409

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: None)

    def _fake_update_user(*, session: Any, db_user: User, user_in: UserUpdate) -> User:
        if user_in.full_name is not None:
            db_user.full_name = user_in.full_name
        if user_in.email is not None:
            db_user.email = user_in.email
        return db_user

    monkeypatch.setattr(users.crud, "update_user", _fake_update_user)

    result = users.update_user_me(
        session=session,
        current_user=current_user,
        user_in=UserUpdate(full_name="Updated User"),
    )
    assert result.full_name == "Updated User"

    with pytest.raises(HTTPException) as exc_info:
        users.update_password_me(
            session=session,
            current_user=current_user,
            body=SimpleNamespace(current_password="wrong", new_password="new-pass-123"),
        )
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(users, "verify_password", lambda plain_password, hashed_password: True)

    with pytest.raises(HTTPException) as exc_info:
        users.update_password_me(
            session=session,
            current_user=current_user,
            body=SimpleNamespace(current_password="same-pass", new_password="same-pass"),
        )
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(users, "get_password_hash", lambda password: "new-hash")

    result = users.update_password_me(
        session=session,
        current_user=current_user,
        body=SimpleNamespace(current_password="old-pass", new_password="new-pass-123"),
    )
    assert result.message
    assert current_user.hashed_password == "new-hash"

    result = users.read_user_me(current_user=current_user)
    assert result.email == current_user.email

    with pytest.raises(HTTPException) as exc_info:
        users.delete_user_me(session=session, current_user=admin_user)
    assert exc_info.value.status_code == 403

    result = users.delete_user_me(session=session, current_user=current_user)
    assert result.message
    assert current_user in session.deleted

    session = _FakeSession(get_result=current_user)
    with pytest.raises(HTTPException) as exc_info:
        users.delete_user(session=session, current_user=admin_user, user_id=admin_user.id)
    assert exc_info.value.status_code == 403

    session = _FakeSession(get_result=other_user)
    result = users.delete_user(session=session, current_user=admin_user, user_id=other_user.id)
    assert result.message
    assert other_user in session.deleted

    monkeypatch.setattr(users.settings, "USERS_OPEN_REGISTRATION", True)
    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: None)
    monkeypatch.setattr(users.crud, "create_user", lambda session, user_create: created_user)
    signup_result = users.register_user(
        session=session,
        user_in=UserCreate(
            email="signup@example.com",
            password="password123",
            full_name="Signup User",
        ),
    )
    assert signup_result.email == created_user.email


def test_login_private_and_utils_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    active_user = _make_user()
    inactive_user = _make_user(is_active=False)

    monkeypatch.setattr(login.crud, "authenticate", lambda session, email, password: None)
    with pytest.raises(HTTPException) as exc_info:
        login.login_access_token(
            session=_FakeSession(),
            form_data=SimpleNamespace(username="x@example.com", password="bad"),
        )
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login.crud, "authenticate", lambda session, email, password: inactive_user)
    with pytest.raises(HTTPException) as exc_info:
        login.login_access_token(
            session=_FakeSession(),
            form_data=SimpleNamespace(username="x@example.com", password="bad"),
        )
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login.crud, "authenticate", lambda session, email, password: active_user)
    monkeypatch.setattr(login.security, "create_access_token", lambda subject, expires_delta: "token")
    token = login.login_access_token(
        session=_FakeSession(),
        form_data=SimpleNamespace(username="user@example.com", password="password123"),
    )
    assert token.access_token == "token"

    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: active_user)
    monkeypatch.setattr(login.settings, "EMAIL_RESET_TOKEN_EXPIRE_HOURS", 2)
    monkeypatch.setattr(login, "generate_password_reset_token", lambda email: "reset-token")
    monkeypatch.setattr(login, "generate_reset_password_email", lambda **kwargs: {"subject": "subj", "html_content": "html"})
    sent: dict[str, Any] = {}
    monkeypatch.setattr(login, "send_email", lambda **kwargs: sent.update(kwargs))
    result = login.recover_password(email="user@example.com", session=_FakeSession())
    assert result.message
    assert sent["email_to"] == "user@example.com"

    monkeypatch.setattr(login, "verify_password_reset_token", lambda token: None)
    with pytest.raises(HTTPException) as exc_info:
        login.reset_password(session=_FakeSession(), body=SimpleNamespace(token="bad", new_password="new-pass"))
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login, "verify_password_reset_token", lambda token: "user@example.com")
    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: active_user)
    monkeypatch.setattr(login, "get_password_hash", lambda password: "reset-hash")
    result = login.reset_password(
        session=_FakeSession(),
        body=SimpleNamespace(token="ok", new_password="new-pass"),
    )
    assert result.message
    assert active_user.hashed_password == "reset-hash"

    assert private.private_route() == {"message": "Hello world"}
    assert route_utils.health_check() is True