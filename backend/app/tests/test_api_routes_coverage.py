from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from app.api.routes import items, login, private, users, utils as route_utils
from app.models import Item, ItemCreate, ItemUpdate, User, UserCreate, UserRegister, UserUpdate
from app.schemas import (
    ItemPublic,
    Message,
    NewPassword,
    Token,
    UpdatePassword,
    UserPublic,
    UsersPublic,
)


def _make_user(
    *,
    user_id: UUID | None = None,
    email: str = "user@example.com",
    is_superuser: bool = False,
    is_active: bool = True,
    full_name: str = "User",
) -> User:
    return User(
        id=user_id or uuid4(),
        email=email,
        is_superuser=is_superuser,
        is_active=is_active,
        full_name=full_name,
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


class _ExecResult:
    def __init__(
        self,
        *,
        one: Any | None = None,
        first: Any | None = None,
        all_values: list[Any] | None = None,
    ) -> None:
        self._one = one
        self._first = first
        self._all_values = all_values or []

    def one(self) -> Any:
        return self._one

    def first(self) -> Any:
        return self._first

    def all(self) -> list[Any]:
        return self._all_values


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
        self.refreshed: list[Any] = []

    def exec(self, statement: Any) -> _ExecResult:
        if self.exec_results:
            return self.exec_results.pop(0)
        return _ExecResult()

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    def delete(self, obj: Any) -> None:
        self.deleted.append(obj)

    def commit(self) -> None:
        self.commits += 1

    def refresh(self, obj: Any) -> None:
        self.refreshed.append(obj)

    def get(self, model: Any, obj_id: UUID) -> Any:
        return self.get_result


def test_items_routes() -> None:
    admin = _make_user(is_superuser=True)
    normal = _make_user()
    own_item = _make_item(normal.id)
    other_item = _make_item(uuid4())

    session = _FakeSession(exec_results=[_ExecResult(one=3), _ExecResult(all_values=[own_item])])
    result = items.read_items(session=session, current_user=admin, skip=0, limit=10)
    assert result.count == 3
    assert len(result.data) == 1
    assert isinstance(result.data[0], ItemPublic)
    assert result.data[0].id == own_item.id

    session = _FakeSession(exec_results=[_ExecResult(one=1), _ExecResult(all_values=[own_item])])
    result = items.read_items(session=session, current_user=normal, skip=0, limit=10)
    assert result.count == 1
    assert result.data[0].owner_id == normal.id

    session = _FakeSession(get_result=own_item)
    result = items.read_item_by_id(item_id=own_item.id, session=session, current_user=normal)
    assert isinstance(result, Item)
    assert result.id == own_item.id

    session = _FakeSession(get_result=None)
    with pytest.raises(HTTPException) as exc_info:
        items.read_item_by_id(item_id=uuid4(), session=session, current_user=normal)
    assert exc_info.value.status_code == 404

    session = _FakeSession(get_result=other_item)
    with pytest.raises(HTTPException) as exc_info:
        items.read_item_by_id(item_id=other_item.id, session=session, current_user=normal)
    assert exc_info.value.status_code == 400

    create_session = _FakeSession()
    created = items.create_item(
        *,
        session=create_session,
        current_user=normal,
        item_in=ItemCreate(title="New", description="Created"),
    )
    assert created.owner_id == normal.id
    assert create_session.commits == 1
    assert len(create_session.added) == 1

    update_session = _FakeSession(get_result=own_item)
    updated = items.update_item(
        item_id=own_item.id,
        session=update_session,
        current_user=normal,
        item_in=ItemUpdate(title="Updated"),
    )
    assert updated.title == "Updated"
    assert update_session.commits == 1

    forbidden_update_session = _FakeSession(get_result=other_item)
    with pytest.raises(HTTPException) as exc_info:
        items.update_item(
            item_id=other_item.id,
            session=forbidden_update_session,
            current_user=normal,
            item_in=ItemUpdate(title="Nope"),
        )
    assert exc_info.value.status_code == 400

    delete_session = _FakeSession(get_result=own_item)
    deleted = items.delete_item(item_id=own_item.id, session=delete_session, current_user=normal)
    assert isinstance(deleted, Message)
    assert delete_session.deleted == [own_item]
    assert delete_session.commits == 1


def test_users_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(exec_results=[_ExecResult(one=2), _ExecResult(all_values=[_make_user()])])
    read_result = users.read_users(session=session, skip=0, limit=10)
    assert isinstance(read_result, UsersPublic)
    assert read_result.count == 2

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
    monkeypatch.setattr(users, "generate_new_account_email", lambda **kwargs: {"subject": "Welcome", "html": "<p>Hi</p>"})
    sent: dict[str, Any] = {}
    monkeypatch.setattr(users, "send_email", lambda **kwargs: sent.update(kwargs))

    result = users.create_user(
        session=session,
        user_in=UserCreate(
            email="new@example.com",
            password="password123",
            full_name="New User",
        ),
    )
    assert isinstance(result, User)
    assert sent["email_to"] == "new@example.com"

    current_user = _make_user()
    same_user_session = _FakeSession(get_result=current_user)
    result = users.read_user_by_id(
        user_id=current_user.id,
        session=same_user_session,
        current_user=current_user,
    )
    assert result == current_user

    other_user = _make_user()
    forbidden_session = _FakeSession(get_result=other_user)
    with pytest.raises(HTTPException) as exc_info:
        users.read_user_by_id(
            user_id=other_user.id,
            session=forbidden_session,
            current_user=current_user,
        )
    assert exc_info.value.status_code == 403

    admin = _make_user(is_superuser=True)
    admin_session = _FakeSession(get_result=other_user)
    result = users.read_user_by_id(
        user_id=other_user.id,
        session=admin_session,
        current_user=admin,
    )
    assert result == other_user

    with pytest.raises(HTTPException) as exc_info:
        users.register_user(
            session=session,
            user_in=UserRegister(
                email="new@example.com",
                password="password123",
                full_name="Dup",
            ),
        )
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: None)
    monkeypatch.setattr(users.crud, "create_user", lambda session, user_create: created_user)
    registered = users.register_user(
        session=session,
        user_in=UserRegister(
            email="new2@example.com",
            password="password123",
            full_name="Reg",
        ),
    )
    assert isinstance(registered, UserPublic)
    assert registered.email == "new@example.com"

    me = users.read_user_me(current_user=current_user)
    assert me.id == current_user.id

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: _make_user(email=email))
    with pytest.raises(HTTPException) as exc_info:
        users.update_user_me(
            session=session,
            current_user=current_user,
            user_in=UserUpdate(email="taken@example.com"),
        )
    assert exc_info.value.status_code == 409

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: None)
    monkeypatch.setattr(
        users.crud,
        "update_user",
        lambda session, db_user, user_in: db_user.sqlmodel_update(user_in.model_dump(exclude_unset=True)) or db_user,
    )
    updated_me = users.update_user_me(
        session=session,
        current_user=current_user,
        user_in=UserUpdate(full_name="Updated Name"),
    )
    assert updated_me.full_name == "Updated Name"

    monkeypatch.setattr(users, "verify_password", lambda plain_password, hashed_password: False)
    with pytest.raises(HTTPException) as exc_info:
        users.update_password_me(
            session=session,
            current_user=current_user,
            body=UpdatePassword(current_password="bad", new_password="new-password"),
        )
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(users, "verify_password", lambda plain_password, hashed_password: True)
    with pytest.raises(HTTPException) as exc_info:
        users.update_password_me(
            session=session,
            current_user=current_user,
            body=UpdatePassword(current_password="password123", new_password="password123"),
        )
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(users, "get_password_hash", lambda password: f"hashed:{password}")
    password_message = users.update_password_me(
        session=session,
        current_user=current_user,
        body=UpdatePassword(current_password="password123", new_password="new-password"),
    )
    assert isinstance(password_message, Message)
    assert session.commits >= 1

    with pytest.raises(HTTPException) as exc_info:
        users.delete_user_me(session=session, current_user=admin)
    assert exc_info.value.status_code == 403

    delete_me_session = _FakeSession()
    delete_message = users.delete_user_me(session=delete_me_session, current_user=current_user)
    assert isinstance(delete_message, Message)
    assert delete_me_session.deleted == [current_user]

    missing_session = _FakeSession(get_result=None)
    with pytest.raises(HTTPException) as exc_info:
        users.delete_user(session=missing_session, current_user=admin, user_id=uuid4())
    assert exc_info.value.status_code == 404

    with pytest.raises(HTTPException) as exc_info:
        users.delete_user(session=_FakeSession(get_result=admin), current_user=admin, user_id=admin.id)
    assert exc_info.value.status_code == 403

    target_user = _make_user()
    delete_user_session = _FakeSession(get_result=target_user)
    delete_message = users.delete_user(session=delete_user_session, current_user=admin, user_id=target_user.id)
    assert isinstance(delete_message, Message)
    assert delete_user_session.deleted == [target_user]


def test_login_private_and_utils_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    db_session = _FakeSession()
    active_user = _make_user()
    inactive_user = _make_user(is_active=False)

    monkeypatch.setattr(login.crud, "authenticate", lambda session, email, password: None)
    with pytest.raises(HTTPException) as exc_info:
        login.login_access_token(db_session=db_session, form_data=SimpleNamespace(username="u", password="p"))
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login.crud, "authenticate", lambda session, email, password: inactive_user)
    with pytest.raises(HTTPException) as exc_info:
        login.login_access_token(db_session=db_session, form_data=SimpleNamespace(username="u", password="p"))
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login.crud, "authenticate", lambda session, email, password: active_user)
    monkeypatch.setattr(login.security, "create_access_token", lambda sub, expires_delta: "token")
    token = login.login_access_token(db_session=db_session, form_data=SimpleNamespace(username="u", password="p"))
    assert isinstance(token, Token)
    assert token.access_token == "token"

    test_token = login.test_token(current_user=active_user)
    assert test_token.email == active_user.email

    private_result = private.private(current_user=active_user)
    assert private_result == {"message": "Hello World"}

    health_result = route_utils.health_check()
    assert health_result is True

    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: None)
    message = login.recover_password(email="missing@example.com", db_session=db_session)
    assert isinstance(message, Message)

    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: inactive_user)
    with pytest.raises(HTTPException) as exc_info:
        login.recover_password(email="inactive@example.com", db_session=db_session)
    assert exc_info.value.status_code == 400

    sent: dict[str, Any] = {}
    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: active_user)
    monkeypatch.setattr(login, "generate_password_reset_token", lambda email: "reset-token")
    monkeypatch.setattr(login, "generate_reset_password_email", lambda **kwargs: {"subject": "Reset", "html": "<p>Reset</p>"})
    monkeypatch.setattr(login, "send_email", lambda **kwargs: sent.update(kwargs))
    message = login.recover_password(email=active_user.email, db_session=db_session)
    assert isinstance(message, Message)
    assert sent["email_to"] == active_user.email

    monkeypatch.setattr(login, "verify_password_reset_token", lambda token: None)
    with pytest.raises(HTTPException) as exc_info:
        login.reset_password(db_session=db_session, body=NewPassword(token="bad", new_password="pass"))
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login, "verify_password_reset_token", lambda token: active_user.email)
    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: None)
    with pytest.raises(HTTPException) as exc_info:
        login.reset_password(db_session=db_session, body=NewPassword(token="bad", new_password="pass"))
    assert exc_info.value.status_code == 404