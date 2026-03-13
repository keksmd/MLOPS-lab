
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from app.api.routes import items, login, private, users, utils as utils_routes
from app.models import (
    Item,
    ItemCreate,
    ItemUpdate,
    NewPassword,
    UpdatePassword,
    User,
    UserCreate,
    UserRegister,
    UserUpdate,
    UserUpdateMe,
)


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
        self.get_map: dict[tuple[object, object], object | None] = {}

    def exec(self, statement: object) -> _ExecResult:
        if self.exec_results:
            return self.exec_results.pop(0)
        return _ExecResult()

    def get(self, model: object, key: object) -> object | None:
        return self.get_map.get((model, key))

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def delete(self, obj: object) -> None:
        self.deleted.append(obj)

    def commit(self) -> None:
        self.commits += 1

    def refresh(self, obj: object) -> None:
        self.refreshed.append(obj)


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
        full_name="User",
        is_superuser=is_superuser,
        is_active=is_active,
    )


def _make_item(owner_id) -> Item:
    return Item(
        id=uuid4(),
        title="Item",
        description="Desc",
        owner_id=owner_id,
    )


@dataclass
class _EmailData:
    subject: str
    html_content: str


def test_items_routes() -> None:
    admin = _make_user(is_superuser=True)
    normal = _make_user()
    own_item = _make_item(normal.id)
    other_item = _make_item(uuid4())

    session = _FakeSession(exec_results=[_ExecResult(one=3), _ExecResult(all_values=[own_item])])
    result = items.read_items(session=session, current_user=admin, skip=0, limit=10)
    assert result.count == 3
    assert len(result.data) == 1
    assert result.data[0].id == own_item.id
    assert result.data[0].owner_id == own_item.owner_id

    session = _FakeSession(exec_results=[_ExecResult(one=1), _ExecResult(all_values=[own_item])])
    result = items.read_items(session=session, current_user=normal, skip=0, limit=10)
    assert result.count == 1
    assert len(result.data) == 1
    assert result.data[0].id == own_item.id

    session = _FakeSession()
    with pytest.raises(Exception) as exc_info:
        items.read_item(session=session, current_user=normal, id=uuid4())
    assert exc_info.value.status_code == 404

    session.get_map[(Item, own_item.id)] = other_item
    with pytest.raises(Exception) as exc_info:
        items.read_item(session=session, current_user=normal, id=own_item.id)
    assert exc_info.value.status_code == 403

    session.get_map[(Item, own_item.id)] = own_item
    assert items.read_item(session=session, current_user=normal, id=own_item.id) is own_item

    created = items.create_item(
        session=session,
        current_user=normal,
        item_in=ItemCreate(title="Created", description="desc"),
    )
    assert created.owner_id == normal.id

    session = _FakeSession()
    with pytest.raises(Exception) as exc_info:
        items.update_item(
            session=session,
            current_user=normal,
            id=uuid4(),
            item_in=ItemUpdate(title="Updated"),
        )
    assert exc_info.value.status_code == 404

    session.get_map[(Item, own_item.id)] = other_item
    with pytest.raises(Exception) as exc_info:
        items.update_item(
            session=session,
            current_user=normal,
            id=own_item.id,
            item_in=ItemUpdate(title="Updated"),
        )
    assert exc_info.value.status_code == 403

    session.get_map[(Item, own_item.id)] = own_item
    updated = items.update_item(
        session=session,
        current_user=normal,
        id=own_item.id,
        item_in=ItemUpdate(title="Updated"),
    )
    assert updated.title == "Updated"

    session = _FakeSession()
    with pytest.raises(Exception):
        items.delete_item(session=session, current_user=normal, id=uuid4())

    session.get_map[(Item, own_item.id)] = other_item
    with pytest.raises(Exception):
        items.delete_item(session=session, current_user=normal, id=own_item.id)

    session.get_map[(Item, own_item.id)] = own_item
    message = items.delete_item(session=session, current_user=normal, id=own_item.id)
    assert message.message == "Item deleted successfully"


def test_users_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(exec_results=[_ExecResult(one=2), _ExecResult(all_values=[_make_user()])])
    read_result = users.read_users(session=session, skip=0, limit=10)
    assert read_result.count == 2

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: _make_user())
    with pytest.raises(Exception) as exc_info:
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
    monkeypatch.setattr(users.settings, "EMAILS_FROM_EMAIL", "demo@example.com")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        users,
        "generate_new_account_email",
        lambda email_to, username, password: _EmailData("subject", "html"),
    )
    monkeypatch.setattr(
        users,
        "send_email",
        lambda email_to, subject, html_content: sent.append((email_to, subject)),
    )
    assert (
        users.create_user(
            session=session,
            user_in=UserCreate(
                email="new@example.com",
                password="password123",
                full_name="New",
            ),
        )
        is created_user
    )
    assert sent == [("new@example.com", "subject")]

    current_user = _make_user()
    duplicate_user = _make_user(email="dup@example.com")
    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: duplicate_user)
    with pytest.raises(Exception) as exc_info:
        users.update_user_me(
            session=session,
            user_in=UserUpdateMe(email="dup@example.com"),
            current_user=current_user,
        )
    assert exc_info.value.status_code == 409

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: None)
    updated_me = users.update_user_me(
        session=session,
        user_in=UserUpdateMe(full_name="Changed"),
        current_user=current_user,
    )
    assert updated_me.full_name == "Changed"

    monkeypatch.setattr(users, "verify_password", lambda plain, hashed: (False, None))
    with pytest.raises(Exception) as exc_info:
        users.update_password_me(
            session=session,
            body=UpdatePassword(current_password="password123", new_password="newpassword123"),
            current_user=current_user,
        )
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(users, "verify_password", lambda plain, hashed: (True, None))
    with pytest.raises(Exception) as exc_info:
        users.update_password_me(
            session=session,
            body=UpdatePassword(current_password="password123", new_password="password123"),
            current_user=current_user,
        )
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(users, "get_password_hash", lambda password: "hashed:new")
    message = users.update_password_me(
        session=session,
        body=UpdatePassword(current_password="password123", new_password="newpassword123"),
        current_user=current_user,
    )
    assert message.message == "Password updated successfully"

    assert users.read_user_me(current_user) is current_user

    with pytest.raises(Exception) as exc_info:
        users.delete_user_me(session=session, current_user=_make_user(is_superuser=True))
    assert exc_info.value.status_code == 403

    deleted_message = users.delete_user_me(session=session, current_user=current_user)
    assert deleted_message.message == "User deleted successfully"

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: _make_user())
    with pytest.raises(Exception):
        users.register_user(
            session=session,
            user_in=UserRegister(
                email="exists@example.com",
                password="password123",
                full_name="Exists",
            ),
        )

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: None)
    monkeypatch.setattr(users.crud, "create_user", lambda session, user_create: created_user)
    assert (
        users.register_user(
            session=session,
            user_in=UserRegister(
                email="new@example.com",
                password="password123",
                full_name="New",
            ),
        )
        is created_user
    )

    own = _make_user()
    assert users.read_user_by_id(own.id, session=session, current_user=own) is own

    with pytest.raises(Exception) as exc_info:
        users.read_user_by_id(uuid4(), session=session, current_user=_make_user())
    assert exc_info.value.status_code == 403

    other_user = _make_user(email="other@example.com")
    session.get_map[(User, other_user.id)] = None
    with pytest.raises(Exception) as exc_info:
        users.read_user_by_id(other_user.id, session=session, current_user=_make_user(is_superuser=True))
    assert exc_info.value.status_code == 404

    session.get_map[(User, other_user.id)] = other_user
    assert users.read_user_by_id(other_user.id, session=session, current_user=_make_user(is_superuser=True)) is other_user

    session.get_map[(User, other_user.id)] = None
    with pytest.raises(Exception) as exc_info:
        users.update_user(
            session=session,
            user_id=other_user.id,
            user_in=UserUpdate(full_name="Updated"),
        )
    assert exc_info.value.status_code == 404

    session.get_map[(User, other_user.id)] = other_user
    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: _make_user(email="dup2@example.com"))
    with pytest.raises(Exception) as exc_info:
        users.update_user(
            session=session,
            user_id=other_user.id,
            user_in=UserUpdate(email="dup2@example.com"),
        )
    assert exc_info.value.status_code == 409

    monkeypatch.setattr(users.crud, "get_user_by_email", lambda session, email: None)
    monkeypatch.setattr(users.crud, "update_user", lambda session, db_user, user_in: db_user)
    assert users.update_user(
        session=session,
        user_id=other_user.id,
        user_in=UserUpdate(full_name="Updated"),
    ) is other_user

    session.get_map[(User, other_user.id)] = None
    with pytest.raises(Exception) as exc_info:
        users.delete_user(
            session=session,
            current_user=_make_user(is_superuser=True),
            user_id=other_user.id,
        )
    assert exc_info.value.status_code == 404

    session.get_map[(User, other_user.id)] = _make_user(is_superuser=True)
    current_superuser = session.get_map[(User, other_user.id)]
    with pytest.raises(Exception) as exc_info:
        users.delete_user(
            session=session,
            current_user=current_superuser,
            user_id=current_superuser.id,
        )
    assert exc_info.value.status_code == 403

    session.get_map[(User, other_user.id)] = other_user
    message = users.delete_user(
        session=session,
        current_user=_make_user(is_superuser=True),
        user_id=other_user.id,
    )
    assert message.message == "User deleted successfully"


def test_login_private_and_utils_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession()
    active_user = _make_user(is_superuser=True)

    monkeypatch.setattr(login.crud, "authenticate", lambda session, email, password: None)
    form = type("Form", (), {"username": "user@example.com", "password": "password123"})()
    with pytest.raises(Exception) as exc_info:
        login.login_access_token(session=session, form_data=form)
    assert exc_info.value.status_code == 400

    inactive_user = _make_user(is_active=False)
    monkeypatch.setattr(login.crud, "authenticate", lambda session, email, password: inactive_user)
    with pytest.raises(Exception) as exc_info:
        login.login_access_token(session=session, form_data=form)
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login.crud, "authenticate", lambda session, email, password: active_user)
    monkeypatch.setattr(login.security, "create_access_token", lambda subject, expires_delta: "token")
    token = login.login_access_token(session=session, form_data=form)
    assert token.access_token == "token"

    assert login.test_token(active_user) is active_user

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: active_user)
    monkeypatch.setattr(login, "generate_password_reset_token", lambda email: "reset-token")
    monkeypatch.setattr(
        login,
        "generate_reset_password_email",
        lambda email_to, email, token: _EmailData("subject", "html"),
    )
    monkeypatch.setattr(
        login,
        "send_email",
        lambda email_to, subject, html_content: sent.append((email_to, subject)),
    )
    message = login.recover_password("user@example.com", session=session)
    assert "If that email is registered" in message.message
    assert sent == [(active_user.email, "subject")]

    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: None)
    message = login.recover_password("missing@example.com", session=session)
    assert "If that email is registered" in message.message

    monkeypatch.setattr(login, "verify_password_reset_token", lambda token: None)
    with pytest.raises(Exception) as exc_info:
        login.reset_password(session=session, body=NewPassword(token="bad", new_password="password123"))
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login, "verify_password_reset_token", lambda token: "user@example.com")
    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: None)
    with pytest.raises(Exception) as exc_info:
        login.reset_password(session=session, body=NewPassword(token="bad", new_password="password123"))
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: inactive_user)
    with pytest.raises(Exception) as exc_info:
        login.reset_password(session=session, body=NewPassword(token="ok", new_password="password123"))
    assert exc_info.value.status_code == 400

    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: active_user)
    monkeypatch.setattr(login.crud, "update_user", lambda session, db_user, user_in: db_user)
    message = login.reset_password(session=session, body=NewPassword(token="ok", new_password="password123"))
    assert message.message == "Password updated successfully"

    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: None)
    with pytest.raises(Exception) as exc_info:
        login.recover_password_html_content("missing@example.com", session=session)
    assert exc_info.value.status_code == 404

    monkeypatch.setattr(login.crud, "get_user_by_email", lambda session, email: active_user)
    monkeypatch.setattr(login, "generate_password_reset_token", lambda email: "reset-token")
    monkeypatch.setattr(
        login,
        "generate_reset_password_email",
        lambda email_to, email, token: _EmailData("subject", "<b>html</b>"),
    )
    html_response = login.recover_password_html_content("user@example.com", session=session)
    assert "html" in html_response.body.decode()

    private_user = private.PrivateUserCreate(
        email="private@example.com",
        password="password123",
        full_name="Private",
    )
    monkeypatch.setattr(private, "get_password_hash", lambda password: "hashed-private")
    created = private.create_user(private_user, session=session)
    assert created.hashed_password == "hashed-private"

    monkeypatch.setattr(utils_routes, "generate_test_email", lambda email_to: _EmailData("subject", "html"))
    utility_sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        utils_routes,
        "send_email",
        lambda email_to, subject, html_content: utility_sent.append((email_to, subject)),
    )
    message = utils_routes.test_email("user@example.com")
    assert message.message == "Test email sent"
    assert utility_sent == [("user@example.com", "subject")]
    assert utils_routes.health_check.__name__ == "health_check"
