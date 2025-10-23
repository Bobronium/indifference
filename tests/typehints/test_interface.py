"""Type assertion tests for indifference public interfaces."""

from typing import Any
from typing import Literal

import pytest_assert_type
from typing_extensions import assert_type

from indifference import Change
from indifference import Changed
from indifference import ChangeKind
from indifference import Diff
from indifference import Path
from indifference import at
from indifference import diff

obj1 = {"a": 1, "b": 2}
obj2 = {"a": 10, "b": 20}


@pytest_assert_type.check
def test_types() -> None:
    # diff function
    d1 = diff(obj1, obj2)
    d2 = diff(obj1, obj2, ignore_atomic_identity=True)
    assert_type(d1, Diff)
    assert_type(d2, Diff)

    # Diff operations
    assert_type(d1.changes, tuple[Change, ...])
    assert_type(d1 - d2, Diff)
    assert_type(d1 - [at["a"] > 10], Diff)

    # Path access
    assert_type(at, Path)
    assert_type(at.name, Path)
    assert_type(at[0], Path)
    assert_type(at.users[0].profile["email"], Path)

    # Path to Change
    assert_type(at.name > "Bob", Change)
    assert_type(at.name < "Alice", Change)
    assert_type(-at.field, Change)
    assert_type(+at.field, Change)
    assert_type(at.cache & Changed.identity, Change)

    # Change properties
    change = at.name > "Bob"
    assert_type(change.path, str)
    assert_type(change.kind, ChangeKind)

    explicit = Change(
        path="test", kind=ChangeKind.MODIFIED, before=42, after="value", detail="info"
    )
    assert_type(explicit.before, Any)
    assert_type(explicit.after, Any)
    assert_type(explicit.detail, str)

    assert_type(Changed.identity, Literal[Changed.identity])
    assert_type(Changed.type, Literal[Changed.type])
    assert_type(Changed.structure, Literal[Changed.structure])
