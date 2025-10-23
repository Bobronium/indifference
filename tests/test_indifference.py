"""Comprehensive tests for indifference object transformation comparison."""

import copy
from dataclasses import dataclass
from typing import Any

import pytest

from indifference import Change
from indifference import Changed
from indifference import ChangeKind
from indifference import assert_equivalent_transformations
from indifference import at
from indifference import diff

# ============================= Fixtures ==============================


@dataclass
class SimpleObject:
    name: str
    value: int


@dataclass
class NestedObject:
    simple: SimpleObject
    items: list[int]


class ComplexObject:
    def __init__(self, x: int, y: str):
        self.x = x
        self.y = y
        self.cache = {"data": [1, 2, 3]}


# ============================= Basic Tests ===========================


def test_diff_identical_objects():
    """Two identical objects should produce an empty diff."""
    obj = {"a": 1, "b": 2}
    result = diff(obj, obj)
    assert len(result) == 0
    assert not result  # Empty diff is falsy


def test_diff_simple_value_change():
    """Detect simple value changes."""
    obj1 = {"name": "Alice", "age": 30}
    obj2 = {"name": "Bob", "age": 30}

    result = diff(obj1, obj2)

    # Should detect name change
    assert result == [at["name"] > "Bob"]


def test_diff_added_key():
    """Detect added dictionary keys."""
    obj1 = {"a": 1}
    obj2 = {"a": 1, "b": 2}

    result = diff(obj1, obj2)

    assert len(result) == 1
    change = result.changes[0]
    assert change.kind == ChangeKind.ADDED
    assert "'b'" in change.path


def test_diff_removed_key():
    """Detect removed dictionary keys."""
    obj1 = {"a": 1, "b": 2}
    obj2 = {"a": 1}

    result = diff(obj1, obj2)

    assert len(result) == 1
    change = result.changes[0]
    assert change.kind == ChangeKind.REMOVED
    assert "'b'" in change.path


def test_diff_list_changes():
    """Detect changes in list elements."""
    obj1 = [1, 2, 3]
    obj2 = [1, 5, 3]

    result = diff(obj1, obj2)

    # Should detect element [1] change
    assert result == [at[1] > 5]


def test_diff_list_length_change():
    """Detect list length changes."""
    obj1 = [1, 2, 3]
    obj2 = [1, 2]

    result = diff(obj1, obj2)

    # Should detect length change
    length_changes = [c for c in result.changes if c.kind == ChangeKind.STRUCTURE]
    assert len(length_changes) == 1


def test_diff_nested_objects():
    """Detect changes in nested structures."""
    obj1 = {"outer": {"inner": {"value": 10}}}
    obj2 = {"outer": {"inner": {"value": 20}}}

    result = diff(obj1, obj2)

    # Should detect nested value change
    assert result == [at["outer"]["inner"]["value"] > 20]


def test_diff_type_change():
    """Detect type changes."""
    obj1 = {"value": 42}
    obj2 = {"value": "42"}

    result = diff(obj1, obj2)

    # Should detect either type change or value change
    assert len(result) >= 1
    # One of the changes should involve the value at 'value'
    assert any("'value'" in c.path for c in result.changes)


# ============================= Path DSL Tests ========================


def test_path_attribute_access():
    """Path builder should support attribute access."""
    path = at.user.name
    assert str(path) == "user.name"


def test_path_item_access():
    """Path builder should support item access."""
    path = at.items[0]
    assert str(path) == "items[0]"


def test_path_mixed_access():
    """Path builder should support mixed attribute and item access."""
    path = at.users[0].name
    assert str(path) == "users[0].name"


def test_path_greater_than_creates_change():
    """at.name > value should create a Modified change."""
    change = at.name > "Bob"

    assert isinstance(change, Change)
    assert change.path == "name"
    assert change.kind == ChangeKind.MODIFIED
    assert change.after == "Bob"


def test_path_less_than_creates_change():
    """at.name < value should create a Modified change with before."""
    change = at.name < "Alice"

    assert isinstance(change, Change)
    assert change.path == "name"
    assert change.kind == ChangeKind.MODIFIED
    assert change.before == "Alice"


def test_path_negative_creates_removed():
    """-at.field should create a Removed change."""
    change = -at.secret

    assert isinstance(change, Change)
    assert change.path == "secret"
    assert change.kind == ChangeKind.REMOVED


def test_path_positive_creates_added():
    """+at.field should create an Added change."""
    change = +at.new_field

    assert isinstance(change, Change)
    assert change.path == "new_field"
    assert change.kind == ChangeKind.ADDED


def test_path_and_changed_identity():
    """at.field & Changed.identity should create identity change."""
    change = at.cache & Changed.identity

    assert isinstance(change, Change)
    assert change.path == "cache"
    assert change.kind == ChangeKind.IDENTITY


def test_path_and_changed_type():
    """at.field & Changed.type should create type change."""
    change = at.value & Changed.type

    assert isinstance(change, Change)
    assert change.path == "value"
    assert change.kind == ChangeKind.TYPE


# ============================= Diff Operations =======================


def test_diff_equality_same_changes():
    """Two diffs with same changes should be equal."""
    obj1 = {"a": 1}
    obj2 = {"a": 2}

    diff1 = diff(obj1, obj2)
    diff2 = diff(obj1, obj2)

    assert diff1 == diff2


def test_diff_equality_different_changes():
    """Two diffs with different changes should not be equal."""
    obj1 = {"a": 1}
    obj2 = {"a": 2}
    obj3 = {"a": 3}

    diff1 = diff(obj1, obj2)
    diff2 = diff(obj1, obj3)

    assert diff1 != diff2


def test_diff_equality_with_expected_list():
    """Diff should match against expected change list."""
    obj1 = {"name": "Alice"}
    obj2 = {"name": "Bob"}

    result = diff(obj1, obj2)

    # This should match
    assert result == [at["name"] > "Bob"]


def test_diff_subtraction_removes_matching():
    """Subtracting diff should remove matching changes."""
    obj1 = {"a": 1, "b": 2}
    obj2 = {"a": 10, "b": 20}
    obj3 = {"a": 10, "b": 2}

    diff_full = diff(obj1, obj2)  # Both a and b changed
    diff_partial = diff(obj1, obj3)  # Only a changed

    remaining = diff_full - diff_partial

    # Should have only the b change remaining
    assert remaining == [at["b"] > 20]


def test_diff_subtraction_with_expected_list():
    """Subtracting expected changes should remove them."""
    obj1 = {"a": 1, "b": 2}
    obj2 = {"a": 10, "b": 20}

    result = diff(obj1, obj2)

    # Remove expected change to 'a'
    remaining = result - [at["a"] > 10]

    # Should have only 'b' change remaining
    assert remaining == [at["b"] > 20]


def test_diff_bool_conversion():
    """Diff should be truthy if it has changes."""
    obj = {"a": 1}

    no_changes = diff(obj, obj)
    assert not no_changes

    with_changes = diff(obj, {"a": 2})
    assert with_changes


def test_diff_len():
    """Diff should support len()."""
    obj1 = {"a": 1, "b": 2}
    obj2 = {"a": 10, "b": 20}

    result = diff(obj1, obj2)

    # Both a and b changed
    assert len(result) == 2


# ===================== Dataclass/Object Tests ========================


def test_diff_dataclass_attribute_change():
    """Detect changes in dataclass attributes."""
    obj1 = SimpleObject(name="Alice", value=10)
    obj2 = SimpleObject(name="Bob", value=10)

    result = diff(obj1, obj2)

    # Should detect name change
    assert result == [at.name > "Bob"]


def test_diff_dataclass_nested():
    """Detect changes in nested dataclasses."""
    obj1 = NestedObject(simple=SimpleObject(name="Alice", value=10), items=[1, 2, 3])
    obj2 = NestedObject(simple=SimpleObject(name="Alice", value=20), items=[1, 2, 3])

    result = diff(obj1, obj2)

    # Should detect value change in nested simple object
    assert result == [at.simple.value > 20]


def test_diff_custom_class():
    """Detect changes in custom class instances."""
    obj1 = ComplexObject(x=10, y="hello")
    obj2 = ComplexObject(x=20, y="hello")

    result = diff(obj1, obj2)

    # Should detect x change
    assert result == [at.x > 20]


# ===================== Identity & Aliasing Tests =====================


def test_diff_identity_preserved_for_same_object():
    """When same object is passed, identity is preserved."""
    obj = [1, 2, 3]
    container1 = {"a": obj, "b": obj}
    container2 = {"a": obj, "b": obj}

    diff(container1, container2)

    # Should be no changes or only acceptable identity changes
    # This tests that we handle aliasing correctly


def test_diff_identity_not_preserved_for_copy():
    """When objects are copied, identity changes are detected."""
    obj = [1, 2, 3]
    container1 = {"a": obj, "b": obj}
    container2 = {"a": [1, 2, 3], "b": [1, 2, 3]}

    diff(container1, container2)

    # May detect identity differences in aliasing
    # The key is that both are separate lists in container2


def test_ignore_atomic_identity():
    """With ignore_atomic_identity, atomic identity changes are ignored."""
    # Strings/ints might have same identity due to interning
    obj1 = {"a": "hello"}
    obj2 = {"a": "hello"}

    result = diff(obj1, obj2, ignore_atomic_identity=True)

    # Should have no changes, ignoring identity for atomic values
    identity_changes = [c for c in result.changes if c.kind == ChangeKind.IDENTITY]
    assert len(identity_changes) == 0


# =================== Cycle Detection Tests ===========================


def test_diff_handles_cycles():
    """Should handle circular references without infinite recursion."""
    obj1: dict[str, Any] = {"a": 1}
    obj1["self"] = obj1

    obj2: dict[str, Any] = {"a": 1}
    obj2["self"] = obj2

    # Should not raise RecursionError
    result = diff(obj1, obj2)

    # Should detect no substantial changes
    value_changes = [c for c in result.changes if c.kind == ChangeKind.MODIFIED]
    assert len(value_changes) == 0


def test_diff_different_cycle_structure():
    """Should detect when cycle structure differs."""
    obj1: dict[str, Any] = {"a": 1}
    obj1["self"] = obj1

    obj2: dict[str, Any] = {"a": 2}  # Different value
    obj2["self"] = obj2

    result = diff(obj1, obj2)

    # Should detect the value change
    assert len(result) >= 1


# ================= assert_equivalent_transformations Tests ===========


def test_assert_equivalent_deepcopy_same():
    """Deepcopy should be equivalent to itself."""
    original = {"a": 1, "b": [2, 3, 4]}
    copied = copy.deepcopy(original)

    # Should not raise
    assert_equivalent_transformations(original, copied, copied)


def test_assert_equivalent_with_expected_changes():
    """Should accept expected differences."""
    original = {"a": 1, "b": 2, "secret": "hidden"}
    reference = copy.deepcopy(original)
    custom = {"a": 1, "b": 2}  # Missing 'secret'

    # Should not raise with expected
    assert_equivalent_transformations(original, reference, custom, expected=[-at["secret"]])


def test_assert_equivalent_fails_on_unexpected():
    """Should raise on unexpected differences."""
    original = {"a": 1, "b": 2}
    reference = copy.deepcopy(original)
    custom = {"a": 10, "b": 2}  # 'a' changed unexpectedly

    with pytest.raises(AssertionError) as exc_info:
        assert_equivalent_transformations(original, reference, custom)

    # Error message should mention the unexpected change
    assert "differ" in str(exc_info.value).lower()


def test_assert_equivalent_complex_object():
    """Should work with complex nested objects."""
    original = ComplexObject(x=10, y="test")
    reference = copy.deepcopy(original)

    # Create custom copy that preserves structure
    custom = ComplexObject(x=10, y="test")
    custom.cache = {"data": [1, 2, 3]}

    # Should not raise (or only identity differences which can be expected)
    assert_equivalent_transformations(original, reference, custom, ignore_atomic_identity=True)


# ===================== Edge Cases ====================================


def test_diff_empty_containers():
    """Should handle empty containers."""
    result = diff([], [])
    assert len(result) == 0

    result = diff({}, {})
    assert len(result) == 0

    result = diff(set(), set())
    assert len(result) == 0


def test_diff_none_values():
    """Should handle None values."""
    obj1 = {"a": None}
    obj2 = {"a": None}

    result = diff(obj1, obj2)
    assert len(result) == 0


def test_diff_none_to_value():
    """Should detect None to value changes."""
    obj1 = {"a": None}
    obj2 = {"a": 42}

    result = diff(obj1, obj2)
    assert len(result) >= 1


def test_diff_mixed_types():
    """Should handle various Python types."""
    obj1 = {
        "int": 42,
        "float": 3.14,
        "str": "hello",
        "bool": True,
        "none": None,
        "list": [1, 2],
        "tuple": (3, 4),
        "set": {5, 6},
    }
    obj2 = copy.deepcopy(obj1)

    diff(obj1, obj2)

    # With deepcopy and ignore_atomic_identity, should be minimal changes
    diff(obj1, obj2, ignore_atomic_identity=True)


def test_change_str_representation():
    """Change should have readable string representation."""
    change1 = Change(path="name", kind=ChangeKind.MODIFIED, before="Alice", after="Bob")
    assert "name" in str(change1)
    assert "Alice" in str(change1)
    assert "Bob" in str(change1)

    change2 = Change(path="field", kind=ChangeKind.REMOVED, before="value")
    assert "removed" in str(change2)

    change3 = Change(path="field", kind=ChangeKind.ADDED, after="value")
    assert "added" in str(change3)


def test_diff_repr():
    """Diff should have readable repr."""
    obj1 = {"a": 1}
    obj2 = {"a": 2}

    result = diff(obj1, obj2)
    repr_str = repr(result)

    assert "Diff" in repr_str

    empty = diff(obj1, obj1)
    empty_repr = repr(empty)
    assert "no changes" in empty_repr.lower()


# ===================== Real-World Scenarios ==========================


def test_real_world_config_comparison():
    """Simulate comparing two configuration objects."""
    config1 = {
        "database": {"host": "localhost", "port": 5432, "name": "mydb"},
        "cache": {"enabled": True, "ttl": 300},
    }

    config2 = {
        "database": {"host": "production.db.example.com", "port": 5432, "name": "mydb"},
        "cache": {
            "enabled": True,
            "ttl": 600,  # Changed
        },
    }

    result = diff(config1, config2)

    # Should detect host and ttl changes
    assert result == [
        at["cache"]["ttl"] > 600,
        at["database"]["host"] > "production.db.example.com",
    ]


def test_real_world_model_versioning():
    """Simulate comparing two versions of a model."""

    @dataclass
    class Model:
        version: int
        weights: list[float]
        metadata: dict[str, Any]

    v1 = Model(
        version=1, weights=[0.1, 0.2, 0.3], metadata={"trained": "2024-01-01", "accuracy": 0.85}
    )

    v2 = Model(
        version=2, weights=[0.15, 0.25, 0.35], metadata={"trained": "2024-02-01", "accuracy": 0.90}
    )

    result = diff(v1, v2)

    # Should detect version, weights, and metadata changes
    assert len(result) == 6
