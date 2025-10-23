# SPDX-FileCopyrightText: 2023-present Arseny Boykov (Bobronium) <mail@bobronium.me>


"""
Compare stories of how objects transform.

When you copy an object, you create a story: which parts stayed the same object,
which became new objects, what changed, what was preserved. This library lets you
compare those stories.

Usage:
    import pickle

    from copy import deepcopy

    from indifference import diff, at, Changed

    original = ...`
    deepcopy_story = diff(original, deepcopy(original))
    pickle_story = diff(original, pickle.loads(pickle.dumps(original)))

    # They should tell the same story
    assert deepcopy_story == custom_story

    # Or use set operations to compare stories
    differences = deepcopy_story ^ custom_story   # symmetric difference
    common = deepcopy_story & custom_story        # intersection
    unique = deepcopy_story - custom_story        # difference
    all_changes = deepcopy_story | custom_story   # union

    # Check what's different between the stories
    assert differences == [
        at.name > "Bob",              # name changed to Bob
        at.cache & Changed.identity,  # cache has different identity
        -at.secret,                   # secret was removed
    ]
"""

from __future__ import annotations

import contextlib
import math
import re
import types
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from types import EllipsisType
from typing import TYPE_CHECKING
from typing import Any
from typing import overload

if TYPE_CHECKING:
    from typing_extensions import assert_never
else:

    def assert_never(_: Any) -> None: ...


# ================================ Public API =================================


class ChangeKind(Enum):
    """Ways an object can change during transformation."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    IDENTITY = "identity"
    TYPE = "type"
    STRUCTURE = "structure"


@dataclass(frozen=True)
class Change:
    """A single change in an object's story.

    Represents one thing that happened during transformation:
    - A value changed
    - Something was added or removed
    - Identity changed but value stayed same
    - Type changed
    - Structure changed (length, keys, etc.)
    """

    path: str
    kind: ChangeKind
    before: Any = None
    after: Any = None
    detail: str = ""

    def __str__(self) -> str:
        # Clean up path for display - remove extra quotes and brackets for simple keys
        display_path = self.path

        # Handle root path
        if display_path == "." or not display_path:
            display_path = "<root>"
        else:
            # Clean up dict key notation: ['key'] -> key
            display_path = display_path.replace("']['", ".").replace("['", ".").replace("']", "")
            # Clean up list index notation: keep [0] as is but remove leading .
            display_path = display_path.removeprefix(".")

        if self.kind == ChangeKind.ADDED:
            return f"{display_path}: (added) {_repr(self.after)}"
        if self.kind == ChangeKind.REMOVED:
            return f"{display_path}: {_repr(self.before)} (removed)"
        if self.kind == ChangeKind.MODIFIED:
            if self.before is None:
                return f"{display_path}: -> {_repr(self.after)}"
            return f"{display_path}: {_repr(self.before)} -> {_repr(self.after)}"
        if self.kind == ChangeKind.IDENTITY:
            return f"{display_path}: identity differs"
        if self.kind == ChangeKind.TYPE:
            return f"{display_path}: type {_type_name(self.before)} -> {_type_name(self.after)}"
        if self.kind == ChangeKind.STRUCTURE:
            return f"{display_path}: length {self.before} -> {self.after}"
        assert_never(self.kind)
        raise NotImplementedError


class Diff:  # noqa: PLW1641
    """The story of how an object transformed.

    Contains all the changes that happened. Can be compared to other stories
    to see if they're equivalent, or combined using set operations.

    Set operations:
        diff1 - diff2    : Changes in diff1 but not in diff2 (difference)
        diff1 ^ diff2    : Changes that differ between stories (symmetric difference)
        diff1 & diff2    : Changes common to both stories (intersection)
        diff1 | diff2    : All unique changes from both stories (union)
        diff - [changes] : Remove expected changes from diff
    """

    def __init__(self, changes: list[Change]) -> None:
        self.changes: tuple[Change, ...] = tuple(changes)  # Immutable
        self._by_path: dict[str, Change] = {c.path: c for c in changes}

    def __eq__(self, other: object) -> bool:
        """Two stories are equal if they contain the same changes."""
        if isinstance(other, list):
            # Allow comparing to expected change list
            return self._matches_expected(other)
        if not isinstance(other, Diff):
            return False

        # Compare changes by building dicts by path
        if len(self.changes) != len(other.changes):
            return False

        self_by_path = {c.path: c for c in self.changes}
        other_by_path = {c.path: c for c in other.changes}

        if set(self_by_path.keys()) != set(other_by_path.keys()):
            return False

        for path, self_change in self_by_path.items():
            other_change = other_by_path[path]
            # Compare changes exactly (including detail)
            if (
                self_change.kind != other_change.kind
                or not _safe_eq(self_change.before, other_change.before)
                or not _safe_eq(self_change.after, other_change.after)
            ):
                return False

        return True

    @overload
    def __sub__(self, other: Diff) -> Diff: ...

    @overload
    def __sub__(self, other: list[Change]) -> Diff: ...

    def __sub__(self, other: Diff | list[Change]) -> Diff:
        """Set difference: changes in this story but not in the other.

        When subtracting a Diff: returns changes that are in self but not in other.
        When subtracting expected changes: removes those changes from this story.
        """
        if isinstance(other, list):
            # Subtracting expected changes - remove them
            other_paths = {_normalize_expected(c).path for c in other}
            remaining = [c for c in self.changes if c.path not in other_paths]
            return Diff(remaining)

        # Set difference with another Diff
        self_by_path = {c.path: c for c in self.changes}
        other_by_path = {c.path: c for c in other.changes}

        differences = []
        for path, self_change in self_by_path.items():
            other_change = other_by_path.get(path)
            if other_change is None or not _changes_equal(self_change, other_change):
                # In self but not in other, or different
                differences.append(self_change)

        return Diff(differences)

    def __xor__(self, other: Diff) -> Diff:
        """Symmetric difference: changes that differ between the two stories.

        Returns changes that appear in one story but not the other, or appear
        differently in both.
        """
        self_by_path = {c.path: c for c in self.changes}
        other_by_path = {c.path: c for c in other.changes}

        all_paths = set(self_by_path.keys()) | set(other_by_path.keys())
        differences: list[Change] = []

        for path in all_paths:
            self_change = self_by_path.get(path)
            other_change = other_by_path.get(path)

            if self_change is None:
                # Only in other
                assert other_change is not None  # noqa: S101
                differences.append(other_change)
            elif other_change is None:
                # Only in self
                assert self_change is not None  # noqa: S101
                differences.append(self_change)
            elif not _changes_equal(self_change, other_change):
                # Different in both - include both
                differences.append(self_change)
                differences.append(other_change)

        return Diff(differences)

    def __and__(self, other: Diff) -> Diff:
        """Intersection: changes that are the same in both stories."""
        self_by_path = {c.path: c for c in self.changes}
        other_by_path = {c.path: c for c in other.changes}

        common = []
        for path, self_change in self_by_path.items():
            other_change = other_by_path.get(path)
            if other_change and _changes_equal(self_change, other_change):
                common.append(self_change)

        return Diff(common)

    def __or__(self, other: Diff) -> Diff:
        """Union: all unique changes from both stories."""
        all_by_path: dict[str, Change] = {}

        for change in self.changes:
            all_by_path[change.path] = change

        for change in other.changes:
            if change.path not in all_by_path:
                all_by_path[change.path] = change
            elif not _changes_equal(all_by_path[change.path], change):
                # Conflict - keep both? Or keep left? Let's keep left for now
                pass

        return Diff(list(all_by_path.values()))

    def __bool__(self) -> bool:
        """A diff is truthy if it contains any changes."""
        return len(self.changes) > 0

    def __len__(self) -> int:
        return len(self.changes)

    def __repr__(self) -> str:
        if not self.changes:
            return "Diff(no changes)"
        return "Diff(\n  " + "\n  ".join(str(c) for c in self.changes) + "\n)"

    def _matches_expected(self, expected: list[Change | Path]) -> bool:
        """Check if this diff matches a list of expected changes."""
        if len(expected) != len(self.changes):
            return False

        expected_normalized = [_normalize_expected(e) for e in expected]
        expected_by_path = {c.path: c for c in expected_normalized}

        for change in self.changes:
            if change.path not in expected_by_path:
                return False
            expected_change = expected_by_path[change.path]
            if not _changes_match(change, expected_change):
                return False

        return True


class Path:
    """Builds paths through object structure with nice syntax.

    at.name              -> "name"
    at.settings.timeout  -> "settings.timeout"
    at.items[0]          -> "items[0]"

    Supports operators for creating expected changes:
    at.name > "Bob"              -> Change(path="name", after="Bob")
    -at.secret                   -> Change(path="secret", kind=REMOVED)
    at.cache & Changed.identity  -> Change(path="cache", kind=IDENTITY)
    """

    def __init__(self, path: str = "") -> None:
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Path:
        current = object.__getattribute__(self, "_path")
        new_path = f"{current}.{name}" if current else name
        return Path(new_path)

    def __getitem__(self, key: Any) -> Path:
        current = object.__getattribute__(self, "_path")
        if isinstance(key, int):
            return Path(f"{current}[{key}]")
        return Path(f"{current}[{key!r}]")

    def __gt__(self, value: Any) -> Change:
        """at.name > "Bob" means name changed to Bob."""
        path = object.__getattribute__(self, "_path")
        return Change(path=path or ".", kind=ChangeKind.MODIFIED, after=value)

    def __lt__(self, value: Any) -> Change:
        """at.name < "Alice" means name changed from Alice."""
        path = object.__getattribute__(self, "_path")
        return Change(path=path or ".", kind=ChangeKind.MODIFIED, before=value)

    def __neg__(self) -> Change:
        """-at.secret means secret was removed."""
        path = object.__getattribute__(self, "_path")
        return Change(path=path or ".", kind=ChangeKind.REMOVED)

    def __pos__(self) -> Change:
        """+at.new_field means new_field was added."""
        path = object.__getattribute__(self, "_path")
        return Change(path=path or ".", kind=ChangeKind.ADDED)

    def __and__(self, change_spec: Changed) -> Change:
        """at.cache & Changed.identity means cache's identity changed."""
        path = object.__getattribute__(self, "_path")
        return Change(
            path=path or ".", kind=change_spec.value, detail=f"{change_spec.name} changed"
        )

    def __str__(self) -> str:
        return object.__getattribute__(self, "_path") or "."

    def __repr__(self) -> str:
        path = object.__getattribute__(self, "_path")
        return f"at.{path}" if path else "at"


at = Path()


class Changed(Enum):
    identity = ChangeKind.IDENTITY
    type = ChangeKind.TYPE
    structure = ChangeKind.STRUCTURE


def diff(
    original: Any,
    transformed: Any,
    *,
    ignore_atomic_identity: bool = False,
    ignore_code_metadata: bool = False,
) -> Diff:
    """Create a story of how 'transformed' differs from 'original'.

    Traces through the object structure, noting:
    - What values changed
    - What was added or removed
    - What identities were preserved or changed
    - What types changed

    Args:
        original: The object before transformation
        transformed: The object after transformation
        ignore_atomic_identity: If True, don't report identity changes for
                              immutable atomic values (int, str, etc.) when
                              values are equal
        ignore_code_metadata: If True, ignore changes in code object metadata like line numbers

    Returns:
        A Diff containing all observed changes
    """
    tracker = _IdentityTracker()
    story = _observe_transformation(original, transformed, tracker)
    changes = _narrate_changes(story, tracker, "", ignore_atomic_identity, ignore_code_metadata)
    return Diff(changes)


def assert_equivalent_transformations(
    original: Any,
    reference: Any,
    reconstructed: Any,
    *,
    expected: list[Change] | None = None,
    ignore_atomic_identity: bool = False,
    ignore_code_metadata: bool = False,
) -> None:
    """Assert that two transformations tell the same story.

    Compares how 'reconstructed' relates to 'original' versus how 'reference'
    relates to 'original'. Useful for verifying custom copy implementations
    behave like deepcopy.

    Args:
        original: The original object
        reference: Result of reference transformation (e.g., deepcopy)
        reconstructed: Result of your custom transformation
        expected: List of changes you expect to differ
        ignore_atomic_identity: Whether to ignore identity changes in atomics
        ignore_code_metadata: Whether to ignore code object metadata changes

    Raises:
        AssertionError: If transformations differ in unexpected ways

    Example:
        assert_equivalent_transformations(
            original,
            deepcopy(original),
            my_copy(original),
            expected=[
                at.cache & Changed.identity,
                -at.internal_field,
            ]
        )
    """
    reference_story = diff(
        original,
        reference,
        ignore_atomic_identity=ignore_atomic_identity,
        ignore_code_metadata=ignore_code_metadata,
    )
    reconstructed_story = diff(
        original,
        reconstructed,
        ignore_atomic_identity=ignore_atomic_identity,
        ignore_code_metadata=ignore_code_metadata,
    )

    # Compare stories with full context
    comparison = compare_stories(reference_story, reconstructed_story)

    if expected:
        comparison = comparison.remove_expected(expected)

    if comparison:
        msg = _format_story_comparison(comparison)
        raise AssertionError(msg)


# ============================= Implementation ================================


def _repr(obj: Any, max_len: int = 80) -> str:
    """Safe, truncated repr."""
    try:
        r = repr(obj)
        if len(r) > max_len:
            r = r[: max_len - 3] + "..."
        return r  # noqa: TRY300
    except Exception:  # noqa: BLE001
        return f"<{type(obj).__name__}>"


def _type_name(obj: Any) -> str:
    """Get readable type name."""
    try:
        return type(obj).__qualname__
    except Exception:  # noqa: BLE001
        return str(type(obj))


def _is_atomic(obj: Any) -> bool:
    """Check if object is an immutable atomic value."""
    atomic_types = (
        type(None),
        EllipsisType,
        type(NotImplemented),
        int,
        float,
        bool,
        complex,
        bytes,
        str,
        range,
        slice,
        re.Pattern,
        Decimal,
        Fraction,
    )

    return isinstance(obj, atomic_types)


def _safe_eq(a: Any, b: Any) -> bool:
    """Safely check equality."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001
        return False


class _IdentityTracker:
    """Tracks which objects in the transformation share identity.

    When an object appears multiple times (aliasing), we need to know if
    those aliases were preserved in the transformation.
    """

    def __init__(self) -> None:
        self._groups: dict[int, int] = {}
        self._next_group = 1

    def get_group(self, obj: Any) -> int:
        """Get or assign a group ID for this object."""
        obj_id = id(obj)
        if obj_id not in self._groups:
            self._groups[obj_id] = self._next_group
            self._next_group += 1
        return self._groups[obj_id]

    def preserves_aliasing(self, orig_obj: Any, trans_obj: Any) -> bool:
        """Check if aliasing relationship was preserved."""
        return id(orig_obj) == id(trans_obj)


def _observe_transformation(
    original: Any,
    transformed: Any,
    tracker: _IdentityTracker,
    seen: set[tuple[int, int]] | None = None,
) -> tuple[Any, ...]:
    """Observe how 'transformed' relates to 'original'.

    Creates a tree structure describing the relationship:
    - For atomic values: tracks value, type, identity preservation
    - For containers: recursively tracks all elements
    - For objects: tracks attributes and slots
    - Detects cycles and maintains aliasing information

    This is the "witness" phase - we're just observing and recording,
    not yet judging what changed.
    """
    if seen is None:
        seen = set()

    pair = (id(original), id(transformed))
    if pair in seen:
        # We've seen this pair before - it's a cycle or repeated reference
        return ("CYCLE", tracker.get_group(transformed), transformed)
    seen.add(pair)

    group = tracker.get_group(transformed)

    # Atomic values - the leaves of our story
    if _is_atomic(original):
        return (
            "ATOM",
            _type_name(original),
            id(original) == id(transformed),  # identity preserved?
            _safe_eq(original, transformed),  # values equal?
            group,
            original,  # Store original for before value
            transformed,  # Store transformed for after value
        )

    # Type changed - this is significant
    if type(original) is not type(transformed):
        return (
            "TYPE_CHANGED",
            _type_name(original),
            _type_name(transformed),
            original,
            transformed,
        )

    # Bytearray - mutable byte sequences
    if isinstance(original, bytearray):
        items = []
        for i in range(min(len(original), len(transformed))):
            item_observation = _observe_transformation(original[i], transformed[i], tracker, seen)
            items.append(item_observation)
        return (
            "BYTEARRAY",
            _type_name(original),
            group,
            len(original),
            len(transformed),
            items,
            id(original) == id(transformed),
            transformed,
        )

    # Lists - ordered sequences
    if isinstance(original, list):
        items = []
        for i in range(min(len(original), len(transformed))):
            item_observation = _observe_transformation(original[i], transformed[i], tracker, seen)
            items.append(item_observation)
        return (
            "LIST",
            _type_name(original),
            group,
            len(original),
            len(transformed),
            items,
            transformed,
        )

    # Tuples - immutable ordered sequences
    if isinstance(original, tuple):
        items = []
        for i in range(min(len(original), len(transformed))):
            item_observation = _observe_transformation(original[i], transformed[i], tracker, seen)
            items.append(item_observation)
        return (
            "TUPLE",
            _type_name(original),
            group,
            len(original),
            len(transformed),
            items,
            transformed,
        )

    # Dicts - key-value mappings
    if isinstance(original, dict):
        orig_keys = set(original.keys())
        trans_keys = set(transformed.keys())

        shared_items = []
        for key in orig_keys & trans_keys:
            key_obs = _observe_transformation(key, key, tracker, seen)
            val_obs = _observe_transformation(original[key], transformed[key], tracker, seen)
            shared_items.append((key_obs, val_obs))

        return (
            "DICT",
            _type_name(original),
            group,
            sorted([_repr(k) for k in orig_keys]),
            sorted([_repr(k) for k in trans_keys]),
            shared_items,
            transformed,
        )

    # Sets and frozensets
    if isinstance(original, set | frozenset):
        equality = _safe_eq(original, transformed)
        return (
            "SET",
            _type_name(original),
            group,
            len(original),
            len(transformed),
            id(original) == id(transformed),
            transformed,
            equality,
        )

    # Custom objects - the complex case
    orig_dict, orig_slots = _get_object_state(original)
    trans_dict, trans_slots = _get_object_state(transformed)

    # Observe __dict__ attributes
    dict_items = []
    shared_attrs = set(orig_dict.keys()) & set(trans_dict.keys())
    for attr in sorted(shared_attrs):
        attr_obs = _observe_transformation(orig_dict[attr], trans_dict[attr], tracker, seen)
        dict_items.append((attr, attr_obs))

    # Observe __slots__ attributes
    slot_items = []
    shared_slots = set(orig_slots.keys()) & set(trans_slots.keys())
    for slot in sorted(shared_slots):
        slot_obs = _observe_transformation(orig_slots[slot], trans_slots[slot], tracker, seen)
        slot_items.append((slot, slot_obs))

    return (
        "OBJECT",
        _type_name(original),
        group,
        sorted(orig_dict.keys()),
        sorted(trans_dict.keys()),
        dict_items,
        sorted(orig_slots.keys()),
        sorted(trans_slots.keys()),
        slot_items,
        transformed,
    )


def _get_object_state(obj: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract an object's state from __dict__ and __slots__."""
    obj_dict = {}
    obj_slots = {}

    # Get __dict__ attributes
    if hasattr(obj, "__dict__"):
        obj_dict_raw = obj.__dict__
        if isinstance(obj_dict_raw, dict | types.MappingProxyType):
            obj_dict = {k: v for k, v in obj_dict_raw.items() if not k.startswith("__")}

    # Get __slots__ attributes
    for cls in type(obj).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots or ():
            if isinstance(name, str) and not name.startswith("__"):
                with contextlib.suppress(Exception):
                    obj_slots[name] = getattr(obj, name)

    # Get member descriptors
    for attr in dir(type(obj)):
        if not attr.startswith("_") and isinstance(
            getattr(type(obj), attr, None), types.MemberDescriptorType
        ):
            try:
                obj_slots[attr] = getattr(obj, attr)
            except AttributeError:
                pass

    return obj_dict, obj_slots


def _narrate_changes(
    observation: tuple[Any, ...],
    tracker: _IdentityTracker,
    path: str,
    ignore_atomic_identity: bool,  # noqa: FBT001
    ignore_code_metadata: bool,  # noqa: FBT001
) -> list[Change]:
    """Narrate the changes we observed.

    Converts the observation tree into a flat list of changes.
    This is where we interpret what we witnessed and tell the story.
    """
    changes: list[Change] = []

    tag = observation[0]

    if tag == "CYCLE":
        # Cycles are handled by identity tracking
        return changes

    if tag == "ATOM":
        type_name, identity_preserved, values_equal, group, orig_obj, trans_obj = observation[1:]

        if not values_equal:
            # Value changed - this is the primary difference
            new_change = Change(
                path=path or ".",
                kind=ChangeKind.MODIFIED,
                before=orig_obj,
                after=trans_obj,
                detail=f"value changed (type={type_name})",
            )
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)
        elif not identity_preserved and not ignore_atomic_identity:
            # Only report identity change if values are equal but identity differs
            new_change = Change(
                path=path or ".",
                kind=ChangeKind.IDENTITY,
                detail=f"identity differs for {type_name}",
            )
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        return changes

    if tag == "TYPE_CHANGED":
        orig_type, trans_type, orig_obj, trans_obj = observation[1:]

        # Check if qualnames are the same but class identities differ
        if orig_type == trans_type and type(orig_obj) != type(trans_obj):  # noqa: E721
            detail = (
                f"class identity differs (same qualname '{orig_type}' but different class objects)"
            )
        else:
            detail = f"type changed from {orig_type} to {trans_type}"

        new_change = Change(
            path=path or ".",
            kind=ChangeKind.TYPE,
            before=orig_obj,
            after=trans_obj,
            detail=detail,
        )
        if not _should_ignore_change(new_change, ignore_code_metadata):
            changes.append(new_change)
        return changes

    if tag == "BYTEARRAY":
        type_name, group, orig_len, trans_len, items, identity_preserved, obj = observation[1:]

        if orig_len != trans_len:
            new_change = Change(
                path=path or ".",
                kind=ChangeKind.STRUCTURE,
                before=orig_len,
                after=trans_len,
                detail="bytearray length changed",
            )
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        for i, item_obs in enumerate(items):
            item_path = f"{path}[{i}]" if path else f"[{i}]"
            changes.extend(
                _narrate_changes(
                    item_obs, tracker, item_path, ignore_atomic_identity, ignore_code_metadata
                )
            )

        return changes

    if tag == "LIST":
        type_name, group, orig_len, trans_len, items, obj = observation[1:]

        if orig_len != trans_len:
            new_change = Change(
                path=path or ".",
                kind=ChangeKind.STRUCTURE,
                before=orig_len,
                after=trans_len,
                detail="list length changed",
            )
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        for i, item_obs in enumerate(items):
            item_path = f"{path}[{i}]" if path else f"[{i}]"
            changes.extend(
                _narrate_changes(
                    item_obs, tracker, item_path, ignore_atomic_identity, ignore_code_metadata
                )
            )

        return changes

    if tag == "TUPLE":
        type_name, group, orig_len, trans_len, items, obj = observation[1:]

        if orig_len != trans_len:
            new_change = Change(
                path=path or ".",
                kind=ChangeKind.STRUCTURE,
                before=orig_len,
                after=trans_len,
                detail="tuple length changed",
            )
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        for i, item_obs in enumerate(items):
            item_path = f"{path}[{i}]" if path else f"[{i}]"
            changes.extend(
                _narrate_changes(
                    item_obs, tracker, item_path, ignore_atomic_identity, ignore_code_metadata
                )
            )

        return changes

    if tag == "DICT":
        type_name, group, orig_keys, trans_keys, shared_items, obj = observation[1:]

        orig_keys_set = set(orig_keys)
        trans_keys_set = set(trans_keys)

        for key in orig_keys_set - trans_keys_set:
            key_path = f"{path}[{key}]" if path else f"[{key}]"
            new_change = Change(
                path=key_path, kind=ChangeKind.REMOVED, detail="key removed from dict"
            )
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        for key in trans_keys_set - orig_keys_set:
            key_path = f"{path}[{key}]" if path else f"[{key}]"
            new_change = Change(path=key_path, kind=ChangeKind.ADDED, detail="key added to dict")
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        for key_obs, val_obs in shared_items:
            key_obj = key_obs[-1] if isinstance(key_obs, tuple) else key_obs
            key_repr = _repr(key_obj)
            val_path = f"{path}[{key_repr}]" if path else f"[{key_repr}]"
            changes.extend(
                _narrate_changes(
                    val_obs, tracker, val_path, ignore_atomic_identity, ignore_code_metadata
                )
            )

        return changes

    if tag == "SET":
        type_name, group, orig_len, trans_len, identity_preserved, obj, equality = observation[1:]  # noqa: RUF059

        if orig_len != trans_len:
            new_change = Change(
                path=path or ".",
                kind=ChangeKind.STRUCTURE,
                before=orig_len,
                after=trans_len,
                detail="set length changed",
            )
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        if not equality:
            new_change = Change(
                path=path or ".",
                kind=ChangeKind.MODIFIED,
                detail="set contents changed",
            )
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        if isinstance(obj, frozenset) and not identity_preserved:
            new_change = Change(
                path=path or ".", kind=ChangeKind.IDENTITY, detail="frozenset identity changed"
            )
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        return changes

    if tag == "OBJECT":
        (
            type_name,
            _group,
            orig_attrs,
            trans_attrs,
            dict_items,
            orig_slots,
            trans_slots,
            slot_items,
            obj,
        ) = observation[1:]

        orig_attrs_set = set(orig_attrs)
        trans_attrs_set = set(trans_attrs)

        for attr in orig_attrs_set - trans_attrs_set:
            attr_path = f"{path}.{attr}" if path else attr
            new_change = Change(path=attr_path, kind=ChangeKind.REMOVED, detail="attribute removed")
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        for attr in trans_attrs_set - orig_attrs_set:
            attr_path = f"{path}.{attr}" if path else attr
            new_change = Change(path=attr_path, kind=ChangeKind.ADDED, detail="attribute added")
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        for attr_name, attr_obs in dict_items:
            attr_path = f"{path}.{attr_name}" if path else attr_name
            changes.extend(
                _narrate_changes(
                    attr_obs, tracker, attr_path, ignore_atomic_identity, ignore_code_metadata
                )
            )

        orig_slots_set = set(orig_slots)
        trans_slots_set = set(trans_slots)

        for slot in orig_slots_set - trans_slots_set:
            slot_path = f"{path}.{slot}" if path else slot
            new_change = Change(path=slot_path, kind=ChangeKind.REMOVED, detail="slot removed")
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        for slot in trans_slots_set - orig_slots_set:
            slot_path = f"{path}.{slot}" if path else slot
            new_change = Change(path=slot_path, kind=ChangeKind.ADDED, detail="slot added")
            if not _should_ignore_change(new_change, ignore_code_metadata):
                changes.append(new_change)

        for slot_name, slot_obs in slot_items:
            slot_path = f"{path}.{slot_name}" if path else slot_name
            changes.extend(
                _narrate_changes(
                    slot_obs, tracker, slot_path, ignore_atomic_identity, ignore_code_metadata
                )
            )

        return changes

    return changes


def _should_ignore_change(change: Change, ignore_code_metadata: bool) -> bool:  # noqa: FBT001
    """Determine if a change should be ignored based on parameters."""
    if not ignore_code_metadata:
        return False
    if change.kind != ChangeKind.MODIFIED:
        return False
    if ".__code__." not in change.path:
        return False
    ignored_fields = {"co_firstlineno", "co_lnotab", "co_linetable", "co_colnotab"}
    return any(change.path.endswith(field) for field in ignored_fields)


def _normalize_expected(change_or_path: Change | Path) -> Change:
    """Convert Path or Change to normalized Change."""
    if isinstance(change_or_path, Change):
        return change_or_path
    if isinstance(change_or_path, Path):
        # Bare path means "any change at this path"
        path = object.__getattribute__(change_or_path, "_path")
        return Change(path=path or ".", kind=ChangeKind.MODIFIED)
    raise TypeError(f"Expected Change or Path, got {type(change_or_path)}")


def _changes_match(actual: Change, expected: Change) -> bool:
    """Check if an actual change matches an expected one."""
    if actual.path != expected.path:
        return False

    # If expected specifies kind, it must match
    if expected.kind != ChangeKind.MODIFIED and actual.kind != expected.kind:
        return False

    # If expected specifies values, they must match
    if expected.after is not None and not _safe_eq(actual.after, expected.after):
        return False

    return not (expected.before is not None and not _safe_eq(actual.before, expected.before))


def _changes_equal(change1: Change, change2: Change) -> bool:
    """Check if two changes are exactly equal."""
    return (
        change1.path == change2.path
        and change1.kind == change2.kind
        and _safe_eq(change1.before, change2.before)
        and _safe_eq(change1.after, change2.after)
    )


def _format_story_comparison(comparison: StoryComparison) -> str:
    """Format a readable error message about story differences with context."""
    lines = [f"\nTransformation stories differ in {len(comparison)} unexpected way(s):\n"]

    counter = 1

    # Changes only in reference (missing in reconstruction)
    if comparison.only_in_reference:
        lines.append("Changes in REFERENCE but missing in RECONSTRUCTED:")
        for change in comparison.only_in_reference:
            lines.append(f"{counter}. {change}")
            if change.detail:
                lines.append(f"   ({change.detail})")
            counter += 1
        lines.append("")

    # Changes only in reconstructed (missing in reference)
    if comparison.only_in_reconstructed:
        lines.append("Changes in RECONSTRUCTED but missing in REFERENCE:")
        for change in comparison.only_in_reconstructed:
            lines.append(f"{counter}. {change}")
            if change.detail:
                lines.append(f"   ({change.detail})")
            counter += 1
        lines.append("")

    # Changes that differ between both
    if comparison.different_in_both:
        lines.append("Changes that DIFFER between stories:")
        for ref_change, rec_change in comparison.different_in_both:
            lines.append(f"{counter}. At path: {ref_change.path}")
            lines.append(f"   Reference:     {ref_change}")
            if ref_change.detail:
                lines.append(f"                  ({ref_change.detail})")
            lines.append(f"   Reconstructed: {rec_change}")
            if rec_change.detail:
                lines.append(f"                  ({rec_change.detail})")
            counter += 1
        lines.append("")

    # Now show how to accept these differences
    lines.append("To accept these differences, use expected=[")

    # Collect all changes with their context
    for change in comparison.only_in_reference:
        path_expr = _format_path_for_dsl(change.path)
        comment = "missing in reconstructed"
        lines.append(f"    {_suggest_expected_change(change, path_expr)},  # {comment}")

    for change in comparison.only_in_reconstructed:
        path_expr = _format_path_for_dsl(change.path)
        comment = "missing in reference"
        lines.append(f"    {_suggest_expected_change(change, path_expr)},  # {comment}")

    for ref_change, rec_change in comparison.different_in_both:
        path_expr = _format_path_for_dsl(ref_change.path)
        # Show what the difference is
        if ref_change.kind == ChangeKind.TYPE and rec_change.kind == ChangeKind.TYPE:
            ref_types = f"{_type_name(ref_change.before)} -> {_type_name(ref_change.after)}"
            rec_types = f"{_type_name(rec_change.before)} -> {_type_name(rec_change.after)}"
            comment = f"ref: {ref_types}, rec: {rec_types}"
        elif ref_change.kind == ChangeKind.MODIFIED and rec_change.kind == ChangeKind.MODIFIED:
            comment = f"ref: {_repr(ref_change.after)}, rec: {_repr(rec_change.after)}"
        else:
            comment = f"ref: {ref_change.kind.value}, rec: {rec_change.kind.value}"
        lines.append(f"    {_suggest_expected_change(ref_change, path_expr)},  # {comment}")

    lines.append("]")

    return "\n".join(lines)


def _format_path_for_dsl(path: str) -> str:
    """Format a path for DSL usage."""
    if path == "." or not path:
        return "at"

    clean_path = path.replace("']['", ".").replace("['", ".").replace("']", "")
    clean_path = clean_path.removeprefix(".")
    return f"at.{clean_path}"


def _suggest_expected_change(change: Change, path_expr: str) -> str:
    """Suggest the DSL expression for an expected change."""
    if change.kind == ChangeKind.REMOVED:
        return f"-{path_expr}"
    if change.kind == ChangeKind.ADDED:
        return f"+{path_expr}"
    if change.kind == ChangeKind.MODIFIED:
        return f"{path_expr} > {_repr(change.after)}"
    if change.kind == ChangeKind.IDENTITY:
        return f"{path_expr} & Changed.identity"
    if change.kind == ChangeKind.TYPE:
        return f"{path_expr} & Changed.type"
    if change.kind == ChangeKind.STRUCTURE:
        return f"{path_expr} & Changed.structure"
    assert_never(change.kind)
    raise NotImplementedError


@dataclass
class StoryComparison:
    """Comparison between two transformation stories.

    Tracks what's different between a reference story and a reconstructed story,
    preserving context about which story each change came from.
    """

    only_in_reference: list[Change]
    only_in_reconstructed: list[Change]
    different_in_both: list[tuple[Change, Change]]  # (reference_change, reconstructed_change)

    def __bool__(self) -> bool:
        """Comparison is truthy if there are any differences."""
        return bool(self.only_in_reference or self.only_in_reconstructed or self.different_in_both)

    def __len__(self) -> int:
        """Total number of differences."""
        return (
            len(self.only_in_reference)
            + len(self.only_in_reconstructed)
            + len(self.different_in_both)
        )

    def remove_expected(self, expected: list[Change]) -> StoryComparison:
        """Remove expected differences from this comparison."""
        expected_paths = {_normalize_expected(c).path for c in expected}

        return StoryComparison(
            only_in_reference=[c for c in self.only_in_reference if c.path not in expected_paths],
            only_in_reconstructed=[
                c for c in self.only_in_reconstructed if c.path not in expected_paths
            ],
            different_in_both=[
                (ref, rec) for ref, rec in self.different_in_both if ref.path not in expected_paths
            ],
        )


def compare_stories(reference: Diff, reconstructed: Diff) -> StoryComparison:
    """Compare two transformation stories with full context.

    Returns a comparison that tracks which changes came from which story.
    """
    ref_by_path = {c.path: c for c in reference.changes}
    rec_by_path = {c.path: c for c in reconstructed.changes}

    all_paths = set(ref_by_path.keys()) | set(rec_by_path.keys())

    only_in_reference: list[Change] = []
    only_in_reconstructed: list[Change] = []
    different_in_both: list[tuple[Change, Change]] = []

    for path in all_paths:
        ref_change = ref_by_path.get(path)
        rec_change = rec_by_path.get(path)

        if ref_change is None:
            # Only in reconstructed
            assert rec_change is not None  # noqa: S101
            only_in_reconstructed.append(rec_change)
        elif rec_change is None:
            # Only in reference
            only_in_reference.append(ref_change)
        elif not _changes_equal(ref_change, rec_change):
            # Different in both
            different_in_both.append((ref_change, rec_change))

    return StoryComparison(
        only_in_reference=only_in_reference,
        only_in_reconstructed=only_in_reconstructed,
        different_in_both=different_in_both,
    )
