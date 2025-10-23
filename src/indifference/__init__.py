__version__ = "0.1.0"

from indifference.core import (
    diff,
    at,
    Change,
    Changed,
    ChangeKind,
    Diff,
    Path,
    assert_equivalent_transformations,
)

assert_changes_equivalent = assert_equivalent_transformations

__all__ = [
    "Change",
    "ChangeKind",
    "Changed",
    "Diff",
    "Path",
    "__version__",
    "assert_equivalent_transformations",
    "at",
    "diff",
]
