"""Typed backend failures that map to distinct HTTP responses."""

from __future__ import annotations


class DatasetContractError(ValueError):
    """The fetched dataset is missing a field the backend consumes."""

    def __init__(self, field: str) -> None:
        super().__init__(f"invalid dataset field: {field}")
        self.field = field


class DatasetAccessError(RuntimeError):
    """The organizer API rejected the token for the requested dataset."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"dataset API HTTP {status_code}")
        self.status_code = status_code


class SessionCapacityError(RuntimeError):
    """All session slots are held by recently active browsers."""
