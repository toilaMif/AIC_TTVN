"""Immutable query revisions make feedback replayable."""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class QueryRevision:
    revision: int
    structured_query: dict[str, object]
    positive_ids: tuple[str, ...] = ()
    negative_ids: tuple[str, ...] = ()


@dataclass
class QuerySession:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    revisions: list[QueryRevision] = field(default_factory=list)

    def start(self, structured_query: dict[str, object]) -> QueryRevision:
        if self.revisions:
            raise ValueError("session already started")
        revision = QueryRevision(revision=0, structured_query=structured_query)
        self.revisions.append(revision)
        return revision

    def revise(self, positive_ids: set[str], negative_ids: set[str]) -> QueryRevision:
        if not self.revisions:
            raise ValueError("session has not started")
        current = self.revisions[-1]
        revision = QueryRevision(
            revision=current.revision + 1,
            structured_query=current.structured_query,
            positive_ids=tuple(sorted(set(current.positive_ids) | positive_ids)),
            negative_ids=tuple(sorted(set(current.negative_ids) | negative_ids)),
        )
        self.revisions.append(revision)
        return revision
