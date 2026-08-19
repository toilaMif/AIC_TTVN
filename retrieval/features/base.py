"""Contracts implemented by visual, OCR, ASR and temporal feature plugins."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from retrieval.domain import Candidate


@dataclass(frozen=True)
class FeatureArtifact:
    feature_type: str
    version: str
    item_count: int
    manifest: dict[str, Any]


class FeatureExtractor(Protocol):
    name: str
    version: str

    def extract(self, items: Sequence[Any]) -> FeatureArtifact: ...


class Retriever(Protocol):
    name: str

    def search(self, query: dict[str, Any], top_k: int) -> list[Candidate]: ...
