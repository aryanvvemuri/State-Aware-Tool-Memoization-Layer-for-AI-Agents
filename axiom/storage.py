"""Storage layer for Axiom memoized tool executions.

Provides in-memory record storage with tag-based invalidation (tombstoning)
and fast candidate retrieval.
"""

from __future__ import annotations
import threading
from typing import Dict, List, Optional, Set
from axiom.models import CacheRecord


class StorageInterface:
    """Interface for Axiom storage engines."""

    def save_record(self, record: CacheRecord) -> str:
        raise NotImplementedError

    def get_record(self, record_id: str) -> Optional[CacheRecord]:
        raise NotImplementedError

    def get_candidates(self, domain: str, tool_name: str) -> List[CacheRecord]:
        raise NotImplementedError

    def tombstone_by_tags(self, tags: List[str], domain: Optional[str] = None) -> int:
        raise NotImplementedError

    def get_all_records(self) -> List[CacheRecord]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class InMemoryStorage(StorageInterface):
    """Fast, thread-safe in-memory storage implementation for Axiom."""

    def __init__(self):
        self._lock = threading.RLock()
        self._records: Dict[str, CacheRecord] = {}

    def save_record(self, record: CacheRecord) -> str:
        with self._lock:
            self._records[record.record_id] = record
            return record.record_id

    def get_record(self, record_id: str) -> Optional[CacheRecord]:
        with self._lock:
            return self._records.get(record_id)

    def get_candidates(self, domain: str, tool_name: str) -> List[CacheRecord]:
        """Retrieve candidates matching domain and tool_name.
        Returns all records (including expired/tombstoned) so the decision engine
        can compute best_similarity_seen and produce precise diagnostic miss reasons.
        """
        candidates = []
        with self._lock:
            for rec in self._records.values():
                if rec.domain == domain and rec.tool_name == tool_name:
                    candidates.append(rec)
        return candidates

    def tombstone_by_tags(self, tags: List[str], domain: Optional[str] = None) -> int:
        """Tombstone records containing any of the specified invalidation tags.
        Does not delete permanently, preserving records for auditability and debugging.
        """
        search_tags: Set[str] = {t.strip().lower() for t in tags}
        purged_count = 0
        with self._lock:
            for rec in self._records.values():
                if domain and rec.domain != domain:
                    continue
                if rec.is_tombstoned:
                    continue
                
                rec_tags = {t.strip().lower() for t in rec.invalidation_tags}
                if search_tags.intersection(rec_tags):
                    rec.is_tombstoned = True
                    purged_count += 1

        return purged_count

    def get_all_records(self) -> List[CacheRecord]:
        with self._lock:
            return list(self._records.values())

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
