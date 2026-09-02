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


class SQLiteStorage(StorageInterface):
    """Durable SQLite storage engine for Axiom metadata and observations."""

    def __init__(self, db_path: str = "axiom.db"):
        import json
        import sqlite3
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS records (
                        record_id TEXT PRIMARY KEY,
                        tool_name TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        canonical_args_json TEXT NOT NULL,
                        semantic_key TEXT NOT NULL,
                        result_payload_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        ttl_seconds INTEGER NOT NULL,
                        invalidation_tags_json TEXT NOT NULL,
                        provenance_json TEXT NOT NULL,
                        is_tombstoned INTEGER NOT NULL DEFAULT 0,
                        embedding_json TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_domain_tool ON records(domain, tool_name)")

    def _row_to_record(self, row) -> CacheRecord:
        import json
        return CacheRecord(
            record_id=row["record_id"],
            tool_name=row["tool_name"],
            domain=row["domain"],
            canonical_args=json.loads(row["canonical_args_json"]),
            semantic_key=row["semantic_key"],
            result_payload=json.loads(row["result_payload_json"]),
            created_at=row["created_at"],
            ttl_seconds=row["ttl_seconds"],
            invalidation_tags=json.loads(row["invalidation_tags_json"]),
            provenance=json.loads(row["provenance_json"]),
            is_tombstoned=bool(row["is_tombstoned"]),
            embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
        )

    def save_record(self, record: CacheRecord) -> str:
        import json
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO records (
                        record_id, tool_name, domain, canonical_args_json,
                        semantic_key, result_payload_json, created_at,
                        ttl_seconds, invalidation_tags_json, provenance_json,
                        is_tombstoned, embedding_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.record_id,
                    record.tool_name,
                    record.domain,
                    json.dumps(record.canonical_args),
                    record.semantic_key,
                    json.dumps(record.result_payload),
                    record.created_at,
                    record.ttl_seconds,
                    json.dumps(record.invalidation_tags),
                    json.dumps(record.provenance),
                    1 if record.is_tombstoned else 0,
                    json.dumps(record.embedding) if record.embedding else None,
                ))
            return record.record_id

    def get_record(self, record_id: str) -> Optional[CacheRecord]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM records WHERE record_id = ?", (record_id,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def get_candidates(self, domain: str, tool_name: str) -> List[CacheRecord]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM records WHERE domain = ? AND tool_name = ?",
                (domain, tool_name),
            )
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def tombstone_by_tags(self, tags: List[str], domain: Optional[str] = None) -> int:
        import json
        search_tags = {t.strip().lower() for t in tags}
        purged = 0
        with self._lock:
            conn = self._get_connection()
            query = "SELECT * FROM records WHERE is_tombstoned = 0"
            params = []
            if domain:
                query += " AND domain = ?"
                params.append(domain)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            to_tombstone = []
            for r in rows:
                rec_tags = {t.strip().lower() for t in json.loads(r["invalidation_tags_json"])}
                if search_tags.intersection(rec_tags):
                    to_tombstone.append(r["record_id"])

            if to_tombstone:
                with conn:
                    conn.executemany(
                        "UPDATE records SET is_tombstoned = 1 WHERE record_id = ?",
                        [(rid,) for rid in to_tombstone],
                    )
                purged = len(to_tombstone)

        return purged

    def get_all_records(self) -> List[CacheRecord]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM records")
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def clear(self) -> None:
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("DELETE FROM records")

