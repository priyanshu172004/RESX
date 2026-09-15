"""Back up a workspace, and prove the backup restores.

A backup nobody has restored is a hypothesis. The failure mode is not that the
dump is missing — you notice that — it is that the dump exists, has a plausible
size, and cannot be read back. Nobody finds out until the day it matters.

So `backup` and `restore` are one script, and `verify` runs both against a
throwaway database and compares counts. That last command is the only one that
proves anything.

    python scripts/backup.py backup  --out ./backups
    python scripts/backup.py restore --from ./backups/resx-2026-09-15.json
    python scripts/backup.py verify                 # the drill

**Scope is one workspace at a time**, because that is the unit a person asks
about ("restore my corpus"), and because a whole-database restore into a live
system is a different and far more dangerous operation than this script should
make easy.

What is deliberately *not* backed up: `users`, `refresh_tokens` and
`audit_log`. Password hashes and session tokens in a JSON file on someone's
laptop is a worse outcome than losing them, and an audit log that can be
restored from a file an operator controls is not an audit log.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.store.factory import build_store
from app.store.local import LocalStore

#: Everything a workspace's analysis rests on. Ordered so that restoring in
#: sequence satisfies the foreign keys — documents before the pages and chunks
#: that reference them.
COLLECTIONS = (
    "documents",
    "pages",
    "chunks",
    "datasets",
    "runs",
    "computations",
    "claims",
    "citations",
    "verdicts",
)

#: Never written to a backup file. See the module docstring.
EXCLUDED = ("users", "refresh_tokens", "audit_log")


def dump(store: Any, workspace_id: str) -> dict[str, Any]:
    """Read a workspace out of whichever backend is configured."""
    payload: dict[str, Any] = {
        "format": 1,
        "workspace_id": workspace_id,
        "created_at": time.time(),
        "collections": {},
    }
    for name in COLLECTIONS:
        payload["collections"][name] = _read_all(store, name, workspace_id)
    return payload


def _read_all(store: Any, collection: str, workspace_id: str) -> list[dict[str, Any]]:
    if getattr(store, "backend", "") == "mongo":
        rows = store.db[collection].find({"workspace_id": workspace_id}, {"_id": 0})
        return [_encodable(dict(r)) for r in rows]

    # SQLite. The table name comes from `COLLECTIONS`, a constant in this file,
    # and is checked against it again here — SQL identifiers cannot be
    # parameterised, so the only safe source for one is a value this file owns.
    if collection not in COLLECTIONS:
        raise ValueError(f"refusing to read unknown table {collection!r}")
    cursor = store._conn.execute(
        f"SELECT * FROM {collection} WHERE workspace_id = ?",  # noqa: S608 - checked above
        (workspace_id,),
    )
    return [_encodable(dict(row)) for row in cursor.fetchall()]


def _real_columns(store: Any, table: str) -> list[str]:
    """The columns this table actually has.

    Load-bearing on restore. The column names in a backup file come from the
    file, and a file is something a person can hand you — so building an INSERT
    from them directly would let a crafted backup put arbitrary text into an
    SQL identifier position, where it cannot be parameterised. Intersecting
    with the live schema means only names the database already has can appear.
    """
    rows = store._conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r["name"]) for r in rows]


def _encodable(row: dict[str, Any]) -> dict[str, Any]:
    """Make one row both JSON-safe and bindable by either backend.

    Two conversions, and the second was found by running the drill rather than
    by reading the code:

    **Buffers are dropped.** Embeddings are raw float32. Base64 would roughly
    double an already large file, and a vector is re-derivable by re-ingesting
    while being worthless if the embedding model has changed in between. The
    restore reports how many it dropped, so nobody is surprised by
    lexical-only retrieval afterwards.

    **Lists and dicts become JSON text.** MongoDB returns `warnings`,
    `low_conf_pages` and `payload` as real arrays and objects; SQLite stores
    the same fields as TEXT and refuses to bind a list at all — the drill
    failed with "Error binding parameter 8, probably unsupported type". A
    backup that can only be restored into the backend it came from is not much
    of a backup, and this is precisely the failure that stays invisible until
    the day it matters.
    """
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (bytes, bytearray, memoryview)):
            out[key] = None
        elif isinstance(value, (list, dict, tuple, set)):
            out[key] = json.dumps(list(value) if isinstance(value, (tuple, set)) else value)
        elif isinstance(value, bool):
            # SQLite has no boolean type and Mongo does. Stored as the integer
            # both backends already use.
            out[key] = int(value)
        else:
            out[key] = value
    return out


def restore(store: Any, payload: dict[str, Any]) -> dict[str, int]:
    """Write a dump back, returning how many rows landed in each collection."""
    workspace_id = str(payload["workspace_id"])
    counts: dict[str, int] = {}

    for name in COLLECTIONS:
        rows = payload["collections"].get(name) or []
        counts[name] = len(rows)
        if not rows:
            continue

        if getattr(store, "backend", "") == "mongo":
            store.db[name].delete_many({"workspace_id": workspace_id})
            store.db[name].insert_many([dict(r) for r in rows])
            continue

        if name not in COLLECTIONS:
            raise ValueError(f"refusing to write unknown table {name!r}")

        # Only columns the live table really has, in the schema's own order.
        # A backup file naming a column that does not exist is ignored rather
        # than interpolated — see `_real_columns`.
        allowed = _real_columns(store, name)
        columns = [c for c in allowed if c in rows[0]]
        if not columns:
            raise ValueError(f"backup for {name!r} has no recognisable columns")

        placeholders = ",".join("?" for _ in columns)
        joined = ",".join(columns)
        with store.tx() as conn:
            conn.execute(
                f"DELETE FROM {name} WHERE workspace_id = ?",  # noqa: S608 - name checked above
                (workspace_id,),
            )
            conn.executemany(
                # Both identifiers are now drawn from the live schema, never
                # from the file.
                f"INSERT OR REPLACE INTO {name} ({joined}) VALUES ({placeholders})",  # noqa: S608
                [tuple(row.get(c) for c in columns) for row in rows],
            )
    return counts


def verify() -> int:
    """The drill: back up, restore into a fresh database, compare.

    This is the only command that proves anything. A dump that writes without
    error and cannot be read back is the failure this exists to catch, and it
    is invisible until the day someone needs it.
    """
    settings = get_settings()
    source = build_store(settings)
    workspaces = _workspaces(source)
    if not workspaces:
        print("no workspaces to verify — ingest something first")
        return 1

    workspace_id = workspaces[0]
    payload = dump(source, workspace_id)
    original = {k: len(v) for k, v in payload["collections"].items()}

    with tempfile.TemporaryDirectory() as tmp:
        target = LocalStore(Path(tmp) / "restored.db")
        try:
            restore(target, payload)
            readback = {
                name: len(_read_all(target, name, workspace_id)) for name in COLLECTIONS
            }
        finally:
            target.close()

    mismatches = [
        f"{name}: backed up {original[name]}, read back {readback[name]}"
        for name in COLLECTIONS
        if original[name] != readback[name]
    ]

    print(f"  workspace {workspace_id}")
    for name in COLLECTIONS:
        mark = "ok " if original[name] == readback[name] else "BAD"
        print(f"    {mark} {name:<14} {original[name]:>6} -> {readback[name]:>6}")

    dropped = sum(
        1 for row in payload["collections"].get("chunks") or [] if row.get("embedding") is None
    )
    if dropped:
        print(
            f"\n  {dropped} chunk(s) restored without embeddings — re-ingest "
            f"those documents, or retrieval on them is lexical only."
        )

    if mismatches:
        print("\n  FAILED")
        for line in mismatches:
            print(f"    {line}")
        return 1
    print("\n  PASSED — the backup restores")
    return 0


def _workspaces(store: Any) -> list[str]:
    if getattr(store, "backend", "") == "mongo":
        return sorted(store.db.documents.distinct("workspace_id"))
    rows = store._conn.execute("SELECT DISTINCT workspace_id FROM documents").fetchall()
    return sorted(str(r["workspace_id"]) for r in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    backup_cmd = sub.add_parser("backup", help="write a workspace to a JSON file")
    backup_cmd.add_argument("--out", default="./backups")
    backup_cmd.add_argument("--workspace", default="")

    restore_cmd = sub.add_parser("restore", help="write a JSON file back")
    restore_cmd.add_argument("--from", dest="source", required=True)

    sub.add_parser("verify", help="back up and restore into a throwaway database")

    args = parser.parse_args()

    if args.command == "verify":
        return verify()

    settings = get_settings()
    store = build_store(settings)

    if args.command == "backup":
        workspace_id = args.workspace or (_workspaces(store) or [""])[0]
        if not workspace_id:
            print("no workspace to back up", file=sys.stderr)
            return 1
        payload = dump(store, workspace_id)
        directory = Path(args.out)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"resx-{workspace_id}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        total = sum(len(v) for v in payload["collections"].values())
        print(f"  wrote {total} rows to {path}")
        return 0

    payload = json.loads(Path(args.source).read_text(encoding="utf-8"))
    counts = restore(store, payload)
    print(f"  restored {sum(counts.values())} rows into {payload['workspace_id']}")
    for name, count in counts.items():
        print(f"    {name:<14} {count:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
