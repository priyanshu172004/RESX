"""The store under parallel writers.

The graph runs the specialist nodes in parallel threads and they all write
through one shared SQLite connection. `append_event` allocated its sequence
number with `SELECT MAX(seq)` followed by `INSERT`, so two agents emitting a
tool call at the same moment computed the same number and the second lost:

    sqlite3.IntegrityError: UNIQUE constraint failed:
        run_events.run_id, run_events.seq

That exception escapes the emit callback, propagates through the node, and
kills the whole run. Being a race it surfaced intermittently, which is worse
than surfacing always -- it looked like a different bug each time.

The Mongo backend never had it: sequences there come from `find_one_and_update`,
which is atomic server-side. The SQLite fallback simply never got the
equivalent.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.store.local import LocalStore

WORKSPACE = "ws_concurrent"


def test_parallel_event_writers_do_not_collide(tmp_path: Path) -> None:
    """The reproduction. Eight writers is more than the graph runs at once."""
    store = LocalStore(tmp_path / "resx.db")
    run_id = store.create_run(workspace_id=WORKSPACE, question="q", corpus_ids=[])

    def emit(index: int) -> int:
        return store.append_event(
            workspace_id=WORKSPACE,
            run_id=run_id,
            kind="tool_call",
            payload={"i": index},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        # Raises if any worker hit the IntegrityError.
        sequences = list(pool.map(emit, range(200)))

    events = store.read_events(workspace_id=WORKSPACE, run_id=run_id)
    assert len(events) == 200, "events were lost"
    assert len(set(sequences)) == 200, "a sequence number was issued twice"
    # Contiguous from 1, because the stream resumes by sequence number and a
    # hole would make a reconnecting client wait for an event that never comes.
    assert sorted(e["seq"] for e in events) == list(range(1, 201))
    store.close()


def test_parallel_claim_writers_all_persist(tmp_path: Path) -> None:
    """Events were where it crashed, but every write shared the same
    unsynchronised connection."""
    from app.agents.schemas import AgentName, Citation, Claim

    store = LocalStore(tmp_path / "resx.db")
    run_id = store.create_run(workspace_id=WORKSPACE, question="q", corpus_ids=[])

    def write(index: int) -> None:
        store.add_claim(
            workspace_id=WORKSPACE,
            run_id=run_id,
            claim=Claim(
                claim_id=f"clm_{index}",
                agent=AgentName.FINANCE,
                statement=f"A finding numbered {index} was established.",
                confidence=0.95,
                citations=[Citation(url="https://example.com/a", quote="Q" * 20)],
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(120)))

    assert len(store.list_claims(workspace_id=WORKSPACE, run_id=run_id)) == 120
    store.close()


def test_the_sequence_is_per_run_not_global(tmp_path: Path) -> None:
    """Two concurrent runs must not consume each other's numbering."""
    store = LocalStore(tmp_path / "resx.db")
    runs = [
        store.create_run(workspace_id=WORKSPACE, question=f"q{i}", corpus_ids=[])
        for i in range(2)
    ]

    def emit(pair: tuple[str, int]) -> int:
        run_id, index = pair
        return store.append_event(
            workspace_id=WORKSPACE, run_id=run_id, kind="node_end", payload={"i": index}
        )

    work = [(run_id, i) for run_id in runs for i in range(50)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(emit, work))

    for run_id in runs:
        events = store.read_events(workspace_id=WORKSPACE, run_id=run_id)
        assert sorted(e["seq"] for e in events) == list(range(1, 51))
    store.close()
