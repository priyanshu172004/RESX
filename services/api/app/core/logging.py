"""Logging setup for the `resx` logger tree.

uvicorn configures its own loggers and nothing else, so an application logger
with no handler discards everything it is given. This attaches one, once.

Deliberately not `logging.basicConfig`: that touches the root logger and would
either duplicate uvicorn's output or be clobbered by it depending on import
order. Configuring the `resx` logger directly and setting `propagate = False`
keeps the two independent.
"""

from __future__ import annotations

import logging
import sys

#: Matches uvicorn's shape closely enough to read as one stream, while the
#: logger name still says which subsystem spoke.
FORMAT = "%(levelname)-8s %(name)s: %(message)s"

_configured = False


def configure_logging(level: str = "info") -> None:
    """Attach a stderr handler to the `resx` logger. Idempotent.

    Called from the app factory rather than at import time, so importing a
    module for a test does not install a handler as a side effect.
    """
    global _configured
    if _configured:
        return

    logger = logging.getLogger("resx")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)

    # The root logger belongs to uvicorn. Propagating would print every line
    # twice under uvicorn and once under pytest, which is the kind of
    # inconsistency that makes people stop reading logs.
    logger.propagate = False

    _configured = True
