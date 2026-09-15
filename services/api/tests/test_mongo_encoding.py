"""BSON encoding for the values this system actually produces.

Found on the first real run against Atlas:

    InvalidDocument: cannot encode object: Decimal('1765000')

The whole point of the analytical engine is that figures are exact `Decimal`s,
and BSON has no type for them. The SQLite backend hid this for months because
it serialised through `json.dumps(..., default=str)` — so every unencodable
value was quietly stringified and the gap only appeared once a real cluster was
on the other end.

These tests need no server: `bson.encode` is the same codec the driver uses, so
asserting against it catches the failure at the same point the driver would.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import bson
import pytest

from app.store.mongo import bson_safe


def encodable(value: object) -> bool:
    """Would the driver accept this document?"""
    try:
        bson.encode({"v": value})
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------- #
# The failure, reproduced
# --------------------------------------------------------------------------- #


def test_a_raw_decimal_really_is_unencodable() -> None:
    """Baseline. If BSON ever gains a Decimal codec, this file can shrink."""
    assert not encodable(Decimal("1765000"))


def test_a_sanitised_decimal_is_encodable() -> None:
    assert encodable(bson_safe(Decimal("1765000")))


# --------------------------------------------------------------------------- #
# Precision
# --------------------------------------------------------------------------- #


def test_full_decimal_precision_survives() -> None:
    """Why a string and not `bson.Decimal128`.

    Decimal128 holds 34 significant digits. A full-precision ratio has more,
    and rounding a figure on the way into storage would break the one promise
    the analytical engine makes.
    """
    exact = Decimal(1) / Decimal(3)
    assert Decimal(bson_safe(exact)) == exact


def test_a_large_exact_integer_is_not_turned_into_a_float() -> None:
    # A float would lose the low digits of a big currency figure, and the loss
    # is invisible: the number still looks like a number.
    huge = Decimal("48920000000000000001")
    assert bson_safe(huge) == "48920000000000000001"
    assert Decimal(bson_safe(huge)) == huge


def test_plain_ints_stay_ints() -> None:
    """Row counts and page numbers must remain queryable as numbers."""
    assert bson_safe(12) == 12
    assert bson_safe({"n_rows": 12})["n_rows"] == 12


# --------------------------------------------------------------------------- #
# The shapes that actually arrive
# --------------------------------------------------------------------------- #


def test_a_pandas_style_result_with_integer_keys() -> None:
    """`DataFrame.to_dict()` produces integer keys; BSON requires strings."""
    result = {"Profit": {0: Decimal("220000"), 1: Decimal("265000")}}
    safe = bson_safe(result)
    assert encodable(safe)
    assert safe["Profit"] == {"0": "220000", "1": "265000"}


def test_nan_and_infinity_become_text() -> None:
    """Both are legal BSON doubles and useless as data.

    NaN does not even equal itself, so a stored NaN is nearly impossible to
    notice downstream. Better to see the string "nan" in a result.
    """
    safe = bson_safe([float("nan"), float("inf"), float("-inf")])
    assert safe == ["nan", "inf", "-inf"]
    assert encodable(safe)


def test_ordinary_floats_are_left_alone() -> None:
    assert bson_safe(0.5) == 0.5


def test_dates_become_iso_strings() -> None:
    assert bson_safe(datetime(2026, 9, 3, 12, 0)) == "2026-09-03T12:00:00"
    assert bson_safe(date(2026, 9, 3)) == "2026-09-03"


def test_sets_become_lists() -> None:
    assert sorted(bson_safe({3, 1, 2})) == [1, 2, 3]


def test_an_unknown_object_is_stringified_not_dropped() -> None:
    """A value that vanishes is harder to debug than one that arrives as text."""

    class Opaque:
        def __str__(self) -> str:
            return "opaque-thing"

    assert bson_safe(Opaque()) == "opaque-thing"
    assert bson_safe({"k": Opaque()}) == {"k": "opaque-thing"}


def test_deeply_nested_mixed_content() -> None:
    payload = {
        "quarters": [
            {"label": "Q1", "profit": Decimal("220000"), "margin": Decimal("0.176")},
            {"label": "Q4", "profit": Decimal("398000"), "margin": Decimal("0.2355")},
        ],
        "meta": {"scale": Decimal("1000"), "rows": 4, "ok": True, "note": None},
    }
    safe = bson_safe(payload)
    assert encodable(safe)
    assert safe["quarters"][0]["profit"] == "220000"
    assert safe["meta"]["rows"] == 4
    assert safe["meta"]["ok"] is True
    assert safe["meta"]["note"] is None


def test_bytes_pass_through_for_embeddings() -> None:
    """Embeddings are a float32 buffer and must not be stringified."""
    buffer = b"\x00\x01\x02\x03"
    assert bson_safe(buffer) == buffer
    assert encodable(bson_safe(buffer))


@pytest.mark.parametrize(
    "value",
    [
        Decimal("0"),
        Decimal("-1765000.005"),
        Decimal("1E+10"),
        Decimal("0.00000000000000000001"),
    ],
)
def test_decimal_round_trip_for_the_awkward_ones(value: Decimal) -> None:
    assert Decimal(bson_safe(value)) == value
