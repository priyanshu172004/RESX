"""Money, scale factors, and units.

`Decimal` everywhere, never `float`. Binary floating point cannot represent
`0.10`, so a P&L built on floats fails its own accounting identities for
reasons that have nothing to do with the source document.

Scale detection lives here because a 1000x error is the single most damaging
failure mode in financial extraction: a statement headed "in thousands" read at
face value turns $48.9M into $48,920. The extractor must resolve the scale
before any figure is trusted, and the resolved factor travels on the claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Final

# 28 significant digits is plenty for financial statements and keeps the
# identity assertions exact to the cent.
DECIMAL_PRECISION: Final = 28

CURRENCY_SYMBOLS: Final[dict[str, str]] = {
    "$": "USD",
    "US$": "USD",
    "£": "GBP",
    "€": "EUR",
    "¥": "JPY",
    "₹": "INR",
    "CHF": "CHF",
    "A$": "AUD",
    "C$": "CAD",
}

# Ordered longest-first so "in thousands of dollars" is not matched by
# "thousands" alone with a different factor.
_SCALE_PATTERNS: Final[tuple[tuple[re.Pattern[str], Decimal], ...]] = (
    (re.compile(r"\bin\s+billions?\b", re.I), Decimal("1000000000")),
    (re.compile(r"\bin\s+millions?\b", re.I), Decimal("1000000")),
    (re.compile(r"\bin\s+thousands?\b", re.I), Decimal("1000")),
    (re.compile(r"\b\$?\s?bn\b", re.I), Decimal("1000000000")),
    (re.compile(r"\b\$?\s?mm?\b(?!\w)", re.I), Decimal("1000000")),
    (re.compile(r"\b\$?\s?k\b(?!\w)", re.I), Decimal("1000")),
    (re.compile(r"\(\s*000s?\s*\)", re.I), Decimal("1000")),
    (re.compile(r"\bamounts?\s+in\s+lakhs?\b", re.I), Decimal("100000")),
    (re.compile(r"\bin\s+crores?\b", re.I), Decimal("10000000")),
)

_NUMBER = re.compile(
    r"""
    (?P<paren>\()?              # accounting negatives: (1,234)
    \s*
    (?P<sign>[-+])?
    \s*
    (?P<sym>US\$|A\$|C\$|[$£€¥₹])?
    \s*
    (?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<suffix>bn|mm|m|k|%)?
    \s*
    (?(paren)\))
    """,
    re.I | re.X,
)

#: What may legitimately precede a figure in a cell. Currency symbols and the
#: compound signs are consumed by `_NUMBER` itself; bare ISO codes are not, so
#: they are permitted here. Anything else before the digits — a letter, a word,
#: a "Q" — means the cell is a label that contains a number rather than a
#: figure, and `parse_amount` declines it.
_ALLOWED_PREFIX = re.compile(
    r"^(?:US\$|A\$|C\$|[$£€¥₹]|USD|EUR|GBP|JPY|INR|CHF|AUD|CAD|[-+(\s])*$",
    re.I,
)

_ANY_DIGIT = re.compile(r"\d")

_SUFFIX_FACTOR: Final[dict[str, Decimal]] = {
    "bn": Decimal("1000000000"),
    "mm": Decimal("1000000"),
    "m": Decimal("1000000"),
    "k": Decimal("1000"),
}


class MoneyError(ValueError):
    """Raised when a monetary operation is not well defined."""


@dataclass(frozen=True, slots=True)
class Money:
    """An amount with a currency. Immutable, and arithmetic is currency-safe."""

    amount: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):  # pragma: no cover - guard
            raise MoneyError("Money.amount must be a Decimal, never a float")
        if len(self.currency) != 3:
            raise MoneyError(f"currency must be a 3-letter code, got {self.currency!r}")

    # -- arithmetic --------------------------------------------------------

    def _same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise MoneyError(
                f"cannot combine {self.currency} with {other.currency} without an "
                "explicit FX conversion"
            )

    def __add__(self, other: Money) -> Money:
        self._same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: Decimal | int) -> Money:
        return Money(self.amount * Decimal(factor), self.currency)

    def ratio(self, other: Money) -> Decimal:
        """A dimensionless ratio, e.g. gross profit over revenue."""
        self._same_currency(other)
        if other.amount == 0:
            raise MoneyError("division by zero amount")
        with localcontext() as ctx:
            ctx.prec = DECIMAL_PRECISION
            return self.amount / other.amount

    def scaled(self, factor: Decimal) -> Money:
        return Money(self.amount * factor, self.currency)

    def __str__(self) -> str:
        return f"{self.currency} {self.amount:,.2f}"


def detect_scale(text: str) -> tuple[Decimal, str | None]:
    """Find a scale declaration in a table header, caption, or footnote.

    Returns `(factor, matched_phrase)`. A factor of 1 with `None` means no
    declaration was found — which is a materially different state from "found
    and it was 1", so the caller can require confirmation before trusting
    face-value figures.
    """
    for pattern, factor in _SCALE_PATTERNS:
        match = pattern.search(text)
        if match:
            return factor, match.group(0).strip()
    return Decimal(1), None


def detect_currency(text: str) -> str | None:
    for symbol, code in sorted(CURRENCY_SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        if symbol in text:
            return code
    match = re.search(r"\b(USD|EUR|GBP|JPY|INR|CHF|AUD|CAD)\b", text)
    return match.group(1) if match else None


def parse_amount(raw: str | None, *, scale: Decimal = Decimal(1)) -> Decimal | None:
    """Parse one figure out of messy source text.

    Handles thousands separators, accounting-parenthesis negatives, currency
    symbols, and inline magnitude suffixes. Returns `None` rather than guessing
    when the text holds no parseable number — a silent 0 would flow into a
    total and be impossible to spot downstream.

    **The cell must be a figure, not merely contain one.** The regex searches,
    so an earlier version answered any string with a digit in it:

        parse_amount("Q1 2024")   -> 1        (the year silently discarded)
        parse_amount("Jan 2024")  -> 2024
        parse_amount("FY2025")    -> 2025

    That corrupted data on the way in, not just on the way out. A column headed
    "Quarter" holding "Q1 2024 / Q2 2024 / Q3 2024" scored as >80% numeric, was
    classified `quantity`, and `write_dataset` then normalised its cells — so
    the stored CSV read `1 / 2 / 3`, with the years gone. Every consumer
    inherited it: a computation summing the column added 1+2+3, a citation
    quoting the row could never match the source page, and a time series was
    impossible because the periods no longer existed.

    So a number is accepted only when nothing but decoration surrounds it:
    currency symbols or codes and signs before, no second digit group after.
    Declining is the safe direction here — a cell this cannot read stays as
    text, keeps its meaning, and gets classified as a label.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    # The dash variants are deliberate: an accounting table writes "nil"
    # as an em or en dash, and both have to be recognised.
    if not text or text in {"-", "—", "–", "n/a", "N/A", "nil", "NIL"}:  # noqa: RUF001
        return None

    match = _NUMBER.search(text)
    if not match:
        return None

    # Anything before the number must be decoration. `US$`, `A$` and the
    # symbols are consumed by the pattern itself; bare codes like "USD 1,234"
    # are not, so they are allowed here explicitly.
    if not _ALLOWED_PREFIX.fullmatch(text[: match.start()]):
        return None

    # A second digit group means the cell holds more than one number, so which
    # one is "the figure" is a guess. "Q1 2024", "2024 Q1", "3 of 12".
    if _ANY_DIGIT.search(text[match.end() :]):
        return None

    digits = match.group("digits").replace(",", "")
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None

    suffix = (match.group("suffix") or "").lower()
    if suffix == "%":
        # A percentage is not a monetary amount; the caller decides what to do
        # with it, and applying a monetary scale to it would be wrong.
        return value

    if suffix in _SUFFIX_FACTOR:
        value *= _SUFFIX_FACTOR[suffix]
    else:
        value *= scale

    negative = bool(match.group("paren")) or match.group("sign") == "-"
    return -value if negative else value


def parse_money(
    raw: str, *, scale: Decimal = Decimal(1), default_currency: str = "USD"
) -> Money | None:
    amount = parse_amount(raw, scale=scale)
    if amount is None:
        return None
    return Money(amount, detect_currency(str(raw)) or default_currency)


@dataclass(frozen=True, slots=True)
class FxRate:
    base: str
    quote: str
    rate: Decimal
    as_of: str  # ISO date — the rate used is recorded on every converted claim


class FxTable:
    """Dated FX rates.

    Conversion records which rate was used, because a converted figure whose
    rate is unknown cannot be reproduced or audited.
    """

    def __init__(self, rates: list[FxRate] | None = None) -> None:
        self._rates: dict[tuple[str, str, str], Decimal] = {}
        for r in rates or []:
            self._rates[(r.base, r.quote, r.as_of)] = r.rate

    def add(self, rate: FxRate) -> None:
        self._rates[(rate.base, rate.quote, rate.as_of)] = rate.rate

    def convert(self, money: Money, to: str, as_of: str) -> tuple[Money, FxRate]:
        if money.currency == to:
            identity = FxRate(to, to, Decimal(1), as_of)
            return money, identity

        direct = self._rates.get((money.currency, to, as_of))
        if direct is not None:
            used = FxRate(money.currency, to, direct, as_of)
            return Money(money.amount * direct, to), used

        inverse = self._rates.get((to, money.currency, as_of))
        if inverse is not None and inverse != 0:
            rate = Decimal(1) / inverse
            used = FxRate(money.currency, to, rate, as_of)
            return Money(money.amount * rate, to), used

        raise MoneyError(
            f"no {money.currency}->{to} rate for {as_of}; refusing to guess a rate"
        )
