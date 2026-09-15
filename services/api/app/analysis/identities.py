"""Accounting identity assertions.

These are the deterministic tripwires described in `docs/04-ACCURACY-VALIDATION.md`
§3. They run in code after the Finance agent and before the Critic.

The critical design decision is what a failure *means*. A broken identity is not
a rounding nuisance to be smoothed over — it means we misread the document
(wrong scale, wrong column, a subtotal read as a total). So the correct
response is to route backwards and re-extract, never to adjust a figure until
the identity closes. Silently reconciling is exactly how a plausible, wrong
report gets written.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class IdentityCheck:
    name: str
    passed: bool
    left_label: str
    left: Decimal
    right_label: str
    right: Decimal
    difference: Decimal
    tolerance: Decimal
    relative: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "passed": self.passed,
            "left_label": self.left_label,
            "left": str(self.left),
            "right_label": self.right_label,
            "right": str(self.right),
            "difference": str(self.difference),
            "tolerance": str(self.tolerance),
            "relative": self.relative,
            "message": self.message,
        }
        return d


@dataclass(frozen=True, slots=True)
class IdentityReport:
    checks: list[IdentityCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[IdentityCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def pass_rate(self) -> float:
        if not self.checks:
            return 1.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        if self.passed:
            return f"All {len(self.checks)} accounting identities hold."
        names = ", ".join(c.name for c in self.failures)
        return (
            f"{len(self.failures)} of {len(self.checks)} identities FAILED ({names}). "
            "This indicates a misread source — re-extract rather than adjusting figures."
        )


class IdentityFailureError(Exception):
    """Raised when a hard-stop identity does not hold.

    Carries the report so the graph can route back to ingestion with the exact
    discrepancy attached, rather than a bare "something was wrong".
    """

    def __init__(self, report: IdentityReport) -> None:
        super().__init__(report.summary())
        self.report = report


def _dec(value: Decimal | float | int | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def assert_close(
    name: str,
    left: Decimal | float | int | str,
    right: Decimal | float | int | str,
    *,
    left_label: str = "left",
    right_label: str = "right",
    tol: str | Decimal = CENT,
    rel_tol: float | None = None,
) -> IdentityCheck:
    """One identity. Absolute tolerance by default, relative when `rel_tol` is set.

    Absolute-to-the-cent is right for identities that are definitionally exact
    (revenue - cogs == gross profit). A relative tolerance is right only where
    the source itself rounds, e.g. segment figures that are each rounded before
    being summed.
    """
    lhs, rhs = _dec(left), _dec(right)
    diff = lhs - rhs

    if rel_tol is not None:
        scale = max(abs(lhs), abs(rhs))
        allowed = scale * _dec(rel_tol)
        relative = True
    else:
        allowed = _dec(tol)
        relative = False

    passed = abs(diff) <= allowed
    if passed:
        message = f"{left_label} == {right_label} (difference {diff})"
    else:
        message = (
            f"{left_label} ({lhs}) != {right_label} ({rhs}); off by {diff}, "
            f"tolerance {allowed}. Most likely a scale, column, or subtotal "
            "misread in extraction."
        )

    return IdentityCheck(
        name=name,
        passed=passed,
        left_label=left_label,
        left=lhs,
        right_label=right_label,
        right=rhs,
        difference=diff,
        tolerance=allowed,
        relative=relative,
        message=message,
    )


def check_income_statement(
    *,
    revenue: Decimal | float | str,
    cogs: Decimal | float | str,
    gross_profit: Decimal | float | str,
    opex: Decimal | float | str | None = None,
    operating_income: Decimal | float | str | None = None,
    other_income: Decimal | float | str = 0,
    tax: Decimal | float | str = 0,
    net_income: Decimal | float | str | None = None,
) -> IdentityReport:
    checks = [
        assert_close(
            "gross_profit",
            _dec(revenue) - _dec(cogs),
            gross_profit,
            left_label="revenue - cogs",
            right_label="gross_profit",
        )
    ]

    if opex is not None and operating_income is not None:
        checks.append(
            assert_close(
                "operating_income",
                _dec(gross_profit) - _dec(opex),
                operating_income,
                left_label="gross_profit - opex",
                right_label="operating_income",
            )
        )

    if operating_income is not None and net_income is not None:
        checks.append(
            assert_close(
                "net_income",
                _dec(operating_income) + _dec(other_income) - _dec(tax),
                net_income,
                left_label="operating_income + other_income - tax",
                right_label="net_income",
            )
        )

    return IdentityReport(checks=checks)


def check_balance_sheet(
    *,
    assets: Decimal | float | str,
    liabilities: Decimal | float | str,
    equity: Decimal | float | str,
) -> IdentityReport:
    return IdentityReport(
        checks=[
            assert_close(
                "balance_sheet",
                assets,
                _dec(liabilities) + _dec(equity),
                left_label="assets",
                right_label="liabilities + equity",
            )
        ]
    )


def check_cash_flow(
    *,
    opening_cash: Decimal | float | str,
    net_cash_flow: Decimal | float | str,
    closing_cash: Decimal | float | str,
) -> IdentityReport:
    return IdentityReport(
        checks=[
            assert_close(
                "cash_flow",
                _dec(opening_cash) + _dec(net_cash_flow),
                closing_cash,
                left_label="opening_cash + net_cash_flow",
                right_label="closing_cash",
            )
        ]
    )


def check_segments(
    *,
    segment_values: Iterable[Decimal | float | str],
    total: Decimal | float | str,
    label: str = "segment_revenue",
    rel_tol: float = 0.005,
) -> IdentityReport:
    """Segments against their total, with a relative tolerance.

    Published segment figures are usually individually rounded, so demanding
    cent-exactness here would produce false alarms. 0.5% catches a genuinely
    missing or double-counted segment while tolerating disclosure rounding.
    """
    values = [_dec(v) for v in segment_values]
    return IdentityReport(
        checks=[
            assert_close(
                "segment_total",
                sum(values, Decimal(0)),
                total,
                left_label=f"sum({label}) over {len(values)} segments",
                right_label="total",
                rel_tol=rel_tol,
            )
        ]
    )


def check_all(statements: dict[str, dict[str, Any]]) -> IdentityReport:
    """Run every identity the supplied statements make checkable.

    Only asserts what the data supports — a missing balance sheet means the
    balance-sheet identity is simply not run, never silently "passed".
    """
    checks: list[IdentityCheck] = []

    if "income_statement" in statements:
        checks.extend(check_income_statement(**statements["income_statement"]).checks)
    if "balance_sheet" in statements:
        checks.extend(check_balance_sheet(**statements["balance_sheet"]).checks)
    if "cash_flow" in statements:
        checks.extend(check_cash_flow(**statements["cash_flow"]).checks)
    if "segments" in statements:
        checks.extend(check_segments(**statements["segments"]).checks)

    return IdentityReport(checks=checks)


def enforce(report: IdentityReport) -> None:
    """Hard stop. Raise `IdentityFailureError` if any identity did not hold."""
    if not report.passed:
        raise IdentityFailureError(report)


# --------------------------------------------------------------------------- #
# Derived financial measures (deterministic)
# --------------------------------------------------------------------------- #


def gross_margin(
    revenue: Decimal | float | str, gross_profit: Decimal | float | str
) -> Decimal:
    rev = _dec(revenue)
    if rev == 0:
        raise ZeroDivisionError("gross margin is undefined at zero revenue")
    return (_dec(gross_profit) / rev) * Decimal(100)


def burn_rate(
    opening_cash: Decimal | float | str,
    closing_cash: Decimal | float | str,
    months: int,
) -> Decimal:
    """Average monthly cash consumption. Positive means cash is being consumed."""
    if months <= 0:
        raise ValueError("months must be positive")
    return (_dec(opening_cash) - _dec(closing_cash)) / Decimal(months)


def runway_months(
    cash: Decimal | float | str, monthly_burn: Decimal | float | str
) -> Decimal | None:
    """Months of runway, or `None` when the business is not burning cash.

    `None` is returned rather than `inf` because "infinite runway" is a
    statement no report should make; cash-generative is the correct finding.
    """
    burn = _dec(monthly_burn)
    if burn <= 0:
        return None
    return _dec(cash) / burn
