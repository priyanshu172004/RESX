"""Tests for the deterministic engine.

These assert the promises in `docs/04-ACCURACY-VALIDATION.md`: that arithmetic
is exact, that a trend is only called a trend when it is significant, that a
broken accounting identity is a hard stop, and that the sandbox actually
contains the code it runs.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.analysis import identities, stats
from app.analysis.sandbox import LocalSubprocessSandbox
from app.core.money import (
    FxRate,
    FxTable,
    Money,
    MoneyError,
    detect_currency,
    detect_scale,
    parse_amount,
    parse_money,
)

# --------------------------------------------------------------------------- #
# Money
# --------------------------------------------------------------------------- #


def test_money_arithmetic_is_exact() -> None:
    # The canonical float failure: 0.1 + 0.2 != 0.3 in binary floating point.
    a = Money(Decimal("0.10"))
    b = Money(Decimal("0.20"))
    assert (a + b).amount == Decimal("0.30")


def test_money_refuses_cross_currency_arithmetic() -> None:
    with pytest.raises(MoneyError, match="explicit FX"):
        Money(Decimal("100"), "USD") + Money(Decimal("100"), "EUR")


def test_money_rejects_float_amounts() -> None:
    with pytest.raises(MoneyError, match="never a float"):
        Money(1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "factor"),
    [
        ("(in thousands)", Decimal("1000")),
        ("Amounts in millions of USD", Decimal("1000000")),
        ("in billions", Decimal("1000000000")),
        ("(000s)", Decimal("1000")),
        ("in crores", Decimal("10000000")),
        ("Revenue by segment", Decimal("1")),
    ],
)
def test_scale_detection(text: str, factor: Decimal) -> None:
    detected, _ = detect_scale(text)
    assert detected == factor


def test_undeclared_scale_is_distinguishable_from_a_scale_of_one() -> None:
    # "found and it was 1" and "not found" must not look the same, or the
    # extractor cannot know whether to require confirmation.
    factor, phrase = detect_scale("Revenue by segment")
    assert factor == Decimal(1)
    assert phrase is None

    factor2, phrase2 = detect_scale("(in thousands)")
    assert phrase2 is not None
    assert factor2 == Decimal("1000")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("48,920", Decimal("48920")),
        ("$1,250.00", Decimal("1250.00")),
        ("(3,720)", Decimal("-3720")),  # accounting negative
        ("-1,234.5", Decimal("-1234.5")),
        ("2.4M", Decimal("2400000")),
        ("15k", Decimal("15000")),
        ("1.2bn", Decimal("1200000000")),
        ("—", None),
        ("n/a", None),
        ("", None),
    ],
)
def test_parse_amount(raw: str, expected: Decimal | None) -> None:
    assert parse_amount(raw) == expected


def test_scale_is_applied_to_parsed_amounts() -> None:
    # The 1000x error this exists to prevent.
    assert parse_amount("48,920", scale=Decimal("1000")) == Decimal("48920000")


def test_percentages_are_not_scaled_as_money() -> None:
    # A monetary scale factor must not be applied to a percentage.
    assert parse_amount("41.8%", scale=Decimal("1000")) == Decimal("41.8")


def test_currency_detection() -> None:
    assert detect_currency("$48,920") == "USD"
    assert detect_currency("€1.2m") == "EUR"
    assert detect_currency("Revenue of 100 GBP") == "GBP"
    assert detect_currency("no currency here") is None


def test_parse_money_carries_currency() -> None:
    money = parse_money("£2,500.50")
    assert money is not None
    assert money.currency == "GBP"
    assert money.amount == Decimal("2500.50")


def test_fx_records_the_rate_used() -> None:
    table = FxTable([FxRate("USD", "EUR", Decimal("0.92"), "2025-12-31")])
    converted, rate = table.convert(Money(Decimal("100"), "USD"), "EUR", "2025-12-31")
    assert converted == Money(Decimal("92.00"), "EUR")
    assert rate.rate == Decimal("0.92")
    assert rate.as_of == "2025-12-31"


def test_fx_refuses_to_invent_a_rate() -> None:
    with pytest.raises(MoneyError, match="refusing to guess"):
        FxTable().convert(Money(Decimal("100"), "USD"), "JPY", "2025-12-31")


# --------------------------------------------------------------------------- #
# Descriptives
# --------------------------------------------------------------------------- #


def test_describe_reports_n_and_missing() -> None:
    d = stats.describe([1, 2, 3, 4, float("nan")])
    assert d.n == 4
    assert d.n_missing == 1
    assert d.mean == 2.5
    assert d.median == 2.5


def test_describe_uses_sample_variance() -> None:
    d = stats.describe([2, 4, 4, 4, 5, 5, 7, 9])
    # Population variance would be 4.0; the sample estimator gives 4.571…
    assert d.variance == pytest.approx(4.5714285, rel=1e-6)


def test_mode_is_none_when_every_value_is_unique() -> None:
    # Reporting "the mode is 1" for [1,2,3,4] would be meaningless.
    assert stats.describe([1, 2, 3, 4]).mode is None
    assert stats.describe([1, 2, 2, 3]).mode == 2


# --------------------------------------------------------------------------- #
# Trend — the significance rule
# --------------------------------------------------------------------------- #


def test_clear_growth_is_reported_as_up() -> None:
    result = stats.trend([10, 12, 14, 16, 18, 20, 22, 24])
    assert result.direction == "up"
    assert result.significant
    assert result.slope == pytest.approx(2.0)
    assert "Mann-Kendall agrees" in result.interpretation


def test_noise_is_reported_as_flat_not_as_a_trend() -> None:
    # The rule that matters: a wobble must not be dressed up as growth.
    noisy = [100, 98, 103, 99, 101, 97, 102, 100, 99, 101]
    result = stats.trend(noisy)
    assert result.direction == "flat"
    assert not result.significant
    assert "Flat within noise" in result.interpretation


def test_decline_is_reported_as_down() -> None:
    assert stats.trend([50, 45, 40, 35, 30, 25]).direction == "down"


def test_trend_needs_enough_observations() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        stats.trend([1, 2])


def test_cagr_is_exact() -> None:
    # 100 -> 121 over 2 years is exactly 10% a year.
    assert stats.cagr(100, 121, 2) == pytest.approx(Decimal("0.1"), rel=1e-9)


def test_cagr_refuses_undefined_inputs() -> None:
    with pytest.raises(ValueError, match="non-positive beginning"):
        stats.cagr(0, 100, 3)
    with pytest.raises(ValueError, match="non-positive ending"):
        stats.cagr(100, -5, 3)


# --------------------------------------------------------------------------- #
# Correlation — association, never causation
# --------------------------------------------------------------------------- #


def test_correlation_reports_p_and_n_and_avoids_causal_language() -> None:
    spend = [100, 120, 140, 160, 180, 200, 220, 240]
    revenue = [420, 500, 560, 660, 700, 810, 860, 950]
    result = stats.correlation(spend, revenue)

    assert result.n == 8
    assert result.pearson_r > 0.95
    assert result.significant
    assert "association, not causation" in result.interpretation
    # No causal verb anywhere in the reported wording.
    for verb in ("drives", "causes", "leads to", "because of"):
        assert verb not in result.interpretation.lower()


def test_correlation_reports_absence_honestly() -> None:
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [5, 1, 7, 2, 8, 3, 6, 4]
    result = stats.correlation(a, b)
    assert not result.significant
    assert "No significant association" in result.interpretation


# --------------------------------------------------------------------------- #
# Outliers — method disagreement is surfaced
# --------------------------------------------------------------------------- #


def test_obvious_outlier_is_caught_by_every_method() -> None:
    values = [10, 11, 9, 10, 12, 11, 10, 9, 11, 10, 500]
    result = stats.outliers(values)
    assert 10 in result.iqr_indices
    assert 10 in result.modified_zscore_indices
    assert 10 in result.consensus_indices


def test_method_dependent_flags_are_reported_separately() -> None:
    values = [10, 11, 9, 10, 12, 11, 10, 9, 11, 10, 18]
    result = stats.outliers(values)
    flagged = (
        set(result.iqr_indices)
        | set(result.zscore_indices)
        | set(result.modified_zscore_indices)
    )
    # Whatever the methods decide, anything only one test found must be
    # disclosed as method-dependent rather than asserted as an outlier.
    for idx in result.method_dependent_indices:
        assert idx in flagged
        assert idx not in result.consensus_indices


# --------------------------------------------------------------------------- #
# Seasonality
# --------------------------------------------------------------------------- #


def test_seasonality_refuses_to_report_below_two_cycles() -> None:
    result = stats.seasonality([1, 5, 2, 6, 3], period=4)
    assert not result.detected
    assert "at least 2 are required" in result.note


def test_seasonality_detects_a_strong_repeating_pattern() -> None:
    values = [10, 40, 20, 50] * 5
    result = stats.seasonality(values, period=4)
    assert result.cycles_available == 5.0
    assert result.seasonal_strength > 0.3


# --------------------------------------------------------------------------- #
# Accounting identities
# --------------------------------------------------------------------------- #


def test_consistent_statements_pass() -> None:
    report = identities.check_income_statement(
        revenue="48920000.00",
        cogs="28471000.00",
        gross_profit="20449000.00",
        opex="9640000.00",
        operating_income="10809000.00",
        tax="2161800.00",
        net_income="8647200.00",
    )
    assert report.passed
    assert report.pass_rate == 1.0
    assert "All 3 accounting identities hold" in report.summary()


def test_a_thousandfold_scale_error_is_caught() -> None:
    # The exact failure mode scale detection exists to prevent: gross profit
    # read from a "in thousands" table at face value.
    report = identities.check_income_statement(
        revenue="48920000.00",
        cogs="28471000.00",
        gross_profit="20449.00",
    )
    assert not report.passed
    failure = report.failures[0]
    assert failure.name == "gross_profit"
    assert "scale, column, or subtotal misread" in failure.message


def test_identity_failure_is_a_hard_stop_carrying_the_report() -> None:
    report = identities.check_balance_sheet(
        assets="1000000", liabilities="600000", equity="399000"
    )
    assert not report.passed
    with pytest.raises(identities.IdentityFailureError) as excinfo:
        identities.enforce(report)
    # The graph needs the discrepancy to route back to ingestion usefully.
    assert excinfo.value.report.failures[0].difference == Decimal("1000")
    assert "re-extract rather than adjusting" in str(excinfo.value)


def test_identity_tolerance_is_exactly_one_cent() -> None:
    # The documented tolerance is +/-0.01, which absorbs cent-level rounding in
    # the source. So one cent out passes and two cents out fails — asserting
    # the boundary itself, because that is the number a future change would
    # loosen without anyone noticing.
    at_tolerance = identities.check_income_statement(
        revenue="100.00", cogs="40.00", gross_profit="60.01"
    )
    assert at_tolerance.passed

    beyond_tolerance = identities.check_income_statement(
        revenue="100.00", cogs="40.00", gross_profit="60.02"
    )
    assert not beyond_tolerance.passed


def test_segments_tolerate_disclosure_rounding_but_not_a_missing_segment() -> None:
    ok = identities.check_segments(segment_values=["10000", "20000", "30001"], total="60000")
    assert ok.passed  # 0.0017% out — published rounding

    missing = identities.check_segments(segment_values=["10000", "20000"], total="60000")
    assert not missing.passed


def test_check_all_only_asserts_what_the_data_supports() -> None:
    report = identities.check_all(
        {
            "income_statement": {
                "revenue": "100",
                "cogs": "40",
                "gross_profit": "60",
            }
        }
    )
    # No balance sheet supplied, so its identity is not run — and certainly not
    # silently passed.
    assert report.passed
    assert [c.name for c in report.checks] == ["gross_profit"]


def test_derived_measures() -> None:
    assert identities.gross_margin("48920000", "20449000") == pytest.approx(
        Decimal("41.8"), abs=Decimal("0.01")
    )
    assert identities.burn_rate("18104000", "14384000", 3) == Decimal("1240000")
    assert identities.runway_months("18104000", "1240000") == pytest.approx(
        Decimal("14.6"), abs=Decimal("0.05")
    )


def test_runway_is_none_when_cash_generative() -> None:
    # "Infinite runway" is not a statement a report should make.
    assert identities.runway_months("1000000", "-50000") is None


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sandbox() -> LocalSubprocessSandbox:
    return LocalSubprocessSandbox(wall_timeout_seconds=20)


def test_local_sandbox_declares_it_is_not_a_security_boundary(
    sandbox: LocalSubprocessSandbox,
) -> None:
    # Callers must be able to tell the dev backend from the real one.
    assert sandbox.is_security_boundary is False


def test_sandbox_computes_and_records(sandbox: LocalSubprocessSandbox) -> None:
    record = sandbox.run(
        "from decimal import Decimal\n"
        'result = Decimal("48920000.00") - Decimal("28471000.00")\n'
        'print("computed")',
        inputs=["ds_pnl_fy25"],
    )
    assert record.ok
    assert record.result == Decimal("20449000.00")
    assert record.stdout.strip() == "computed"
    assert record.inputs == ["ds_pnl_fy25"]
    assert record.image_digest.startswith("local-subprocess:sha256:")
    assert record.duration_ms >= 0
    assert record.computation_id.startswith("cmp_")


def test_sandbox_blocks_network(sandbox: LocalSubprocessSandbox) -> None:
    record = sandbox.run("import socket\nsocket.socket()")
    assert not record.ok
    assert "network access is disabled" in (record.error or "")


def test_sandbox_blocks_subprocesses(sandbox: LocalSubprocessSandbox) -> None:
    record = sandbox.run('import subprocess\nsubprocess.run(["echo", "x"])')
    assert not record.ok


def test_sandbox_blocks_os_system(sandbox: LocalSubprocessSandbox) -> None:
    record = sandbox.run('import os\nos.system("echo x")')
    assert not record.ok


def test_sandbox_cannot_import_the_application(sandbox: LocalSubprocessSandbox) -> None:
    # Model-written code must not be able to reach RESX internals and step
    # around the guards. Checked through several spellings, because an editable
    # install makes `app` resolvable via a meta-path finder in site-packages
    # rather than via sys.path, and a single probe missed that entirely.
    for attempt in (
        "import app",
        "import app.core.config",
        "import app.analysis.sandbox",
        "from app.core import config",
        "result = __import__('app')",
    ):
        record = sandbox.run(attempt + "\nresult = 1")
        assert not record.ok, f"{attempt!r} was allowed to reach the application"
        assert "No module named 'app'" in (record.error or ""), record.error


def test_sandbox_cannot_reach_the_application_via_the_editable_finder(
    sandbox: LocalSubprocessSandbox,
) -> None:
    # The finder an editable install registers must not be usable directly
    # either: denying `app` while leaving its loader exposed would be theatre.
    record = sandbox.run("import resx_api\nresult = 1")
    assert not record.ok
    assert "No module named" in (record.error or "")


def test_sandbox_can_still_import_the_analysis_stack(
    sandbox: LocalSubprocessSandbox,
) -> None:
    # The denial must be narrow. Blocking pandas would make the deterministic
    # engine useless, which is the whole reason the sandbox exists.
    record = sandbox.run("import pandas, numpy\nresult = int(pandas.Series([1, 2]).sum())")
    assert record.ok, record.error
    assert record.result == 3


def test_sandbox_enforces_wall_clock(sandbox: LocalSubprocessSandbox) -> None:
    record = sandbox.run("while True:\n    pass", timeout_seconds=3)
    assert not record.ok
    assert "TimeoutError" in (record.error or "")


def test_sandbox_captures_errors_rather_than_raising(
    sandbox: LocalSubprocessSandbox,
) -> None:
    record = sandbox.run("result = 1 / 0")
    assert not record.ok
    assert "ZeroDivisionError" in (record.error or "")


def test_sandbox_exposes_the_scientific_stack(sandbox: LocalSubprocessSandbox) -> None:
    record = sandbox.run(
        "import pandas as pd\n"
        "import numpy as np\n"
        "from scipy import stats as st\n"
        "df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})\n"
        "result = {'mean': float(df.x.mean()), 'n': int(len(df))}"
    )
    assert record.ok
    assert record.result == {"mean": 3.0, "n": 5}


def test_code_preview_is_truncated(sandbox: LocalSubprocessSandbox) -> None:
    record = sandbox.run("\n".join(f"x{i} = {i}" for i in range(10)) + "\nresult = 1")
    assert "(+8 lines)" in record.code_preview
