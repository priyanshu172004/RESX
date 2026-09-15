"""Deterministic statistics.

Every function here is pure Python/numpy/scipy. No model ever computes any of
it — an LLM may only decide *which* of these to call. That division is the
whole reason the numbers in a RESX report can be trusted.

Two conventions run through the module and both are deliberate:

  * `n` is always reported. A mean over n=3 and a mean over n=30,000 are
    different kinds of statement, and the UI shows which one you are reading.
  * A trend is only called a trend when it is statistically significant.
    "Flat within noise" is a real and frequently correct finding, and reporting
    a +0.2% wobble as growth is how a dashboard lies to a CEO.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Literal

import numpy as np
from scipy import stats as scipy_stats

#: What every array in this system is: float32, because that is what the store
#: persists and what the providers return. Named rather than repeated so the
#: dtype is stated once.
FloatArray = np.ndarray[Any, np.dtype[np.float32]]

Direction = Literal["up", "down", "flat"]


# --------------------------------------------------------------------------- #
# Descriptives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Descriptives:
    n: int
    n_missing: int
    mean: float
    median: float
    mode: float | None
    variance: float
    std: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    iqr: float
    skew: float | None
    kurtosis: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(values: Sequence[float]) -> tuple[FloatArray, int]:
    arr = np.asarray(list(values), dtype="float64")
    finite = arr[np.isfinite(arr)]
    return finite, int(arr.size - finite.size)


def describe(values: Sequence[float]) -> Descriptives:
    arr, n_missing = _clean(values)
    if arr.size == 0:
        raise ValueError("describe() needs at least one finite observation")

    # Sample variance (ddof=1) is the right estimator for business data, which
    # is a sample and not a population. With n=1 it is undefined, so report 0
    # rather than nan.
    ddof = 1 if arr.size > 1 else 0

    mode_value: float | None
    mode_result = scipy_stats.mode(arr, keepdims=False)
    mode_count = int(np.atleast_1d(mode_result.count)[0])
    mode_value = float(np.atleast_1d(mode_result.mode)[0]) if mode_count > 1 else None

    q1, q3 = (float(x) for x in np.percentile(arr, [25, 75]))

    return Descriptives(
        n=int(arr.size),
        n_missing=n_missing,
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        mode=mode_value,
        variance=float(np.var(arr, ddof=ddof)),
        std=float(np.std(arr, ddof=ddof)),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        skew=float(scipy_stats.skew(arr)) if arr.size > 2 else None,
        kurtosis=float(scipy_stats.kurtosis(arr)) if arr.size > 3 else None,
    )


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TrendResult:
    n: int
    slope: float
    intercept: float
    slope_ci_low: float
    slope_ci_high: float
    r_squared: float
    p_value: float
    # Non-parametric confirmation. A least-squares slope assumes linearity;
    # Mann-Kendall only assumes monotonicity, so agreement between the two is
    # much stronger evidence than either alone.
    mann_kendall_tau: float
    mann_kendall_p: float
    significant: bool
    direction: Direction
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def trend(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    x: Sequence[float] | None = None,
) -> TrendResult:
    """OLS slope plus a Mann-Kendall monotonic-trend test."""
    arr, _ = _clean(values)
    if arr.size < 3:
        raise ValueError("trend() needs at least 3 observations")

    xs = (
        np.arange(arr.size, dtype="float64")
        if x is None
        else np.asarray(list(x), dtype="float64")
    )
    if xs.size != arr.size:
        raise ValueError("x and values must be the same length")

    reg = scipy_stats.linregress(xs, arr)
    tau, mk_p = scipy_stats.kendalltau(xs, arr)

    # linregress reports the standard error of the slope; turn it into a CI.
    dof = arr.size - 2
    t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, dof)) if dof > 0 else 0.0
    half_width = t_crit * float(reg.stderr)

    p_value = float(reg.pvalue)
    significant = p_value < alpha

    if not significant:
        direction: Direction = "flat"
        interpretation = (
            f"Flat within noise (p = {p_value:.3f}, n = {arr.size}). "
            "The slope is not distinguishable from zero at the chosen level."
        )
    else:
        direction = "up" if reg.slope > 0 else "down"
        agree = (tau > 0) == (reg.slope > 0) and mk_p < alpha
        interpretation = (
            f"Trending {direction} (slope {reg.slope:.4g} per period, p = {p_value:.3g}, "
            f"n = {arr.size})."
            + (
                " Mann-Kendall agrees, so the trend is monotonic and not an artefact of "
                "the linear fit."
                if agree
                else " Mann-Kendall does NOT confirm monotonicity — treat the linear fit "
                "with caution."
            )
        )

    return TrendResult(
        n=int(arr.size),
        slope=float(reg.slope),
        intercept=float(reg.intercept),
        slope_ci_low=float(reg.slope) - half_width,
        slope_ci_high=float(reg.slope) + half_width,
        r_squared=float(reg.rvalue) ** 2,
        p_value=p_value,
        mann_kendall_tau=float(tau),
        mann_kendall_p=float(mk_p),
        significant=significant,
        direction=direction,
        interpretation=interpretation,
    )


def cagr(begin: Decimal | float, end: Decimal | float, periods: float) -> Decimal:
    """Compound annual growth rate over `periods` years.

    Raises on a non-positive beginning value: CAGR from zero or from a loss is
    mathematically undefined, and the usual "just return a big number" fudge
    produces confident nonsense in a board pack.
    """
    b, e = Decimal(str(begin)), Decimal(str(end))
    if periods <= 0:
        raise ValueError("periods must be positive")
    if b <= 0:
        raise ValueError("CAGR is undefined for a non-positive beginning value")
    if e <= 0:
        raise ValueError("CAGR is undefined for a non-positive ending value")

    ratio = float(e / b)
    growth = ratio ** (1.0 / periods) - 1.0
    return Decimal(str(round(growth, 10)))


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    ci_low: float
    ci_high: float
    significant: bool
    strength: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fisher_ci(r: float, n: int, alpha: float) -> tuple[float, float]:
    if n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    crit = float(scipy_stats.norm.ppf(1 - alpha / 2))
    return (math.tanh(z - crit * se), math.tanh(z + crit * se))


def correlation(
    a: Sequence[float], b: Sequence[float], *, alpha: float = 0.05
) -> CorrelationResult:
    """Pearson and Spearman, each with its p-value, plus a Fisher-z CI.

    Association only. The wording of `interpretation` is deliberately
    non-causal, and the Critic agent is prompted to reject causal claims that
    rest on this output — "marketing spend drives revenue" and "marketing spend
    correlates with revenue" imply completely different decisions.
    """
    x = np.asarray(list(a), dtype="float64")
    y = np.asarray(list(b), dtype="float64")
    if x.size != y.size:
        raise ValueError("correlation() inputs must be the same length")

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < 3:
        raise ValueError("correlation() needs at least 3 paired observations")

    pr, pp = scipy_stats.pearsonr(x, y)
    sr, sp = scipy_stats.spearmanr(x, y)
    lo, hi = _fisher_ci(float(pr), n, alpha)

    magnitude = abs(float(pr))
    strength = (
        "negligible"
        if magnitude < 0.1
        else "weak"
        if magnitude < 0.3
        else "moderate"
        if magnitude < 0.5
        else "strong"
        if magnitude < 0.7
        else "very strong"
    )
    significant = float(pp) < alpha

    if significant:
        sign = "positive" if pr > 0 else "negative"
        interpretation = (
            f"A {strength} {sign} association (r = {pr:.2f}, p = {pp:.3g}, n = {n}). "
            "This is association, not causation; direction of causality is not "
            "established by these data."
        )
    else:
        interpretation = f"No significant association (r = {pr:.2f}, p = {pp:.3g}, n = {n})."

    return CorrelationResult(
        n=n,
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_rho=float(sr),
        spearman_p=float(sp),
        ci_low=lo,
        ci_high=hi,
        significant=significant,
        strength=strength,
        interpretation=interpretation,
    )


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OutlierResult:
    n: int
    iqr_indices: list[int]
    zscore_indices: list[int]
    modified_zscore_indices: list[int]
    consensus_indices: list[int] = field(default_factory=list)
    method_dependent_indices: list[int] = field(default_factory=list)
    lower_fence: float = 0.0
    upper_fence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def outliers(values: Sequence[float], *, z_threshold: float = 3.0) -> OutlierResult:
    """Three methods, and the disagreement between them is reported.

    A point flagged by only one test usually says more about that test than
    about the business, so `method_dependent_indices` is surfaced separately
    rather than being folded into a single "outliers" list.

    The modified z-score uses the median and MAD, so unlike the plain z-score
    it is not dragged around by the very outliers it is looking for.
    """
    arr, _ = _clean(values)
    n = int(arr.size)
    if n < 4:
        raise ValueError("outliers() needs at least 4 observations")

    q1, q3 = (float(v) for v in np.percentile(arr, [25, 75]))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    lo, hi = float(lower), float(upper)
    iqr_idx = [i for i, v in enumerate(arr) if v < lo or v > hi]

    std = float(np.std(arr, ddof=1))
    mean = float(np.mean(arr))
    z_idx = (
        [i for i, v in enumerate(arr) if abs((float(v) - mean) / std) > z_threshold]
        if std > 0
        else []
    )

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad > 0:
        mod_idx = [
            i for i, v in enumerate(arr) if abs(0.6745 * (float(v) - median) / mad) > 3.5
        ]
    else:
        mod_idx = []

    sets = [set(iqr_idx), set(z_idx), set(mod_idx)]
    consensus = sorted(sets[0] & sets[1] & sets[2])
    union = sets[0] | sets[1] | sets[2]
    method_dependent = sorted(i for i in union if sum(1 for s in sets if i in s) == 1)

    return OutlierResult(
        n=n,
        iqr_indices=iqr_idx,
        zscore_indices=z_idx,
        modified_zscore_indices=mod_idx,
        consensus_indices=consensus,
        method_dependent_indices=method_dependent,
        lower_fence=lower,
        upper_fence=upper,
    )


# --------------------------------------------------------------------------- #
# Seasonality
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SeasonalityResult:
    period: int
    cycles_available: float
    seasonal_strength: float
    detected: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seasonality(values: Sequence[float], *, period: int) -> SeasonalityResult:
    """Additive seasonal strength via a centred moving-average decomposition.

    Requires two full cycles. Below that, any "seasonality" found is an
    artefact of the window, so the function says so instead of returning a
    number that would be quoted as fact.
    """
    arr, _ = _clean(values)
    if period < 2:
        raise ValueError("period must be at least 2")

    cycles = arr.size / period
    if cycles < 2:
        return SeasonalityResult(
            period=period,
            cycles_available=float(cycles),
            seasonal_strength=0.0,
            detected=False,
            note=(
                f"Only {cycles:.1f} cycles of data at period {period}; at least 2 are "
                "required before seasonality can be distinguished from noise."
            ),
        )

    kernel = np.ones(period) / period
    trend_component = np.convolve(arr, kernel, mode="same")
    detrended = arr - trend_component

    seasonal = np.zeros(period)
    for phase in range(period):
        seasonal[phase] = float(np.mean(detrended[phase::period]))
    seasonal -= seasonal.mean()

    seasonal_full = np.resize(seasonal, arr.size)
    residual = detrended - seasonal_full

    var_resid = float(np.var(residual, ddof=1))
    var_total = float(np.var(detrended, ddof=1))
    strength = 0.0 if var_total == 0 else max(0.0, 1.0 - var_resid / var_total)

    return SeasonalityResult(
        period=period,
        cycles_available=float(cycles),
        seasonal_strength=strength,
        detected=strength >= 0.3,
        note=(f"Seasonal strength {strength:.2f} at period {period} over {cycles:.1f} cycles."),
    )
