"""Conformal calibration and the scoring rules.

The coverage test is the one that matters. The timing engine samples scenarios
from these intervals and takes a CVaR over the tail, so an interval that says
80% and delivers 60% propagates straight into an over-confident recommendation.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.ml import conformal, evaluate


def test_conformity_score_is_negative_when_the_truth_is_inside():
    scores = conformal.conformity_scores(
        np.array([5.0]), np.array([4.0]), np.array([6.0])
    )
    assert scores[0] < 0


def test_conformity_score_measures_how_far_outside():
    scores = conformal.conformity_scores(
        np.array([9.0]), np.array([4.0]), np.array([6.0])
    )
    assert scores[0] == pytest.approx(3.0)


def test_widening_uses_the_finite_sample_order_statistic():
    """The (n+1)(1-alpha) rank is what makes the guarantee exact, not asymptotic."""
    scores = np.arange(1.0, 101.0)  # n = 100
    widening = conformal.conformal_widening(scores, alpha=0.2)
    # ceil(101 * 0.8) = 81st order statistic, which is the value 81.
    assert widening == pytest.approx(81.0)


def test_widening_is_conservative_when_calibration_is_too_small():
    """Under-covering is the failure that propagates into bad decisions."""
    scores = np.array([1.0, 2.0, 3.0])
    widening = conformal.conformal_widening(scores, alpha=0.01)
    assert widening == pytest.approx(3.0)


def test_calibrated_interval_achieves_nominal_coverage():
    """The bar the strategy document sets: 80% intervals covering 78-82%."""
    rng = np.random.default_rng(7)
    n = 4000

    # A deliberately mis-calibrated model: nominal 80% band that is far too
    # narrow, exactly the failure mode conformal prediction exists to repair.
    truth = rng.standard_normal(n) * 3.0
    predictions = {0.10: np.full(n, -1.0), 0.50: np.zeros(n), 0.90: np.full(n, 1.0)}

    naive = conformal.empirical_coverage(truth, predictions[0.10], predictions[0.90])
    assert naive < 0.5, "test setup is wrong: the raw band should under-cover badly"

    split = n // 2
    calibration = {level: values[:split] for level, values in predictions.items()}
    adjustment = conformal.calibrate(truth[:split], calibration)

    widened_low, widened_high = [], []
    for index in range(split, n):
        row = {level: float(values[index]) for level, values in predictions.items()}
        widened = conformal.apply(row, adjustment, floor=None)
        widened_low.append(widened[0.10])
        widened_high.append(widened[0.90])

    coverage = conformal.empirical_coverage(
        truth[split:], np.array(widened_low), np.array(widened_high)
    )
    assert 0.78 <= coverage <= 0.84, f"calibrated coverage was {coverage:.3f}"


def test_apply_repairs_crossed_quantiles():
    """Independently fitted quantile models can cross; that is not a distribution."""
    crossed = {0.10: 9.0, 0.50: 5.0, 0.90: 2.0}
    fixed = conformal.apply(crossed, conformal.ConformalAdjustment(), floor=None)
    values = [fixed[level] for level in sorted(fixed)]
    assert values == sorted(values)


def test_apply_floors_at_zero_because_freight_cannot_be_negative():
    adjustment = conformal.ConformalAdjustment(widening={0.2: 100.0})
    result = conformal.apply({0.10: 5.0, 0.50: 6.0, 0.90: 7.0}, adjustment, floor=0.0)
    assert min(result.values()) >= 0.0


# --- scoring rules ------------------------------------------------------------


def test_pinball_loss_penalises_asymmetrically():
    """A high quantile should be punished more for being too low."""
    under = evaluate.pinball_loss(y_true=10.0, predicted=8.0, level=0.9)
    over = evaluate.pinball_loss(y_true=10.0, predicted=12.0, level=0.9)
    assert under > over
    assert under == pytest.approx(0.9 * 2.0)
    assert over == pytest.approx(0.1 * 2.0)


def test_pinball_loss_is_zero_for_a_perfect_prediction():
    assert evaluate.pinball_loss(5.0, 5.0, 0.5) == pytest.approx(0.0)


def test_crps_is_lower_for_a_sharper_correctly_centred_distribution():
    truth = 10.0
    sharp = {0.1: 9.5, 0.5: 10.0, 0.9: 10.5}
    diffuse = {0.1: 5.0, 0.5: 10.0, 0.9: 15.0}
    assert evaluate.crps_from_quantiles(truth, sharp) < evaluate.crps_from_quantiles(
        truth, diffuse
    )


def test_pit_is_uniform_for_a_correctly_specified_distribution():
    """A flat PIT histogram is the signature of calibration."""
    rng = np.random.default_rng(11)
    levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    from scipy.stats import norm

    quantiles = {level: float(norm.ppf(level)) for level in levels}
    pits = [evaluate.pit_value(float(rng.standard_normal()), quantiles) for _ in range(4000)]
    histogram, _ = np.histogram(pits, bins=5, range=(0.0, 1.0))
    share = histogram / histogram.sum()
    assert share.max() < 0.32, f"PIT is not close to uniform: {share}"


def test_mase_is_one_when_the_forecast_is_the_naive_forecast():
    import pandas as pd

    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    scale = evaluate.mase_scale(values, season_days=1)
    assert scale == pytest.approx(1.0)


def test_benchmark_against_synthetic_data_is_refused():
    """SYNTHETIC_DATA.md: beating a curve we generated ourselves proves nothing."""
    # Self-referential baselines stay legal on a synthetic target.
    evaluate.assert_benchmark_allowed("random_walk", "SYN_CDFI_TABONEO_GUANGZHOU_50K")

    with pytest.raises(evaluate.BenchmarkUnavailable):
        evaluate.assert_benchmark_allowed(
            "SYN_FFA_CURVE", "SYN_CDFI_TABONEO_GUANGZHOU_50K"
        )
