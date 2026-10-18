"""The most important test in the repository.

A lookahead leak does not make a model slightly optimistic. It makes every
backtest number in the platform a lie, and it is invisible from the output --
a leaking model looks *excellent*. These tests assert the causality rules
structurally rather than trusting the panel builder to have got them right.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml import features, panel


def _series(start: date, days: int, base: float = 10.0) -> pd.Series:
    index = pd.date_range(pd.Timestamp(start), periods=days, freq="D")
    values = base + np.sin(np.arange(days) / 12.0) + np.arange(days) * 0.01
    return pd.Series(values, index=index)


def _panel(days: int = 400, target: str = "TGT") -> pd.DataFrame:
    start = date(2024, 1, 1)
    frame = pd.DataFrame(index=pd.date_range(pd.Timestamp(start), periods=days, freq="D"))
    frame[target] = _series(start, days).to_numpy()
    frame["COAL_AU_NEWC"] = _series(start, days, base=120.0).to_numpy()
    frame["COAL_ZA_RB"] = _series(start, days, base=95.0).to_numpy()
    frame["CRUDE_BRENT"] = _series(start, days, base=80.0).to_numpy()
    frame["queue_inprt_waiting"] = np.linspace(5, 25, days)
    frame.index.name = "date"
    return frame


def test_every_feature_is_lagged_off_its_own_date():
    """A feature row dated t must not equal a driver observed on t.

    ``build_features`` shifts the whole design by one day. If that shift is
    ever removed, the level feature would equal the same-day observation and
    the model would be reading the present.
    """
    frame = _panel()
    design = features.build_features(frame, "TGT", lag_days=1)

    for stamp in frame.index[10:20]:
        same_day = frame.loc[stamp, "TGT"]
        previous = frame.loc[stamp - pd.Timedelta(days=1), "TGT"]
        assert design.loc[stamp, "tgt_level"] == pytest.approx(previous)
        assert design.loc[stamp, "tgt_level"] != pytest.approx(same_day)


def test_calendar_features_are_not_lagged():
    """The day of the year on date t *is* known on date t.

    Lagging it would blur the seasonality for no causal benefit, so the
    exemption is deliberate and is asserted here so nobody "fixes" it.
    """
    frame = _panel()
    design = features.build_features(frame, "TGT")
    for stamp in frame.index[5:10]:
        expected = np.sin(2.0 * np.pi * stamp.dayofyear / 365.25)
        assert design.loc[stamp, "cal_sin1"] == pytest.approx(expected)


def test_no_feature_column_is_fabricated_where_no_data_exists():
    """An all-NaN region must stay NaN, not become zero.

    pandas sums an all-NaN row to 0.0 by default, which would silently put a
    "queue was empty" reading on every date before the berthing archive
    begins. That is a fabricated observation wearing the clothes of a real one.
    """
    frame = _panel()
    frame.loc[frame.index[:200], "queue_inprt_waiting"] = np.nan

    design = features.build_features(frame, "TGT")
    assert "queue_waiting_total" in design.columns
    early = design["queue_waiting_total"].iloc[:150]
    assert early.isna().all(), "queue total was fabricated where no snapshots exist"


def test_horizon_target_has_no_label_for_the_final_rows():
    """The last h rows must have no label; that is what stops training on the future."""
    frame = _panel(days=100)
    label = panel.horizon_target(frame, "TGT", horizon_days=14)
    assert label.iloc[-14:].isna().all()
    assert label.notna().iloc[:80].all()
    # And the label at t really is the value at t+h.
    assert label.iloc[0] == pytest.approx(frame["TGT"].iloc[14])


def test_monthly_series_are_lagged_by_publication_delay():
    """A Pink Sheet figure stamped 1 August is not knowable on 1 August.

    It is published well into September, so joining on the stamp would hand
    the model a month of hindsight on the strongest driver it has.
    """
    assert panel.MONTHLY_PUBLICATION_LAG_DAYS >= 30, (
        "monthly commodity data must be lagged by at least a month; the World "
        "Bank publishes each month's figure the following month"
    )


def test_assert_causal_rejects_a_panel_that_runs_past_as_of():
    frame = _panel(days=30)
    as_of = frame.index[-1].date()
    panel.assert_causal(frame, as_of)  # exact end is fine
    with pytest.raises(AssertionError):
        panel.assert_causal(frame, as_of - timedelta(days=1))


def test_log_returns_do_not_produce_infinities_on_zero_values():
    values = pd.Series(
        [10.0, 0.0, 5.0], index=pd.date_range("2026-01-01", periods=3, freq="D")
    )
    returns = panel.log_returns(values)
    assert not np.isinf(returns.to_numpy(dtype=float)).any()
