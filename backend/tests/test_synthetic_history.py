"""Properties the disclosed synthetic freight history must have.

This is the only generator in the platform, so its claims are worth asserting
rather than trusting. SYNTHETIC_DATA.md promises three things about it: the
level is real, the cycle is driven by real data, and the path is reproducible.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from app.ingest import synthetic_history as gen


def _calendar(days: int) -> list[date]:
    start = date(2020, 1, 1)
    return [start + timedelta(days=i) for i in range(days)]


def _driver(days: int) -> np.ndarray:
    monthly = [
        (date(2020, 1, 1) + timedelta(days=30 * i), 80.0 + 20.0 * np.sin(i / 6.0))
        for i in range(days // 30 + 2)
    ]
    return gen._daily_driver(monthly, _calendar(days)[0], _calendar(days)[-1])


def test_path_ends_exactly_on_the_real_anchor():
    """The level is not invented: it is pinned to the real CDFI assessment."""
    days = 900
    calendar = _calendar(days)
    driver = _driver(days)
    path = gen.generate_path("SYN_TEST", anchor_value=12.66, driver=driver, calendar=calendar)
    assert path[-1] == pytest.approx(12.66, abs=1e-3)


def test_generation_is_reproducible_for_the_same_series():
    """A disclosed dataset that changes on every run cannot be audited."""
    days = 400
    calendar, driver = _calendar(days), _driver(days)
    first = gen.generate_path("SYN_TEST", 10.0, driver, calendar)
    second = gen.generate_path("SYN_TEST", 10.0, driver, calendar)
    assert first == second


def test_different_routes_get_different_paths():
    days = 400
    calendar, driver = _calendar(days), _driver(days)
    a = gen.generate_path("SYN_ROUTE_A", 10.0, driver, calendar)
    b = gen.generate_path("SYN_ROUTE_B", 10.0, driver, calendar)
    assert a != b


def test_path_is_strictly_positive():
    """Freight cannot be negative, and a lognormal construction must guarantee it."""
    days = 1500
    path = gen.generate_path("SYN_TEST", 9.0, _driver(days), _calendar(days))
    assert min(path) > 0.0


def test_realised_volatility_matches_the_documented_target():
    """SYNTHETIC_DATA.md states 55% annualised; it must actually be that.

    Volatility is the property the GARCH model and the MARKET_VOL alert are
    tested against, so if the generator quietly produced a placid series those
    components would be passing vacuously.
    """
    days = 2500
    path = np.array(gen.generate_path("SYN_TEST", 12.0, _driver(days), _calendar(days)))
    returns = np.diff(np.log(path))
    annualised = returns.std() * np.sqrt(gen.TRADING_DAYS)
    assert 0.40 <= annualised <= 0.70, f"annualised vol was {annualised:.1%}"


def test_volatility_clusters_rather_than_being_homoscedastic():
    """Without clustering the volatility model would have nothing to detect."""
    days = 2500
    path = np.array(gen.generate_path("SYN_TEST", 12.0, _driver(days), _calendar(days)))
    squared = np.diff(np.log(path)) ** 2
    autocorrelation = np.corrcoef(squared[:-1], squared[1:])[0, 1]
    assert autocorrelation > 0.05, f"no volatility clustering: acf {autocorrelation:.3f}"


def test_synthetic_codes_are_namespaced_apart_from_real_ones():
    """Separate namespaces are what keep the provenance badge unambiguous."""
    assert gen.synthetic_code("CDFI_TABONEO_GUANGZHOU_50K").startswith("SYN_")
    assert gen.synthetic_code("X") != "X"


def test_driver_interpolation_refuses_a_degenerate_series():
    with pytest.raises(ValueError):
        gen._daily_driver([(date(2020, 1, 1), 50.0)], date(2020, 1, 1), date(2020, 6, 1))


def test_adapter_refuses_to_run_without_a_real_anchor(tmp_path):
    """It will not invent a level as well as a path.

    If the real CDFI feed has never run, the generator must skip rather than
    pick a plausible-looking number out of the air.
    """
    from app.ingest.synthetic_history import SyntheticFreightHistoryAdapter

    class NoAnchorSession:
        def execute(self, *_args, **_kwargs):
            class Result:
                def all(self_inner):
                    # A driver series exists...
                    return [(date(2020, 1, 1), 80.0), (date(2020, 2, 1), 82.0)]

                def first(self_inner):
                    # ...but no real CDFI assessment does.
                    return None

            return Result()

    adapter = SyntheticFreightHistoryAdapter()
    result = adapter.run(NoAnchorSession())
    assert result.status == "skipped"
    assert "anchor" in result.message
