"""Chronos context construction, step selection, and saved-cache endpoints."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")


def test_context_keeps_missing_hours_as_nan_and_ends_at_legal_hour():
    from models.chronos_delayed_cache import CONTEXT, _contexts

    idx = pd.date_range("2024-01-01", periods=700, freq="h", tz="UTC")
    hourly = pd.Series(np.arange(700, dtype=float), index=idx).drop(idx[600])
    for delay in (1, 2, 6):
        grid, values, positions, end_pos = _contexts(hourly, delay)
        p = positions[-1]
        end = end_pos[p]
        window = values[end - CONTEXT + 1:end + 1]
        assert len(window) == CONTEXT
        assert grid[end] == grid[p] - pd.Timedelta(hours=1 + delay)
        assert np.isnan(window).sum() == 1
        assert np.isnan(window[600 - (end - CONTEXT + 1)])
        assert window[-1] == float(end)


def test_summarize_reads_the_requested_step_and_uses_lower_median():
    from models.pes2027.chronos_cache import summarize

    samples = torch.arange(1, 21, dtype=torch.float32)
    paths = torch.stack([torch.full((20,), -1e6), samples, torch.full((20,), 1e6)], dim=1)[None]
    out = summarize(paths, 1)[0]
    assert out[0] == 10.0  # lower median of 1..20
    assert out[1] == pytest.approx(np.quantile(np.arange(1, 21), 0.1))
    assert out[2] == pytest.approx(np.quantile(np.arange(1, 21), 0.9))
    assert out[3] == pytest.approx(10.5)
    assert out[4] == 0.0
    spikes = summarize(torch.full((1, 20, 2), 250.0), 1)[0]
    assert spikes[4] == 1.0
    edge = summarize(torch.full((1, 20, 2), 200.0), 1)[0]
    assert edge[4] == 0.0  # the exceedance share is strict


def test_summarize_matches_original_cache_arithmetic():
    from models.pes2027.chronos_cache import summarize

    g = torch.Generator().manual_seed(0)
    paths = torch.randn(4, 20, 3, generator=g) * 100
    delay = 2
    at = paths[:, :, delay].float()
    q10, q90 = torch.quantile(at, torch.tensor([0.1, 0.9]), dim=1)
    expected = np.column_stack([at.median(dim=1).values.numpy(), q10.numpy(), q90.numpy(),
                                at.mean(dim=1).numpy(), (at > 200).float().mean(dim=1).numpy()])
    np.testing.assert_array_equal(summarize(paths, delay), expected)


@pytest.mark.skipif(not (ROOT / "runs" / "pit_verified" / "chronos" / "delay_1h.parquet").exists(),
                    reason="saved caches not present")
@pytest.mark.parametrize("delay", [1, 2, 3, 6])
def test_every_saved_cache_row_ends_at_t_minus_one_plus_delay(delay):
    cache = pd.read_parquet(ROOT / "runs" / "pit_verified" / "chronos" / f"delay_{delay}h.parquet",
                            columns=["latest_input_timestamp"])
    latest = pd.DatetimeIndex(cache["latest_input_timestamp"])
    latest = latest.tz_localize("UTC") if latest.tz is None else latest
    assert (latest == cache.index - pd.Timedelta(hours=1 + delay)).all()
