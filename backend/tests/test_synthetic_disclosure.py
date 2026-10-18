"""Guards on the synthetic-data disclosure contract.

These tests are the enforcement mechanism described in SYNTHETIC_DATA.md.
They are deliberately strict: it must not be possible to ship synthetic data
without a matching declaration.
"""
from __future__ import annotations

import pytest

from app.ingest import ADAPTERS
from app.ingest.base import (
    SYNTHETIC_DISCLOSURE_FILE,
    DisclosureError,
    _disclosed_source_names,
    assert_disclosed,
)


def test_disclosure_file_exists():
    assert SYNTHETIC_DISCLOSURE_FILE.exists(), (
        "SYNTHETIC_DATA.md is missing. It is required: the ingestion layer "
        "refuses to write synthetic rows without it."
    )


def test_every_synthetic_adapter_is_disclosed():
    """No adapter may produce synthetic data without declaring itself.

    This is the test that stops the disclosure file drifting out of date as
    new adapters are added.
    """
    disclosed = _disclosed_source_names()
    undisclosed = [
        cls.name
        for cls in ADAPTERS.values()
        if getattr(cls, "produces_synthetic", False) and cls.name not in disclosed
    ]
    assert not undisclosed, (
        f"These adapters produce synthetic data but are not declared in "
        f"SYNTHETIC_DATA.md: {undisclosed}. Add a line of the form "
        f"`source_name: <name>` for each."
    )


def test_undisclosed_synthetic_source_is_rejected():
    """The guard must actually fire, not just exist."""
    with pytest.raises(DisclosureError, match="not declared"):
        assert_disclosed("a_source_that_was_never_disclosed")


def test_disclosed_source_passes_when_declared(tmp_path, monkeypatch):
    """A properly declared source is accepted."""
    import app.ingest.base as base

    fake = tmp_path / "SYNTHETIC_DATA.md"
    fake.write_text(
        "# Disclosure\n\nSome prose.\n\n- `source_name: demo_synthetic_series`\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(base, "SYNTHETIC_DISCLOSURE_FILE", fake)

    base.assert_disclosed("demo_synthetic_series")  # must not raise

    with pytest.raises(DisclosureError):
        base.assert_disclosed("some_other_source")


def test_real_adapters_are_not_flagged_synthetic():
    """Adapters reading genuine sources must not claim to be synthetic."""
    for key, cls in ADAPTERS.items():
        if key in {"paradip"}:
            assert cls.produces_synthetic is False, (
                f"{key} reads a real published source and must not be flagged synthetic"
            )


def test_provenance_carries_required_fields():
    """Every adapter must emit a complete provenance record."""
    for cls in ADAPTERS.values():
        adapter = cls()
        try:
            record = adapter.provenance()
        finally:
            adapter.close()

        assert record["source_name"] == cls.name
        assert "retrieved_at" in record
        assert "is_synthetic" in record
        assert isinstance(record["is_synthetic"], bool)
