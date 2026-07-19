"""Regression tests for phoenix_ml/__init__.py's package-level setup.

Covers the Windows MAX_PATH fix: `import phoenix_ml` used to crash on a
fresh install whose path pushed dcor's numba JIT cache filename past the
260-character Windows limit (see tests/ISSUES.md, resolved 2026-07-19).
"""
import importlib
import os

import pytest

import phoenix_ml


def _reload_phoenix_ml():
    return importlib.reload(phoenix_ml)


@pytest.mark.skipif(os.name != "nt", reason="NUMBA_CACHE_DIR redirect is Windows-only")
def test_numba_cache_dir_defaults_to_short_path_on_windows(monkeypatch):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    _reload_phoenix_ml()
    assert os.environ["NUMBA_CACHE_DIR"] == r"C:\phoenix_ml_numba_cache"


@pytest.mark.skipif(os.name != "nt", reason="NUMBA_CACHE_DIR redirect is Windows-only")
def test_numba_cache_dir_respects_a_user_supplied_value(monkeypatch):
    monkeypatch.setenv("NUMBA_CACHE_DIR", r"D:\my_own_cache")
    _reload_phoenix_ml()
    assert os.environ["NUMBA_CACHE_DIR"] == r"D:\my_own_cache"


@pytest.mark.skipif(os.name == "nt", reason="checks the redirect is skipped off Windows")
def test_numba_cache_dir_left_untouched_off_windows(monkeypatch):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    _reload_phoenix_ml()
    assert "NUMBA_CACHE_DIR" not in os.environ


def test_chosen_cache_dir_leaves_safety_margin_under_windows_max_path():
    chosen = r"C:\phoenix_ml_numba_cache"
    # Fixed overhead numba/dcor add on top of NUMBA_CACHE_DIR, confirmed against
    # a real failing install path: a 45-char `dcor_<hash>` subdirectory plus the
    # longest known cached filename (_fast_dcov_mergesort's inner function, 163
    # chars). See phoenix_ml/__init__.py's comment for the full derivation.
    fixed_overhead = 45 + 163
    assert len(chosen) + fixed_overhead < 260
