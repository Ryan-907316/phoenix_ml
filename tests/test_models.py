"""Tests for models.py — the user-editable model/hyperparameter-space registry.

models_dict and param_spaces are two hand-synced dictionaries with nothing
else enforcing it; a model added to one but not the other used to fail later
with a confusing KeyError deep inside HPO instead of a clear message at
import time.
"""
import pytest

from phoenix_ml.models import _check_dicts_in_sync, models_dict, param_spaces


def test_real_registry_is_in_sync():
    # Locks in the actual current state of the file — must not raise.
    _check_dicts_in_sync(models_dict, param_spaces)


def test_model_missing_its_param_space_is_named_in_the_error():
    models = {"A": object(), "B": object()}
    spaces = {"A": {}}
    with pytest.raises(AssertionError, match=r"missing from param_spaces: \['B'\]"):
        _check_dicts_in_sync(models, spaces)


def test_param_space_missing_its_model_is_named_in_the_error():
    models = {"A": object()}
    spaces = {"A": {}, "B": {}}
    with pytest.raises(AssertionError, match=r"missing from models_dict: \['B'\]"):
        _check_dicts_in_sync(models, spaces)
