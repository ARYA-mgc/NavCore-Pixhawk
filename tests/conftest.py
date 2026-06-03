#!/usr/bin/env python3
"""Shared fixtures and markers for the NavCore test suite."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
