"""
Pytest Configuration
====================
Shared fixtures and configuration for all tests.
"""

import pytest
import os
from unittest.mock import patch


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests that require real API calls (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(autouse=False)
def mock_api_key():
    """
    Optionally mock API key for tests that need it
    (not autouse so tests can opt-in)
    """
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-123'}):
        yield

