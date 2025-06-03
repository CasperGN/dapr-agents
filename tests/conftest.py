import os
import pytest
import logging


# Configure pytest
def pytest_configure(config):
    """Configure pytest for dapr-agents tests."""
    # Set up basic logging for tests
    logging.basicConfig(level=logging.INFO)

    # Add markers
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line(
        "markers", "integration: mark a test as an integration test"
    )
    config.addinivalue_line("markers", "telemetry: mark a test as related to telemetry")


# Global fixtures can be defined here
@pytest.fixture(scope="session")
def base_test_dir():
    """Return the base directory for tests."""
    return os.path.dirname(os.path.abspath(__file__))
