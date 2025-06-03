"""Direct coverage test for mock_otel module."""

import pytest
import sys
import os
import inspect
from tests.agent.telemetry.mock_otel import DaprAgentsOTel


def test_endpoint_validator_direct():
    """Direct test of the endpoint validator to ensure 100% coverage."""
    # Create a direct instance of DaprAgentsOTel without any mocking
    otel = DaprAgentsOTel()

    # Let's use a different approach: import the module and examine/call the function directly
    import tests.agent.telemetry.mock_otel as mock_otel

    # Use introspection to get the source code and see what we're missing
    source_lines = inspect.getsourcelines(mock_otel.DaprAgentsOTel._endpoint_validator)[
        0
    ]
    print("\nEndpoint validator source code:")
    for i, line in enumerate(source_lines):
        print(f"{i + 1}: {line.strip()}")

    # Create an instance of the class to test the method
    otel = mock_otel.DaprAgentsOTel()

    # First, test URL without trailing slash
    url = "http://my-collector:4317"  # No trailing slash
    signal_type = "metrics"
    print(f"\nTesting URL without trailing slash: {url}")
    print(f"URL.endswith('/'): {url.endswith('/')}")
    print(f"URL.endswith('/v1/{signal_type}'): {url.endswith(f'/v1/{signal_type}')}")

    result = otel._endpoint_validator(url, signal_type)
    print(f"Result: {result}")
    assert result == f"{url}/v1/{signal_type}"

    # Second, test URL with trailing slash
    url_with_slash = "http://my-collector:4317/"  # With trailing slash
    print(f"\nTesting URL with trailing slash: {url_with_slash}")
    print(f"URL.endswith('/'): {url_with_slash.endswith('/')}")
    print(
        f"URL.endswith('/v1/{signal_type}'): {url_with_slash.endswith(f'/v1/{signal_type}')}"
    )

    result_with_slash = otel._endpoint_validator(url_with_slash, signal_type)
    print(f"Result: {result_with_slash}")
    assert result_with_slash == f"{url_with_slash}v1/{signal_type}"

    # Third, test URL that already has the path
    url_with_path = f"http://my-collector:4317/v1/{signal_type}"  # Already has path
    print(f"\nTesting URL with path already: {url_with_path}")
    print(
        f"URL.endswith('/v1/{signal_type}'): {url_with_path.endswith(f'/v1/{signal_type}')}"
    )

    result_with_path = otel._endpoint_validator(url_with_path, signal_type)
    print(f"Result: {result_with_path}")
    assert result_with_path == url_with_path
