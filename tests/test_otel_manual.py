#!/usr/bin/env python3
"""
Manual test script for OpenTelemetry implementation

This script verifies the key functionality of the enhanced OpenTelemetry implementation
without requiring pytest or additional test dependencies.

Run this script directly with: python3 tests/test_otel_manual.py
"""

import os
import sys
import logging
import uuid
from unittest import mock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dapr_agents.agent.telemetry.otel import (
    DaprAgentsOTel,
    extract_otel_context,
    restore_otel_context
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('otel_test')

def run_test(name, test_func):
    """Run a test function and report results"""
    logger.info(f"Running test: {name}")
    try:
        test_func()
        logger.info(f"✅ PASSED: {name}")
        return True
    except Exception as e:
        logger.error(f"❌ FAILED: {name} - {str(e)}")
        return False

def test_init_with_defaults():
    """Test initialization with default values"""
    otel = DaprAgentsOTel()
    assert otel.service_name == "dapr-agents", f"Expected service_name to be 'dapr-agents', got {otel.service_name}"
    assert otel.deployment_environment == "development", f"Expected deployment_environment to be 'development', got {otel.deployment_environment}"
    assert hasattr(otel, "_resource"), "_resource attribute not found"

def test_init_with_params():
    """Test initialization with custom parameters"""
    otel = DaprAgentsOTel(
        service_name="custom-service", 
        otlp_endpoint="http://custom-endpoint:4318"
    )
    assert otel.service_name == "custom-service", f"Expected service_name to be 'custom-service', got {otel.service_name}"
    assert otel.otlp_endpoint == "http://custom-endpoint:4318", f"Expected otlp_endpoint to be 'http://custom-endpoint:4318', got {otel.otlp_endpoint}"

def test_init_with_env_vars():
    """Test initialization with environment variables"""
    # Save original environment
    original_env = {}
    for key in ['OTEL_SERVICE_NAME', 'OTEL_DEPLOYMENT_ENVIRONMENT']:
        if key in os.environ:
            original_env[key] = os.environ[key]
    
    try:
        # Set test environment variables
        os.environ['OTEL_SERVICE_NAME'] = "test-service"
        os.environ['OTEL_DEPLOYMENT_ENVIRONMENT'] = "testing"
        
        # Create instance with environment variables
        otel = DaprAgentsOTel()
        assert otel.service_name == "test-service", f"Expected service_name to be 'test-service', got {otel.service_name}"
        assert otel.deployment_environment == "testing", f"Expected deployment_environment to be 'testing', got {otel.deployment_environment}"
    finally:
        # Restore original environment
        for key in ['OTEL_SERVICE_NAME', 'OTEL_DEPLOYMENT_ENVIRONMENT']:
            if key in original_env:
                os.environ[key] = original_env[key]
            elif key in os.environ:
                del os.environ[key]

def test_setup_resources():
    """Test resource setup with proper attributes"""
    otel = DaprAgentsOTel(service_name="test-service")
    resource = otel._resource
    assert resource is not None, "Resource should not be None"
    
    # Check resource attributes
    attributes = resource.attributes
    assert attributes.get("service.name") == "test-service", f"Expected service.name to be 'test-service', got {attributes.get('service.name')}"
    assert "service.instance.id" in attributes, "service.instance.id not found in attributes"
    assert "host.name" in attributes, "host.name not found in attributes"
    assert "os.type" in attributes, "os.type not found in attributes"

def test_endpoint_validator_http():
    """Test endpoint validator with HTTP URLs"""
    otel = DaprAgentsOTel()
    
    # Test with already properly formatted URL
    url = "http://localhost:4318/v1/traces"
    result = otel._endpoint_validator(url, "traces")
    assert result == url, f"Expected {url}, got {result}"
    
    # Test with URL missing protocol
    url_no_protocol = "localhost:4318"
    result = otel._endpoint_validator(url_no_protocol, "traces")
    assert result == "http://localhost:4318/v1/traces", f"Expected 'http://localhost:4318/v1/traces', got {result}"
    
    # Test with URL missing path
    url_no_path = "http://localhost:4318"
    result = otel._endpoint_validator(url_no_path, "metrics")
    assert result == "http://localhost:4318/v1/metrics", f"Expected 'http://localhost:4318/v1/metrics', got {result}"

def test_endpoint_validator_https():
    """Test endpoint validator with HTTPS URLs"""
    otel = DaprAgentsOTel()
    
    # Test with HTTPS URL
    url = "https://otel-collector.example.com:4318"
    result = otel._endpoint_validator(url, "logs")
    assert result == "https://otel-collector.example.com:4318/v1/logs", f"Expected 'https://otel-collector.example.com:4318/v1/logs', got {result}"
    
    # Test with HTTPS URL already having path
    url_with_path = "https://otel-collector.example.com:4318/v1/logs"
    result = otel._endpoint_validator(url_with_path, "logs")
    assert result == url_with_path, f"Expected {url_with_path}, got {result}"

def test_endpoint_validator_error():
    """Test endpoint validator error handling"""
    otel = DaprAgentsOTel()
    
    # Test with empty URL
    try:
        otel._endpoint_validator("", "traces")
        assert False, "Expected ValueError for empty URL"
    except ValueError:
        # This is expected
        pass

def test_extract_and_restore_context():
    """Test context extraction and restoration"""
    # Simple test for extract_otel_context
    context_dict = extract_otel_context()
    assert isinstance(context_dict, dict), f"Expected dict, got {type(context_dict)}"
    
    # Simple test for restore_otel_context
    context = restore_otel_context({})
    assert context is not None, "Context should not be None"

def run_all_tests():
    """Run all test functions"""
    tests = [
        ("Initialization with defaults", test_init_with_defaults),
        ("Initialization with parameters", test_init_with_params),
        ("Initialization with environment variables", test_init_with_env_vars),
        ("Resource setup", test_setup_resources),
        ("Endpoint validator - HTTP", test_endpoint_validator_http),
        ("Endpoint validator - HTTPS", test_endpoint_validator_https),
        ("Endpoint validator - Error handling", test_endpoint_validator_error),
        ("Context extraction and restoration", test_extract_and_restore_context),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        if run_test(name, test_func):
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTest Summary: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    logger.info("Starting manual tests for OpenTelemetry implementation")
    success = run_all_tests()
    sys.exit(0 if success else 1)
