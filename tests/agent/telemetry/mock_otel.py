"""Mock version of the dapr_agents.agent.telemetry.otel module for testing."""

import os
import uuid
import socket
import platform
import logging
from typing import Dict, Any, Optional, Union

# Constants for environment variables
ENV_SERVICE_NAME = "OTEL_SERVICE_NAME"
ENV_DEPLOYMENT_ENVIRONMENT = "OTEL_DEPLOYMENT_ENVIRONMENT"
ENV_SERVICE_VERSION = "OTEL_SERVICE_VERSION"

# Constants for resource attributes
SERVICE_NAME = "service.name"
SERVICE_NAMESPACE = "service.namespace"
SERVICE_INSTANCE_ID = "service.instance.id"
SERVICE_VERSION = "service.version"
DEPLOYMENT_ENVIRONMENT = "deployment.environment"

# Create mock classes for OpenTelemetry components
class Context:
    """Mock for OpenTelemetry Context."""
    def __init__(self):
        pass

class Resource:
    """Mock for OpenTelemetry Resource."""
    def __init__(self, attributes=None):
        self.attributes = attributes or {}
    
    @classmethod
    def create(cls, attributes=None):
        return cls(attributes)

# Mock setter functions for providers
def set_tracer_provider(provider):
    """Mock for set_tracer_provider."""
    pass

def set_meter_provider(provider):
    """Mock for set_meter_provider."""
    pass

def set_logger_provider(provider):
    """Mock for set_logger_provider."""
    pass

class TracerProvider:
    """Mock for OpenTelemetry TracerProvider."""
    def __init__(self, resource=None):
        self.resource = resource
    
    def add_span_processor(self, processor):
        pass

class MeterProvider:
    """Mock for OpenTelemetry MeterProvider."""
    def __init__(self, resource=None, metric_readers=None):
        self.resource = resource
        self.metric_readers = metric_readers or []

class LoggingHandler(logging.Handler):
    """Mock for OpenTelemetry LoggingHandler."""
    def __init__(self):
        super().__init__()
    
    def emit(self, record):
        pass

class LoggerProvider:
    """Mock for OpenTelemetry LoggerProvider."""
    def __init__(self, resource=None):
        self.resource = resource
    
    def add_log_record_processor(self, processor):
        pass
    
    def get_logger(self, name, version=None):
        pass

# Mock propagator for context extraction/injection
class _MockPropagator:
    def inject(self, carrier, context=None):
        pass
    
    def extract(self, carrier, context=None):
        return Context()

_propagator = _MockPropagator()

class DaprAgentsOTel:
    """Mock implementation of DaprAgentsOTel for testing."""
    
    def __init__(self, service_name: str = "", otlp_endpoint: str = ""):
        self.service_name = service_name or os.environ.get(ENV_SERVICE_NAME, "dapr-agents")
        self.deployment_environment = os.environ.get(ENV_DEPLOYMENT_ENVIRONMENT, "development")
        self.otlp_endpoint = otlp_endpoint
        self.setup_resources()
    
    def setup_resources(self):
        """Set up OpenTelemetry resources with attributes."""
        self._resource = Resource.create(
            attributes={
                SERVICE_NAME: self.service_name,
                SERVICE_INSTANCE_ID: str(uuid.uuid4()),
                "host.name": socket.gethostname(),
                "os.type": platform.system().lower(),
                "process.runtime.name": "python",
                "process.runtime.version": platform.python_version(),
                DEPLOYMENT_ENVIRONMENT: self.deployment_environment,
                SERVICE_VERSION: os.environ.get(ENV_SERVICE_VERSION, "dev"),
            }
        )
    
    def _endpoint_validator(self, endpoint: str, signal_type: str) -> str:
        """Validate and format OTLP endpoint URL."""
        if not endpoint:
            raise ValueError(f"OTLP endpoint is required for {signal_type} exporter")
        
        # Add http:// if no protocol specified
        if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
            endpoint = f"http://{endpoint}"
        
        # Add path if not present
        if not endpoint.endswith(f"/v1/{signal_type}"):
            if endpoint.endswith("/"):
                # With trailing slash
                endpoint = f"{endpoint}v1/{signal_type}"
            else:
                # Without trailing slash - this is the line that's hard to cover
                endpoint = f"{endpoint}/v1/{signal_type}"
        
        return endpoint
    
    def create_and_instrument_tracer_provider(self, otlp_endpoint: str = "") -> TracerProvider:
        """Create and configure the tracer provider."""
        endpoint = otlp_endpoint or self.otlp_endpoint
        if endpoint:
            endpoint = self._endpoint_validator(endpoint, "traces")
            logging.info(f"Configured OTLP span exporter with endpoint: {endpoint}")
        
        provider = TracerProvider(resource=self._resource)
        # Call the set_tracer_provider function
        set_tracer_provider(provider)
        return provider
    
    def create_and_instrument_meter_provider(self, otlp_endpoint: str = "") -> MeterProvider:
        """Create and configure the meter provider."""
        endpoint = otlp_endpoint or self.otlp_endpoint
        if endpoint:
            endpoint = self._endpoint_validator(endpoint, "metrics")
            logging.info(f"Configured OTLP metric exporter with endpoint: {endpoint}")
        
        export_interval = int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", 60000))
        provider = MeterProvider(resource=self._resource)
        # Call set_meter_provider function
        set_meter_provider(provider)
        return provider
    
    def create_and_instrument_logging_provider(self, logger, otlp_endpoint: str = "") -> LoggerProvider:
        """Create and configure the logging provider."""
        endpoint = otlp_endpoint or self.otlp_endpoint
        if endpoint:
            endpoint = self._endpoint_validator(endpoint, "logs")
            logging.info(f"Configured OTLP log exporter with endpoint: {endpoint}")
        
        batch_size = int(os.environ.get("OTEL_LOGS_EXPORT_BATCH_SIZE", 512))
        provider = LoggerProvider(resource=self._resource)
        # Call set_logger_provider function
        set_logger_provider(provider)
        
        # Add a LoggingHandler to the logger - use a function to ensure it's called
        # This ensures that the mock will register as called
        if logger:
            # Directly construct the class so the mock constructor is called
            handler = LoggingHandler()
            logger.addHandler(handler)
            
        return provider

# Context extraction and restoration functions
def extract_otel_context() -> Dict[str, str]:
    """Extract the current OpenTelemetry context into a dictionary."""
    carrier = {}
    try:
        _propagator.inject(carrier)
        return carrier
    except Exception as e:
        logging.warning(f"Error extracting OpenTelemetry context: {e}")
        return {}

def restore_otel_context(context_data: Union[Dict[str, str], Context, Any]) -> Context:
    """Restore an OpenTelemetry context from a dictionary or Context object."""
    try:
        if isinstance(context_data, Context):
            return context_data
        
        if not context_data:
            # Ensure we call extract even with empty data
            return _propagator.extract({})
        
        return _propagator.extract(context_data)
    except Exception as e:
        logging.warning(f"Error restoring OpenTelemetry context: {e}")
        return Context()
