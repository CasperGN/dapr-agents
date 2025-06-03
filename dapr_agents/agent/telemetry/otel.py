from logging import Logger
import os
import platform
import socket
import sys
from typing import Any, Dict, Optional, Union

import functools
import logging
import uuid

from opentelemetry import trace, context
from opentelemetry._logs import set_logger_provider
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, DEPLOYMENT_ENVIRONMENT
from opentelemetry.sdk.resources import SERVICE_VERSION, SERVICE_INSTANCE_ID
from opentelemetry.sdk.resources import PROCESS_RUNTIME_NAME, PROCESS_RUNTIME_VERSION
from opentelemetry.sdk.resources import HOST_NAME, OS_TYPE, OS_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider, Status, StatusCode
from opentelemetry.context.context import Context
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.trace.propagation import baggage, composite

logger = logging.getLogger(__name__)

# Define environment variable constants
ENV_SERVICE_NAME = "OTEL_SERVICE_NAME"
ENV_DEPLOYMENT_ENVIRONMENT = "OTEL_DEPLOYMENT_ENVIRONMENT"
ENV_SERVICE_VERSION = "OTEL_SERVICE_VERSION"
ENV_TRACE_EXPORTER = "OTEL_TRACES_EXPORTER"
ENV_METRIC_EXPORTER = "OTEL_METRICS_EXPORTER"
ENV_LOG_EXPORTER = "OTEL_LOGS_EXPORTER"


class DaprAgentsOTel:
    """
    OpenTelemetry configuration for Dapr agents.
    """

    def __init__(self, service_name: str = "", otlp_endpoint: str = ""):
        # Configure OpenTelemetry
        self.service_name = service_name or os.getenv(ENV_SERVICE_NAME, "dapr-agents")
        self.otlp_endpoint = otlp_endpoint
        self.deployment_environment = os.getenv(ENV_DEPLOYMENT_ENVIRONMENT, "development")
        self.service_version = os.getenv(ENV_SERVICE_VERSION, "unknown")
        
        # Exporters enabled by default
        self.traces_enabled = os.getenv(ENV_TRACE_EXPORTER, "otlp") != "none"
        self.metrics_enabled = os.getenv(ENV_METRIC_EXPORTER, "otlp") != "none"
        self.logs_enabled = os.getenv(ENV_LOG_EXPORTER, "otlp") != "none"
        
        # Initialize resources
        self.setup_resources()
        
        # Set up composite propagator for better interoperability
        self._setup_propagators()

    def setup_resources(self):
        """
        Set up the resource for OpenTelemetry with enhanced attributes.
        Includes service information, host details, and runtime information.
        """
        # Get hostname
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "unknown"
            
        # Create resource with detailed attributes
        self._resource = Resource.create(
            attributes={
                # Service information
                SERVICE_NAME: str(self.service_name),
                SERVICE_INSTANCE_ID: str(uuid.uuid4()),
                SERVICE_VERSION: self.service_version,
                DEPLOYMENT_ENVIRONMENT: self.deployment_environment,
                
                # Host information
                HOST_NAME: hostname,
                OS_TYPE: platform.system(),
                OS_VERSION: platform.version(),
                
                # Runtime information
                PROCESS_RUNTIME_NAME: "python",
                PROCESS_RUNTIME_VERSION: platform.python_version(),
                
                # Additional Dapr-specific attributes
                "dapr.agents.version": self.service_version,
            }
        )

    def create_and_instrument_meter_provider(
        self,
        otlp_endpoint: str = "",
    ) -> MeterProvider:
        """
        Returns a `MeterProvider` that is configured to export metrics using the `PeriodicExportingMetricReader`.
        
        Features:
        - Configurable export interval via OTEL_METRIC_EXPORT_INTERVAL (default: 60000ms)
        - Sets the global OpenTelemetry meter provider
        - Supports both HTTP and HTTPS endpoints
        - Graceful error handling with console fallback
        
        Returns:
            Configured MeterProvider instance
        """
        if not self.metrics_enabled:
            logger.info("Metrics are disabled. Creating default meter provider.")
            meter_provider = MeterProvider(resource=self._resource)
            set_meter_provider(meter_provider)
            return meter_provider

        # Configure export interval (in milliseconds)
        export_interval_ms = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "60000"))
        
        # Ensure the endpoint is set correctly
        try:
            endpoint = self._endpoint_validator(
                endpoint=self.otlp_endpoint if otlp_endpoint == "" else otlp_endpoint,
                telemetry_type="metrics",
            )
        except ValueError as e:
            logger.warning(f"Error configuring metrics endpoint: {e}. Using console exporter.")
            # Fall back to console exporter if endpoint is not configured
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            metric_exporter = ConsoleMetricExporter()
        else:
            # Configure the OTLP exporter
            metric_exporter = OTLPMetricExporter(endpoint=str(endpoint))

        # Create and configure the metric reader with the configured interval
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=export_interval_ms
        )
        
        # Initialize meter provider
        meter_provider = MeterProvider(
            resource=self._resource, 
            metric_readers=[metric_reader]
        )
        set_meter_provider(meter_provider)
        logger.info(f"Initialized meter provider with export interval {export_interval_ms}ms")
        return meter_provider

    def create_and_instrument_tracer_provider(
        self,
        otlp_endpoint: str = "",
    ) -> TracerProvider:
        """
        Returns a `TracerProvider` that is configured to export traces using the `BatchSpanProcessor`.
        
        Features:
        - Uses ParentBased sampling with TraceIdRatioBased as root sampling strategy
        - Configurable sampling ratio via OTEL_TRACES_SAMPLER_ARG (default: 1.0)
        - Configurable batch export size via OTEL_TRACES_EXPORT_BATCH_SIZE (default: 512)
        - Sets the global OpenTelemetry tracer provider
        - Supports both HTTP and HTTPS endpoints
        
        Returns:
            Configured TracerProvider instance
        """
        if not self.traces_enabled:
            logger.info("Tracing is disabled. Creating no-op tracer provider.")
            tracer_provider = TracerProvider(sampler=TraceIdRatioBased(0))
            set_tracer_provider(tracer_provider)
            return tracer_provider

        # Ensure the endpoint is set correctly
        try:
            endpoint = self._endpoint_validator(
                endpoint=self.otlp_endpoint if otlp_endpoint == "" else otlp_endpoint,
                telemetry_type="traces",
            )
        except ValueError as e:
            logger.warning(f"Error configuring trace endpoint: {e}. Using console exporter.")
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            trace_exporter = ConsoleSpanExporter()
        else:
            # Configure the OTLP exporter
            trace_exporter = OTLPSpanExporter(endpoint=str(endpoint))
            
        # Configure sampling strategy
        try:
            sampling_ratio = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))
            if sampling_ratio > 1.0 or sampling_ratio < 0.0:
                logging.warning("OTEL_TRACES_SAMPLER_ARG must be between 0.0 and 1.0. Defaulting to 1.0.")
                sampling_ratio = 1.0
        except ValueError:
            logging.warning("OTEL_TRACES_SAMPLER_ARG is not a valid float. Defaulting to 1.0.")
            sampling_ratio = 1.0

        # Always sample if parent span is sampled, otherwise use probability sampling
        root_sampler = TraceIdRatioBased(sampling_ratio)
        sampler = ParentBased(root_sampler)

        # Configure batch processing
        max_export_batch_size = int(os.getenv("OTEL_TRACES_EXPORT_BATCH_SIZE", "512"))
        export_timeout_ms = int(os.getenv("OTEL_TRACES_EXPORT_TIMEOUT", "30000"))
        
        # Create span processor with optimized settings
        tracer_processor = BatchSpanProcessor(
            trace_exporter,
            max_export_batch_size=max_export_batch_size,
            export_timeout_millis=export_timeout_ms
        )
        
        # Initialize tracer provider
        tracer_provider = TracerProvider(resource=self._resource, sampler=sampler)
        tracer_provider.add_span_processor(tracer_processor)
        set_tracer_provider(tracer_provider)
        logger.info(f"Initialized tracer provider with sampling ratio {sampling_ratio}")
        return tracer_provider

    def create_and_instrument_logging_provider(
        self,
        logger: Logger,
        otlp_endpoint: str = "",
    ) -> LoggerProvider:
        """
        Returns a `LoggerProvider` that is configured to export logs using the `BatchLogProcessor`.
        
        Features:
        - Configurable batch size via OTEL_LOGS_EXPORT_BATCH_SIZE (default: 512)
        - Sets the global OpenTelemetry logging provider
        - Configures Python's standard logging to use OpenTelemetry
        - Supports both HTTP and HTTPS endpoints
        - Graceful error handling with console fallback
        
        Args:
            logger: Logger instance to add OpenTelemetry handler to
            otlp_endpoint: Optional custom endpoint for logs export
            
        Returns:
            Configured LoggerProvider instance
        """
        if not self.logs_enabled:
            logger.info("Logging telemetry is disabled. Creating default logger provider.")
            logging_provider = LoggerProvider(resource=self._resource)
            set_logger_provider(logging_provider)
            return logging_provider

        # Configure batch size
        batch_size = int(os.getenv("OTEL_LOGS_EXPORT_BATCH_SIZE", "512"))
        
        # Configure log level for OpenTelemetry logs
        log_level = os.getenv("OTEL_LOG_LEVEL", "INFO")
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        # Ensure the endpoint is set correctly
        try:
            endpoint = self._endpoint_validator(
                endpoint=self.otlp_endpoint if otlp_endpoint == "" else otlp_endpoint,
                telemetry_type="logs",
            )
        except ValueError as e:
            logger.warning(f"Error configuring logs endpoint: {e}. Using console exporter.")
            # Fall back to console exporter if endpoint is not configured
            from opentelemetry.sdk._logs.export import ConsoleLogExporter
            log_exporter = ConsoleLogExporter()
        else:
            # Configure the OTLP exporter
            log_exporter = OTLPLogExporter(endpoint=str(endpoint))

        # Create and configure the log processor with the configured batch size
        log_processor = BatchLogRecordProcessor(
            log_exporter, 
            max_export_batch_size=batch_size
        )
        
        # Initialize logger provider
        logging_provider = LoggerProvider(resource=self._resource)
        logging_provider.add_log_record_processor(log_processor)
        set_logger_provider(logging_provider)

        # Add OpenTelemetry handler to the logger
        handler = LoggingHandler(logger_provider=logging_provider)
        logger.addHandler(handler)
        
        logger.info(f"Initialized logger provider with batch size {batch_size}")
        return logging_provider

    def _setup_propagators(self):
        """
        Set up the composite propagator for distributed tracing context propagation.
        Combines W3C Trace Context and W3C Baggage propagation formats.
        """
        # Create a composite propagator that combines TraceContext and Baggage
        global _propagator
        _propagator = composite.CompositePropagator(
            [TraceContextTextMapPropagator(), W3CBaggagePropagator()]
        )
        logger.debug("Initialized composite context propagator")

    def _endpoint_validator(
        self,
        endpoint: str,
        telemetry_type: str,
    ) -> str:
        """
        Validates and formats the endpoint URL for the specified telemetry type.
        Supports both HTTP and HTTPS protocols.
        
        Args:
            endpoint: The base endpoint URL
            telemetry_type: Type of telemetry (traces, metrics, logs)
            
        Returns:
            Properly formatted endpoint URL
            
        Raises:
            ValueError: If the endpoint is not provided
        """
        if not endpoint:
            raise ValueError(
                "OTLP endpoint must be set either in the environment variable OTEL_EXPORTER_OTLP_ENDPOINT or in the constructor."
            )

        # Add telemetry type path if not present
        if not endpoint.endswith(f"/v1/{telemetry_type}"):
            endpoint = f"{endpoint}/v1/{telemetry_type}"
            
        # Add protocol if missing
        if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
            # Default to HTTP
            endpoint = f"http://{endpoint}"
            
        logger.info(f"OpenTelemetry {telemetry_type} endpoint: {endpoint}")
        return endpoint


# Default propagator - will be replaced by composite propagator in DaprAgentsOTel._setup_propagators
_propagator = TraceContextTextMapPropagator()


def restore_otel_context(otel_context: Union[Context, dict[str, str]]) -> Context:
    """
    Restore OpenTelemetry context from a previously extracted context dictionary.
    Creates a fresh context to avoid token errors across async boundaries.

    Args:
        otel_context: Dictionary containing context information

    Returns:
        Context object that can be used with tracer.start_as_current_span()
        
    Raises:
        ValueError: If otel_context is invalid
    """
    try:
        if isinstance(otel_context, Context):
            return _propagator.extract(carrier=extract_otel_context())
        if not otel_context:  # Handle empty context case
            logger.warning("Received empty OpenTelemetry context, creating new context")
            return Context()
        return _propagator.extract(carrier=otel_context)
    except Exception as e:
        logger.warning(f"Error restoring OpenTelemetry context: {e}. Creating new context.")
        return Context()


def extract_otel_context() -> dict[str, str]:
    """
    Extract current OpenTelemetry context for cross-boundary propagation.
    Returns a format that can be properly serialized by Dapr workflows.
    
    Returns:
        Dictionary containing the serialized context
    """
    try:
        otel_context: dict[str, str] = {}
        _propagator.inject(carrier=otel_context, context=context.get_current())
        return otel_context
    except Exception as e:
        logger.warning(f"Error extracting OpenTelemetry context: {e}. Returning empty context.")
        return {}
