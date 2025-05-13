from logging import Logger
import os
from typing import Union

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
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider, Status, StatusCode
from opentelemetry.context.context import Context
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)


class DaprAgentsOTel:
    """
    OpenTelemetry configuration for Dapr agents.
    """

    def __init__(self, service_name: str = "", otlp_endpoint: str = ""):
        # Configure OpenTelemetry
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint

        self.setup_resources()

    def setup_resources(self):
        """
        Set up the resource for OpenTelemetry.
        """

        self._resource = Resource.create(
            attributes={
                SERVICE_NAME: str(self.service_name),
                "service.instance.id": str(uuid.uuid4()),
            }
        )

    def create_and_instrument_meter_provider(
        self,
        otlp_endpoint: str = "",
    ) -> MeterProvider:
        """
        Returns a `MeterProvider` that is configured to export metrics using the `PeriodicExportingMetricReader`
        which means that metrics are exported periodically in the background. The interval can be set by
        the environment variable `OTEL_METRIC_EXPORT_INTERVAL`. The default value is 60000ms (1 minute).

        Also sets the global OpenTelemetry meter provider to the returned meter provider.
        """

        # Ensure the endpoint is set correctly
        endpoint = self._endpoint_validator(
            endpoint=self.otlp_endpoint if otlp_endpoint == "" else otlp_endpoint,
            telemetry_type="metrics",
        )

        metric_exporter = OTLPMetricExporter(endpoint=str(endpoint))
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(
            resource=self._resource, metric_readers=[metric_reader]
        )
        set_meter_provider(meter_provider)
        return meter_provider

    def create_and_instrument_tracer_provider(
        self,
        otlp_endpoint: str = "",
    ) -> TracerProvider:
        """
        Returns a `TracerProvider` that is configured to export traces using the `BatchSpanProcessor`
        which means that traces are exported in batches. The batch size can be set by
        the environment variable `OTEL_TRACES_EXPORT_BATCH_SIZE`. The default value is 512.
        Also sets the global OpenTelemetry tracer provider to the returned tracer provider.
        """

        # Ensure the endpoint is set correctly
        endpoint = self._endpoint_validator(
            endpoint=self.otlp_endpoint if otlp_endpoint == "" else otlp_endpoint,
            telemetry_type="traces",
        )
        try:
            sampling_ratio = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))
        except ValueError:
            # There's no need to actually raise an error here, just set to 1.0
            logging.warning(
                "OTEL_TRACES_SAMPLER_ARG is not a valid float. Defaulting to 1.0."
            )
            sampling_ratio = 1.0

        sampler = TraceIdRatioBased(sampling_ratio / 1000)
        trace_exporter = OTLPSpanExporter(endpoint=str(endpoint))
        tracer_processor = BatchSpanProcessor(trace_exporter)
        tracer_provider = TracerProvider(resource=self._resource, sampler=sampler)
        tracer_provider.add_span_processor(tracer_processor)
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        trace.get_tracer_provider().add_span_processor(  # type: ignore
            SimpleSpanProcessor(ConsoleSpanExporter())
        )
        set_tracer_provider(tracer_provider)
        return tracer_provider

    def create_and_instrument_logging_provider(
        self,
        logger: Logger,
        otlp_endpoint: str = "",
    ) -> LoggerProvider:
        """
        Returns a `LoggingProvider` that is configured to export logs using the `BatchLogProcessor`
        which means that logs are exported in batches. The batch size can be set by
        the environment variable `OTEL_LOGS_EXPORT_BATCH_SIZE`. The default value is 512.
        Also sets the global OpenTelemetry logging provider to the returned logging provider.
        """

        # Ensure the endpoint is set correctly
        endpoint = self._endpoint_validator(
            endpoint=self.otlp_endpoint if otlp_endpoint == "" else otlp_endpoint,
            telemetry_type="logs",
        )

        log_exporter = OTLPLogExporter(endpoint=str(endpoint))
        logging_provider = LoggerProvider(resource=self._resource)
        logging_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(logging_provider)

        handler = LoggingHandler(logger_provider=logging_provider)
        logger.addHandler(handler)
        return logging_provider

    def _endpoint_validator(
        self,
        endpoint: str,
        telemetry_type: str,
    ) -> Union[str | Exception]:
        """
        Validates the endpoint and method.
        """

        if endpoint == "":
            raise ValueError(
                "OTLP endpoint must be set either in the environment variable OTEL_EXPORTER_OTLP_ENDPOINT or in the constructor."
            )
        if endpoint.startswith("https://"):
            raise NotImplementedError(
                "OTLP over HTTPS is not supported. Please use HTTP."
            )

        endpoint = (
            endpoint
            if endpoint.endswith(f"/v1/{telemetry_type}")
            else f"{endpoint}/v1/{telemetry_type}"
        )
        endpoint = endpoint if endpoint.startswith("http://") else f"http://{endpoint}"
        logger.info(f"Endpoint: {endpoint}")

        return endpoint


_propagator = TraceContextTextMapPropagator()


def async_span_decorator(name):
    """
    Decorator for OpenTelemetry spans in async functions with proper error handling.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            otel_context = kwargs.get("otel_context")

            if not otel_context:
                otel_context = context.get_current()

            tracer = getattr(self, "_tracer", None)
            if not tracer:
                logger.info("No tracer found, executing function without tracing.")
                return func(self, *args, **kwargs)

            with tracer.start_as_current_span(
                name, context=otel_context, end_on_exit=False
            ) as span:
                span.set_attribute("function.name", func.__name__)

                # Set the context here as context is either derived from otel_context or current context
                kwargs["otel_context"] = context.get_current()

                try:
                    result = await func(self, *args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def span_decorator(name):
    """Decorator that creates an OpenTelemetry span for a synchronous method."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            otel_context = kwargs.get("otel_context")

            if not otel_context:
                otel_context = context.get_current()

            tracer = getattr(self, "_tracer", None)
            if not tracer:
                logger.info("No tracer found, executing function without tracing.")
                return func(self, *args, **kwargs)

            with tracer.start_as_current_span(
                name, context=otel_context, end_on_exit=False
            ) as span:
                span.set_attribute("function.name", func.__name__)

                # Set the context here as context is either derived from otel_context or current context
                kwargs["otel_context"] = context.get_current()

                try:
                    result = func(self, *args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def restore_otel_context(otel_context: dict[str, str]) -> Context:
    """
    Restore OpenTelemetry context from a previously extracted context dictionary.
    Creates a fresh context to avoid token errors across async boundaries.

    Args:
        otel_context: Dictionary containing context information

    Returns:
        Context object that can be used with tracer.start_as_current_span()
    """
    return _propagator.extract(carrier=otel_context)


def extract_otel_context() -> dict[str, str]:
    """
    Extract current OpenTelemetry context for cross-boundary propagation.
    Returns a format that can be properly serialized by Dapr workflows.
    """
    otel_context: dict[str, str] = {}
    _propagator.inject(carrier=otel_context, context=context.get_current())

    return otel_context
