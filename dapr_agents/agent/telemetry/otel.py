from logging import Logger
import os
import time
from typing import Any, Optional, Union

import functools
import logging
import uuid

from opentelemetry import trace
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
from opentelemetry.trace import set_tracer_provider
from opentelemetry.context.context import Context
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


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
            sampling_ratio = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "0.0"))
        except ValueError:
            # There's no need to actually raise an error here, just set to 1.0
            logging.warning(
                "OTEL_TRACES_SAMPLER_ARG is not a valid float. Defaulting to 0.0."
            )
            sampling_ratio = 0.0

        sampler = TraceIdRatioBased(sampling_ratio)
        trace_exporter = OTLPSpanExporter(endpoint=str(endpoint))
        tracer_processor = BatchSpanProcessor(trace_exporter)
        tracer_provider = TracerProvider(resource=self._resource, sampler=sampler)
        tracer_provider.add_span_processor(tracer_processor)
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

        return endpoint


_propagator = TraceContextTextMapPropagator()


def llm_span_decorator(name="llm.generate"):
    """
    Decorator for tracing LLM API calls with detailed metrics.

    Args:
        name (str): Name for the span. Defaults to "llm.generate".
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract OpenTelemetry context if provided
            otel_context = kwargs.get("otel_context")
            logging.info(f"Received otel_context={otel_context}")

            # Get the tracer if available
            tracer = getattr(self, "_tracer", None)
            if not tracer:
                # Just execute the function without tracing if no tracer
                return func(self, *args, **kwargs)

            logging.info(f"Creating span {name} for {func.__name__}")

            # Get info about the request
            # model = kwargs.get("model") or getattr(self, "model", "unknown")
            messages = kwargs.get("messages") or args[0] if args else None
            tools = kwargs.get("tools", None)

            # Calculate message token count estimate
            message_count = 0
            token_estimate = 0
            if messages:
                if isinstance(messages, list):
                    message_count = len(messages)
                    token_estimate = sum(
                        _estimate_tokens_from_message(msg) for msg in messages
                    )
                else:
                    message_count = 1
                    token_estimate = _estimate_tokens_from_message(messages)

            # Try to restore context if provided
            current_context = None
            if otel_context:
                try:
                    current_context = restore_otel_context(otel_context)
                except Exception as e:
                    logging.warning(f"Failed to restore OpenTelemetry context: {e}")

            # Start the span with the restored context
            with tracer.start_as_current_span(name, context=current_context) as span:
                # span.set_attribute("llm.provider", getattr(self, "provider", "openai"))
                # span.set_attribute("llm.model", model)
                span.set_attribute("llm.request.message_count", message_count)
                span.set_attribute("llm.request.estimated_tokens", token_estimate)
                span.set_attribute("llm.tool_count", len(tools) if tools else 0)

                logging.info(f"LLM span created successfully: {span}")

                # Record start time
                start_time = time.time()

                try:
                    # Make the actual LLM API call
                    response = func(self, *args, **kwargs)

                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("llm.latency_ms", duration_ms)

                    # Extract usage information if available
                    if hasattr(response, "usage") and response.usage:
                        usage = response.usage
                        span.set_attribute(
                            "llm.tokens.prompt", getattr(usage, "prompt_tokens", 0)
                        )
                        span.set_attribute(
                            "llm.tokens.completion",
                            getattr(usage, "completion_tokens", 0),
                        )
                        span.set_attribute(
                            "llm.tokens.total", getattr(usage, "total_tokens", 0)
                        )

                    # Check for tool calls
                    if hasattr(response, "choices") and response.choices:
                        choice = response.choices[0]
                        if hasattr(choice, "message") and hasattr(
                            choice.message, "tool_calls"
                        ):
                            tool_calls = choice.message.tool_calls
                            if tool_calls:
                                span.set_attribute(
                                    "llm.response.tool_calls_count", len(tool_calls)
                                )

                    logging.info(f"LLM function executed: {response}")
                    return response

                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    span.record_exception(e)
                    span.set_attribute("error.message", str(e))
                    raise

        return wrapper

    return decorator


def _estimate_tokens_from_message(message):
    """Simple token estimation function"""
    if isinstance(message, str):
        return len(message.split()) // 3 * 4  # Rough approximation

    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content.split()) // 3 * 4  # Rough approximation

    return 0  # Can't estimate


def async_span_decorator(name):
    """Decorator that creates an OpenTelemetry span for an async method."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            otel_context = kwargs.get("otel_context")
            current_context = None

            try:
                # Get the tracer if available
                tracer = getattr(self, "_tracer", None)
                if not tracer:
                    # Just execute the function without tracing if no tracer
                    return await func(self, *args, **kwargs)

                # Try to restore context if provided, otherwise use current context
                if otel_context:
                    try:
                        current_context = restore_otel_context(otel_context)
                    except Exception as e:
                        logging.warning(f"Failed to restore OpenTelemetry context: {e}")

                # Start a new span with appropriate context
                with tracer.start_as_current_span(
                    name, context=current_context
                ) as span:
                    # Add attributes to span based on function name and args if desired
                    span.set_attribute("function.name", func.__name__)

                    # Execute the actual function
                    return await func(self, *args, **kwargs)

            except Exception as e:
                # Log error and re-raise
                logging.error(f"Error in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


def span_decorator(name):
    """Decorator that creates an OpenTelemetry span for a synchronous method."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            otel_context = kwargs.get("otel_context")
            current_context = None

            try:
                # Get the tracer if available
                tracer = getattr(self, "_tracer", None)
                if not tracer:
                    # Just execute the function without tracing if no tracer
                    return func(self, *args, **kwargs)

                # Try to restore context if provided, otherwise use current context
                if otel_context:
                    try:
                        current_context = restore_otel_context(otel_context)
                    except Exception as e:
                        logging.warning(f"Failed to restore OpenTelemetry context: {e}")

                # Start a new span with appropriate context
                with tracer.start_as_current_span(
                    name, context=current_context
                ) as span:
                    # Add attributes to span based on function name
                    span.set_attribute("function.name", func.__name__)

                    # Execute the actual function
                    return func(self, *args, **kwargs)

            except Exception as e:
                # Log error and re-raise
                logging.error(f"Error in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


def restore_otel_context(otel_context: dict[str, Any]) -> Optional[Context]:
    ctx = Context()
    if otel_context and "traceparent" in otel_context:
        carrier = {
            "traceparent": otel_context.get("traceparent", ""),
            "tracestate": otel_context.get("tracestate", ""),
        }

        temp_ctx = _propagator.extract(carrier=carrier)

        temp_span = trace.get_current_span(temp_ctx)
        if temp_span:
            span_context = temp_span.get_span_context()
            if span_context.is_valid:
                current_span = trace.NonRecordingSpan(span_context)
                ctx = trace.set_span_in_context(current_span, ctx)
        
    return ctx


def extract_otel_context() -> dict[str, Any]:
    """
    Extract current OpenTelemetry context for cross-boundary propagation.
    Returns a format that can be properly serialized by Dapr workflows.
    """
    carrier: dict[str, Any] = {}
    _propagator.inject(carrier)

    span = trace.get_current_span()
    ctx = span.get_span_context()

    # Always extract these values regardless of condition
    trace_id = format(ctx.trace_id, "032x")
    span_id = format(ctx.span_id, "016x")
    flags = "01" if ctx.trace_flags.sampled else "00"

    # If the propagator didn't inject a traceparent, create it manually
    if "traceparent" not in carrier and span and span.is_recording():
        carrier["traceparent"] = f"00-{trace_id}-{span_id}-{flags}"

    # Ensure tracestate exists in the carrier (default to empty string if not present)
    if "tracestate" not in carrier:
        carrier["tracestate"] = ""

    # Add the extra fields directly to the carrier
    carrier["trace_id"] = trace_id
    carrier["span_id"] = span_id
    carrier["trace_flags"] = flags

    return carrier
