from typing import Optional, Any, Union
import logging
import requests

from pydantic import BaseModel, Field, PrivateAttr
from dapr_agents.types import ToolError
from urllib.parse import urlparse
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry import trace
from dapr_agents.agent.telemetry.otel import extract_otel_context


logger = logging.getLogger(__name__)


class DaprHTTPClient(BaseModel):
    """
    Client for sending HTTP requests to Dapr endpoints.
    """

    dapr_app_id: Optional[str] = Field(
        default="", description="Optional name of the Dapr App ID to invoke."
    )

    dapr_http_endpoint: Optional[str] = Field(
        default="",
        description="Optional name of the HTTPEndpoint to call for invocation",
    )

    http_endpoint: Optional[str] = Field(
        default="", description="Optional FQDN URL to request to."
    )

    path: Optional[str] = Field(
        default="", description="Optional name of the path to invoke."
    )

    headers: Optional[dict[str, str]] = Field(
        default={},
        description="Default headers to include in all requests.",
    )

    # Private attributes not exposed in model schema
    _base_url: str = PrivateAttr(default="http://localhost:3500/v1.0/invoke")

    def model_post_init(self, __context: Any) -> None:
        """Initialize the client after the model is created."""

        try:
            provider = trace.get_tracer_provider()

            self._tracer = provider.get_tracer("http_tool_tracer")

        except Exception as e:
            logger.warning(
                f"OpenTelemetry initialization failed: {e}. Continuing without telemetry."
            )

        RequestsInstrumentor().instrument()

        logger.debug("Initializing DaprHTTPClient client")

        super().model_post_init(__context)

    def do_http_request(
        self,
        payload: dict[str, str],
        endpoint: str = "",
        path: str = "",
        verb: str = "GET",
    ) -> Union[tuple[int, str] | ToolError]:
        """
        Send a POST request to the specified endpoint with the given input.

        Args:
            endpoint_url (str): The host of the URI to send the request to.
            payload (dict[str, str]): The payload to include in the request.
            path (str): The path of the URI to invoke including any query strings appended.
            verb (str): The HTTP verb. Either GET or POST.
        Returns:
            A tuple with the http status code and respose or a ToolError.
        """

        try:
            url = self._validate_endpoint_type(
                endpoint=endpoint, path=self.path if path == "" else path
            )
        except ToolError as e:
            logger.error(f"Error validating endpoint: {e}")
            raise e

        span = self._tracer.start_span(name="http_tool_request")
        with trace.use_span(span, end_on_exit=False):
            headers = self._generate_cloudevent_headers()
            if self.headers:
                headers.update(self.headers)
            logger.info(f"Sending with CloudEvents headers: {headers}")

            logger.debug(
                f"[HTTP] Sending POST request to '{url}' with input '{payload}' and headers '{headers}"
            )

            match verb.upper():
                case "GET":
                    response = requests.get(url=str(url), headers=headers)
                case "POST":
                    response = requests.post(
                        url=str(url), headers=headers, json=payload
                    )
                case _:
                    raise ValueError(
                        f"Value for 'verb' not in expected format ['GET'|'POST']: {verb}"
                    )

            logger.debug(
                f"Request returned status code '{response.status_code}' and '{response.text}'"
            )

            if not response.ok:
                raise ToolError(
                    f"Error occured sending the request. Received '{response.status_code}' - '{response.text}'"
                )

            return (response.status_code, response.text)

    def _validate_endpoint_type(
        self, endpoint: str, path: Optional[str | None]
    ) -> Union[str | ToolError]:
        if path == "":
            raise ToolError("No path provided. Please provide a valid path.")

        if isinstance(path, str) and path.startswith("/"):
            # Remove leading slash
            path = path[1:]

        try:
            if self.dapr_app_id != "":
                # Prefered option
                if isinstance(self.dapr_app_id, str) and self.dapr_app_id.endswith("/"):
                    # Remove trailing slash
                    self.dapr_app_id = self.dapr_app_id[:-1]
                url = f"{self._base_url}/{self.dapr_app_id}/method/{self.path if path == '' else path}"
            elif self.dapr_http_endpoint != "":
                # Dapr HTTPEndpoint
                if isinstance(
                    self.dapr_http_endpoint, str
                ) and self.dapr_http_endpoint.endswith("/"):
                    # Remove trailing slash
                    self.dapr_http_endpoint = self.dapr_http_endpoint[:-1]
                url = f"{self._base_url}/{self.dapr_http_endpoint}/method/{self.path if path == '' else path}"
            elif self.http_endpoint != "":
                # FQDN URL
                if isinstance(self.http_endpoint, str) and self.http_endpoint.endswith(
                    "/"
                ):
                    # Remove trailing slash
                    self.http_endpoint = self.http_endpoint[:-1]
                url = f"{self._base_url}/{self.http_endpoint}/method/{self.path if path == '' else path}"
            elif endpoint != "":
                # Fallback to default
                if isinstance(endpoint, str) and endpoint.endswith("/"):
                    # Remove trailing slash
                    endpoint = endpoint[:-1]
                url = f"{self._base_url}/{endpoint}/method/{self.path if path == '' else path}"
            else:
                raise ToolError(
                    "No endpoint provided. Please provide a valid dapr-app-id, HTTPEndpoint or endpoint."
                )
        except Exception as e:
            logger.error(f"Error validating endpoint: {e}")
            raise ToolError(
                "Error occured while validating the endpoint. Please check the provided values."
            )

        if not self._validate_url(url):
            raise ToolError(f"'{url}' is not a valid URL.")

        return url

    def _validate_url(self, url) -> bool:
        """
        Valides URL for HTTP requests
        """
        logger.debug(f"[HTTP] Url to be validated: {url}")
        try:
            parsed_url = urlparse(url=url)
            return all([parsed_url.scheme, parsed_url.netloc])
        except AttributeError:
            return False

    def _generate_cloudevent_headers(self) -> dict[str, str]:
        """
        Generate CloudEvent headers with trace context for HTTP requests.

        Args:
            otel_context: OpenTelemetry context from upstream services

        Returns:
            Dictionary of CloudEvent headers with trace context
        """
        headers = {
            "cloudevent.specversion": "1.0",
        }

        try:
            trace_context = extract_otel_context()

            # Add W3C standard headers
            if "traceparent" in trace_context:
                headers["traceparent"] = trace_context["traceparent"]
            if "tracestate" in trace_context:
                headers["tracestate"] = trace_context["tracestate"]

            # Add CloudEvents extensions for trace context
            if "traceparent" in trace_context:
                headers["cloudevent.traceparent"] = trace_context["traceparent"]
            if "tracestate" in trace_context:
                headers["cloudevent.tracestate"] = trace_context["tracestate"]
        except Exception as e:
            logger.warning(f"Failed to extract trace context: {e}")

        return headers
