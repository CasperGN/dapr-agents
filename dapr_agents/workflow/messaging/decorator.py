import logging
from copy import deepcopy
from typing import Any, Callable, Optional, get_type_hints
from dapr_agents.workflow.messaging.utils import (
    is_valid_routable_model,
    extract_message_models,
)
import functools
from opentelemetry import context
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


def message_router(
    func: Optional[Callable[..., Any]] = None,
    *,
    pubsub: Optional[str] = None,
    topic: Optional[str] = None,
    dead_letter_topic: Optional[str] = None,
    broadcast: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for registering message handlers by inspecting type hints on the 'message' argument."""

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        is_workflow = hasattr(f, "_is_workflow")
        workflow_name = getattr(f, "_workflow_name", None)

        type_hints = get_type_hints(f)
        raw_hint = type_hints.get("message", None)

        message_models = extract_message_models(raw_hint)

        if not message_models:
            raise ValueError(
                f"Message handler '{f.__name__}' must have a 'message' parameter with a valid type hint."
            )

        for model in message_models:
            if not is_valid_routable_model(model):
                raise TypeError(
                    f"Handler '{f.__name__}' has unsupported message type: {model}"
                )

        logger.debug(
            f"@message_router: '{f.__name__}' => models {[m.__name__ for m in message_models]}"
        )

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            otel_context = kwargs.get("otel_context", None)
            if not otel_context:
                otel_context = context.get_current()
                kwargs["otel_context"] = otel_context

            self_obj = args[0] if args else None
            tracer = getattr(self_obj, "_tracer", None)

            if tracer:
                with tracer.start_as_current_span(
                    f"message_handler_{f.__name__}",
                    context=otel_context,
                    end_on_exit=True,
                ) as span:
                    span.set_attribute("pubsub.name", pubsub)
                    span.set_attribute("pubsub.topic", topic)
                    span.set_attribute("handler.name", f.__name__)

                    try:
                        return f(*args, **kwargs)
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR))
                        span.record_exception(e)
                        raise
            else:
                return f(*args, **kwargs)

        wrapper._is_message_handler = True
        wrapper._message_router_data = deepcopy(
            {
                "pubsub": pubsub,
                "topic": topic,
                "dead_letter_topic": dead_letter_topic
                or (f"{topic}_DEAD" if topic else None),
                "is_broadcast": broadcast,
                "message_schemas": message_models,
                "message_types": [model.__name__ for model in message_models],
            }
        )

        if is_workflow:
            wrapper._is_workflow = True
            wrapper._workflow_name = workflow_name

        return wrapper

    return decorator(func) if func else decorator
