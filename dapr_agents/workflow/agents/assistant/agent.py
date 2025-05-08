import json
import logging
from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Union

from dapr_agents.types.exceptions import AgentToolExecutorError, ToolError
from pydantic import Field

from dapr.ext.workflow import DaprWorkflowContext

from dapr_agents.types import (
    AgentError,
    ChatCompletion,
    ToolMessage,
)
from dapr_agents.workflow.agents.assistant.schemas import (
    AgentTaskResponse,
    BroadcastMessage,
    TriggerAction,
)
from dapr_agents.workflow.agents.assistant.state import (
    AssistantWorkflowEntry,
    AssistantWorkflowMessage,
    AssistantWorkflowState,
    AssistantWorkflowToolMessage,
)
from dapr_agents.workflow.agents.base import AgentWorkflowBase
from dapr_agents.workflow.decorators import task, workflow
from dapr_agents.workflow.messaging.decorator import message_router

from pydantic import PrivateAttr
from dapr_agents.agent.telemetry import (
    DaprAgentsOTel,
    async_span_decorator,
    span_decorator,
    extract_otel_context,
)

from opentelemetry import trace
from opentelemetry.context import get_current
from opentelemetry.trace import Tracer, Status, StatusCode

logger = logging.getLogger(__name__)


class AssistantAgent(AgentWorkflowBase):
    """
    A conversational AI agent that responds to user messages, engages in discussions,
    and dynamically utilizes external tools when needed.

    The AssistantAgent follows an agentic workflow, iterating on responses based on
    contextual understanding, reasoning, and tool-assisted execution. It ensures
    meaningful interactions by selecting the right tools, generating relevant responses,
    and refining outputs through iterative feedback loops.
    """

    tool_history: List[ToolMessage] = Field(
        default_factory=list, description="Executed tool calls during the conversation."
    )
    tool_choice: Optional[str] = Field(
        default=None,
        description="Strategy for selecting tools ('auto', 'required', 'none'). Defaults to 'auto' if tools are provided.",
    )

    _tracer: Optional[Tracer] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initializes the workflow with agentic execution capabilities."""

        try:
            otel_client = DaprAgentsOTel(
                service_name=self.name,
                otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
            )
            provider = otel_client.create_and_instrument_tracer_provider()

            self._tracer = provider.get_tracer(f"{self.name}_tracer")

        except Exception as e:
            logger.warning(
                f"OpenTelemetry initialization failed: {e}. Continuing without telemetry."
            )
            self._tracer = None

        # Initialize Agent State
        self.state = AssistantWorkflowState()

        # Name of main Workflow
        self._workflow_name = "ToolCallingWorkflow"

        # Define Tool Selection Strategy
        self.tool_choice = self.tool_choice or ("auto" if self.tools else None)

        super().model_post_init(__context)

    @message_router
    @workflow(name="ToolCallingWorkflow")
    @span_decorator("exec_tool_calling_wf")
    def tool_calling_workflow(
        self,
        ctx: DaprWorkflowContext,
        message: TriggerAction,
        otel_context: Optional[Dict[str, str]] = None,
    ):
        """
        Executes a tool-calling workflow, determining the task source (either an agent or an external user).
        """

        span = trace.get_current_span()

        # Step 0: Retrieve task and iteration input
        task = message.get("task")
        iteration = message.get("iteration", 0)
        instance_id = ctx.instance_id

        if isinstance(task, str):
            # Only set content attribute if it's reasonable in size
            if len(task) < 1000:
                span.set_attribute("workflow.task", task)
            span.set_attribute("workflow.task_size", len(task))
        else:
            span.set_attribute("workflow.task_type", str(type(task)))

        span.set_attribute("workflow.id", instance_id)
        span.set_attribute("workflow.iteration", iteration)
        span.set_attribute("workflow.task", task)

        if not ctx.is_replaying:
            logger.info(
                f"Workflow iteration {iteration + 1} started (Instance ID: {instance_id})."
            )

        # Step 1: Initialize instance entry on first iteration
        if iteration == 0:
            metadata = message.get("_message_metadata", {})

            # Ensure "instances" key exists
            self.state.setdefault("instances", {})

            # Extract workflow metadata with proper defaults
            source = metadata.get("source") or None
            source_workflow_instance_id = message.get("workflow_instance_id") or None

            # Create a new workflow entry
            workflow_entry = AssistantWorkflowEntry(
                input=task or "Triggered without input.",
                source=source,
                source_workflow_instance_id=source_workflow_instance_id,
            )

            # Store in state, converting to JSON only if necessary
            self.state["instances"].setdefault(
                instance_id, workflow_entry.model_dump(mode="json")
            )

            if not ctx.is_replaying:
                logger.info(
                    f"Initial message from {self.state['instances'][instance_id]['source']} -> {self.name}"
                )

        # Step 2: Retrieve workflow entry for this instance
        workflow_entry = self.state["instances"][instance_id]
        source = workflow_entry["source"]
        source_workflow_instance_id = workflow_entry["source_workflow_instance_id"]

        # Step 3: Generate Response
        response = yield ctx.call_activity(
            self.generate_response,
            input={
                "instance_id": instance_id,
                "task": task,
                "otel_context": otel_context,
            },
        )
        response_message = yield ctx.call_activity(
            self.get_response_message,
            input={"response": response, "otel_context": otel_context},
        )

        # Step 4: Extract Finish Reason
        finish_reason = yield ctx.call_activity(
            self.get_finish_reason,
            input={"response": response, "otel_context": otel_context},
        )

        # Step 5: Choose execution path based on LLM response
        if finish_reason == "tool_calls":
            if not ctx.is_replaying:
                logger.info(
                    "Tool calls detected in LLM response, extracting and preparing for execution.."
                )

            # Retrieve the list of tool calls extracted from the LLM response
            tool_calls = yield ctx.call_activity(
                self.get_tool_calls,
                input={"response": response, "otel_context": otel_context},
            )
            span.add_event(
                "tool_calls_detected", {"count": len(tool_calls) if tool_calls else 0}
            )

            # Execute tool calls in parallel
            if not ctx.is_replaying:
                logger.info(f"Executing {len(tool_calls)} tool call(s)..")

            parallel_tasks = [
                ctx.call_activity(
                    self.execute_tool,
                    input={
                        "instance_id": instance_id,
                        "tool_call": tool_call,
                        "otel_context": otel_context,
                    },
                )
                for tool_call in tool_calls
            ]
            if self._tracer:
                with self._tracer.start_as_current_span(
                    "parallel_tool_execution"
                ) as parallel_span:
                    parallel_span.set_attribute("tool_calls.count", len(tool_calls))
                    yield self.when_all(parallel_tasks)
            else:
                # Fall back if no tracer is available
                span.add_event(
                    "parallel_tool_execution", {"tool_calls.count": len(tool_calls)}
                )
                yield self.when_all(parallel_tasks)
        else:
            span.add_event("no_tool_calls_detected")
            if not ctx.is_replaying:
                logger.info("Agent generating response without tool execution..")

            # No Tool Calls → Clear tools
            self.tool_history.clear()

        # Step 6: Determine if Workflow Should Continue
        next_iteration_count = iteration + 1
        max_iterations_reached = next_iteration_count > self.max_iterations

        if finish_reason == "stop" or max_iterations_reached:
            # Determine the reason for stopping
            if max_iterations_reached:
                span.add_event("max_iterations_reached", {"limit": self.max_iterations})
                verdict = "max_iterations_reached"
                if not ctx.is_replaying:
                    logger.warning(
                        f"Workflow {instance_id} reached the max iteration limit ({self.max_iterations}) before finishing naturally."
                    )

                # Modify the response message to indicate forced stop
                response_message[
                    "content"
                ] += "\n\nThe workflow was terminated because it reached the maximum iteration limit. The task may not be fully complete."

            else:
                span.add_event("natural_stop")
                verdict = "model hit a natural stop point."

            # Step 8: Broadcasting Response to all agents if available
            yield ctx.call_activity(
                self.broadcast_message_to_agents,
                input={
                    "message": response_message,
                    "otel_context": extract_otel_context(),
                },
            )

            # Step 9: Respond to source agent if available
            yield ctx.call_activity(
                self.send_response_back,
                input={
                    "response": response_message,
                    "target_agent": source,
                    "target_instance_id": source_workflow_instance_id,
                    "otel_context": extract_otel_context(),
                },
            )

            # Step 10: Share Final Message
            yield ctx.call_activity(
                self.finish_workflow,
                input={
                    "instance_id": instance_id,
                    "message": response_message,
                    "otel_context": extract_otel_context(),
                },
            )

            if not ctx.is_replaying:
                logger.info(
                    f"Workflow {instance_id} has been finalized with verdict: {verdict}"
                )

            return response_message

        else:
            span.add_event(
                "continuing_workflow", {"next_iteration": next_iteration_count}
            )

        # Step 7: Continue Workflow Execution
        message.update({"task": None, "iteration": next_iteration_count})
        ctx.continue_as_new(message)

    @task
    @async_span_decorator("generate_response")
    async def generate_response(
        self,
        instance_id: str,
        task: Union[str, Dict[str, Any]] = None,
        otel_context: Dict[str, Any] = None,
    ) -> ChatCompletion:
        """
        Generates a response using a language model based on the provided task input.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            task (Union[str, Dict[str, Any]], optional): The task description or structured input
                used to generate the response. Defaults to None.

        Returns:
            ChatCompletion: The generated AI response encapsulated in a ChatCompletion object.
        """

        # TODO: Finish me!!
        span = trace.get_current_span()

        # Construct prompt messages
        messages = self.construct_messages(task or {})
        span.set_attribute("workflow.messages_size", len(messages))
        span.set_attribute("workflow.id", instance_id)

        # Store message in workflow state and local memory
        if task:
            task_message = {"role": "user", "content": task}
            await self.update_workflow_state(
                instance_id=instance_id, message=task_message, otel_context=otel_context
            )

        # Process conversation iterations
        messages += self.tool_history

        # Generate Tool Calls
        response: ChatCompletion = self.llm.generate(
            messages=messages,
            tools=self.tools,
            tool_choice=self.tool_choice,
            otel_context=otel_context,
        )

        # Return chat completion as a dictionary
        return response.model_dump()

    @task
    def get_response_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts the response message from the first choice in the LLM response.

        Args:
            response (Dict[str, Any]): The response dictionary from the LLM, expected to contain a "choices" key.

        Returns:
            Dict[str, Any]: The extracted response message with the agent's name added.
        """
        choices = response.get("choices", [])
        response_message = choices[0].get("message", {})

        return response_message

    @task
    @span_decorator("get_finish_reason")
    def get_finish_reason(
        self, response: Dict[str, Any], otel_context: Dict[str, Any] = None
    ) -> str:
        """
        Extracts the finish reason from the LLM response, indicating why generation stopped.

        Args:
            response (Dict[str, Any]): The response dictionary from the LLM, expected to contain a "choices" key.

        Returns:
            str: The reason the model stopped generating tokens. Possible values include:
                - "stop": Natural stop point or stop sequence encountered.
                - "length": Maximum token limit reached.
                - "content_filter": Content flagged by filters.
                - "tool_calls": The model called a tool.
                - "function_call" (deprecated): The model called a function.
                - None: If no valid choice exists in the response.
        """
        choices = response.get("choices", [])

        # Ensure there is at least one choice available
        if not choices:
            logger.warning("No choices found in LLM response.")
            return None  # Explicit return to avoid returning 'None' implicitly

        # Extract finish reason safely
        choice = choices[0].get("finish_reason", None)

        return choice

    @task
    @span_decorator("get_tool_calls")
    def get_tool_calls(
        self, response: Dict[str, Any], otel_context: Dict[str, Any] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extracts tool calls from the first choice in the LLM response, if available.

        Args:
            response (Dict[str, Any]): The response dictionary from the LLM, expected to contain "choices"
                                    and potentially tool call information.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of tool calls if present, otherwise None.
        """
        choices = response.get("choices", [])

        if not choices:
            logger.warning("No choices found in LLM response.")
            return None

        # Save Tool Call Response Message
        response_message = choices[0].get("message", {})
        self.tool_history.append(response_message)

        # Extract tool calls safely
        tool_calls = choices[0].get("message", {}).get("tool_calls")

        if not tool_calls:
            logger.info("No tool calls found in LLM response.")
            return None

        return tool_calls

    @task
    @async_span_decorator("exec_tool")
    async def execute_tool(
        self,
        instance_id: str,
        tool_call: Dict[str, Any],
        otel_context: Dict[str, Any] = None,
    ):
        """
        Executes a tool call by invoking the specified function with the provided arguments.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            tool_call (Dict[str, Any]): A dictionary containing tool execution details, including the function name and arguments.

        Raises:
            AgentError: If the tool call is malformed or execution fails.
        """

        span = trace.get_current_span()
        span.set_attribute("workflow.id", instance_id)

        function_details = tool_call.get("function", {})
        function_name = function_details.get("name")
        span.set_attribute("tool.call.name", function_name)
        span.set_attribute("tool.call.details", str(function_details))

        if not function_name:
            raise AgentError("Missing function name in tool execution request.")

        try:
            function_args = function_details.get("arguments", "")
            function_args_as_dict = json.loads(function_args) if function_args else {}
            span.set_attribute("tool.call.args", str(function_args_as_dict))

            # Execute tool function
            result = await self.tool_executor.run_tool(
                function_name, otel_context=otel_context, **function_args_as_dict
            )
            # Construct tool execution message payload
            workflow_tool_message = {
                "tool_call_id": tool_call.get("id"),
                "function_name": function_name,
                "function_args": function_args,
                "content": str(result),
            }

            # Update workflow state and agent tool history
            await self.update_workflow_state(
                instance_id=instance_id, tool_message=workflow_tool_message
            )

        except (ToolError, AgentToolExecutorError) as e:
            logger.info(e)

            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)

            workflow_tool_message = {
                "tool_call_id": tool_call.get("id"),
                "function_name": function_name,
                "function_args": function_args,
                "content": str(e),
            }
            # Update workflow state and agent tool history
            await self.update_workflow_state(
                instance_id=instance_id, tool_message=workflow_tool_message
            )
        except json.JSONDecodeError:
            logger.error(
                f"Invalid JSON in tool arguments for function '{function_name}'"
            )
            raise AgentError(
                f"Invalid JSON format in arguments for tool '{function_name}'."
            )

        except Exception as e:
            logger.error(f"Error executing tool '{function_name}': {e}", exc_info=True)
            span.set_attribute("error.type", type(e).__name__)
            raise AgentError(f"Error executing tool '{function_name}': {e}") from e

    @task
    @async_span_decorator("broadcast_msg_to_agents")
    async def broadcast_message_to_agents(
        self, message: Dict[str, Any], otel_context: Dict[str, Any] = None
    ):
        """
        Broadcasts it to all registered agents.

        Args:
            message (Dict[str, Any]): A message to append to the workflow state and broadcast to all agents.
        """
        # Format message for broadcasting
        message["role"] = "user"
        message["name"] = self.name
        response_message = BroadcastMessage(**message)

        # Broadcast message to all agents
        await self.broadcast_message(
            message=response_message, otel_context=otel_context
        )

    @task
    @async_span_decorator("respond_to_agent")
    async def send_response_back(
        self,
        response: Dict[str, Any],
        target_agent: str,
        target_instance_id: str,
        otel_context: Dict[str, Any] = None,
    ):
        """
        Sends a task response back to a target agent within a workflow.

        Args:
            response (Dict[str, Any]): The response payload to be sent.
            target_agent (str): The name of the agent that should receive the response.
            target_instance_id (str): The workflow instance ID associated with the response.

        Raises:
            ValidationError: If the response does not match the expected structure for `AgentTaskResponse`.
        """
        # Format Response
        response["role"] = "user"
        response["name"] = self.name
        response["workflow_instance_id"] = target_instance_id
        agent_response = AgentTaskResponse(**response)

        # Send the message to the target agent
        await self.send_message_to_agent(
            name=target_agent, message=agent_response, otel_context=otel_context
        )

    @task
    @async_span_decorator("finish_workflow")
    async def finish_workflow(
        self,
        instance_id: str,
        message: Dict[str, Any],
        otel_context: Dict[str, Any] = None,
    ):
        """
        Finalizes the workflow by storing the provided message as the final output.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            summary (Dict[str, Any]): The final summary to be stored in the workflow state.
        """
        # Store message in workflow state
        await self.update_workflow_state(
            instance_id=instance_id, message=message, otel_context=otel_context
        )

        # Store final output
        await self.update_workflow_state(
            instance_id=instance_id,
            final_output=message["content"],
            otel_context=otel_context,
        )

    @async_span_decorator("update_wf_state")
    async def update_workflow_state(
        self,
        instance_id: str,
        message: Optional[Dict[str, Any]] = None,
        tool_message: Optional[Dict[str, Any]] = None,
        final_output: Optional[str] = None,
        otel_context: Dict[str, Any] = None,
    ):
        """
        Updates the workflow state by appending a new message or setting the final output.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            message (Optional[Dict[str, Any]]): A dictionary representing a user/assistant message.
            tool_message (Optional[Dict[str, Any]]): A dictionary representing a tool execution message.
            final_output (Optional[str]): The final output of the workflow, marking its completion.

        Raises:
            ValueError: If no workflow entry is found for the given instance_id.
        """
        span = trace.get_current_span()
        span.set_attribute("workflow.id", instance_id)
        span.set_attribute(
            "update.type",
            "message"
            if message
            else "tool_message"
            if tool_message
            else "final_output"
            if final_output
            else "unknown",
        )
        workflow_entry: AssistantWorkflowEntry = self.state["instances"].get(
            instance_id
        )
        if not workflow_entry:
            raise ValueError(
                f"No workflow entry found for instance_id {instance_id} in local state."
            )

        # Store user/assistant messages separately
        if message is not None:
            serialized_message = AssistantWorkflowMessage(**message).model_dump(
                mode="json"
            )
            workflow_entry["messages"].append(serialized_message)
            workflow_entry["last_message"] = serialized_message

            # Add to memory only if it's a user/assistant message
            self.memory.add_message(message)

        # Store tool execution messages separately in tool_history
        if tool_message is not None:
            serialized_tool_message = AssistantWorkflowToolMessage(
                **tool_message
            ).model_dump(mode="json")
            workflow_entry["tool_history"].append(serialized_tool_message)

            # Also update agent-level tool history (execution tracking)
            agent_tool_message = ToolMessage(
                tool_call_id=tool_message["tool_call_id"],
                name=tool_message["function_name"],
                content=tool_message["content"],
            )
            self.tool_history.append(agent_tool_message)

        # Store final output
        if final_output is not None:
            workflow_entry["output"] = final_output
            workflow_entry["end_time"] = datetime.now().isoformat()

        # Persist updated state
        self.save_state()

    @message_router(broadcast=True)
    @async_span_decorator("process_broadcast_msg")
    async def process_broadcast_message(
        self, message: BroadcastMessage, otel_context: Dict[str, Any] = None
    ):
        """
        Processes a broadcast message, filtering out messages sent by the same agent
        and updating local memory with valid messages.

        Args:
            message (BroadcastMessage): The received broadcast message.

        Returns:
            None: The function updates the agent's memory and ignores unwanted messages.
        """
        try:
            # Extract metadata safely from message["_message_metadata"]
            metadata = message.get("_message_metadata", {})

            if not isinstance(metadata, dict):
                logger.warning(
                    f"{self.name} received a broadcast message with invalid metadata format. Ignoring."
                )
                return

            source = metadata.get("source", "unknown_source")
            message_type = metadata.get("type", "unknown_type")
            message_content = message.get("content", "No Data")

            logger.info(
                f"{self.name} received broadcast message of type '{message_type}' from '{source}'."
            )

            # Ignore messages sent by this agent
            if source == self.name:
                logger.info(
                    f"{self.name} ignored its own broadcast message of type '{message_type}'."
                )
                return

            # Log and process the valid broadcast message
            logger.debug(
                f"{self.name} processing broadcast message from '{source}'. Content: {message_content}"
            )

            # Store the message in local memory
            self.memory.add_message(message)

        except Exception as e:
            logger.error(f"Error processing broadcast message: {e}", exc_info=True)
