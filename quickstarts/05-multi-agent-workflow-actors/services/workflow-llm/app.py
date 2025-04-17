from dapr_agents import LLMOrchestrator
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        llm_workflow = LLMOrchestrator(
            name="LLMOrchestrator",
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
            max_iterations=3,
        ).as_service(port=8004)

        await llm_workflow.start()
    except Exception as e:
        print(f"Error starting workflow: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
