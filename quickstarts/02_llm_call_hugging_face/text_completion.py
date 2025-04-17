from dapr_agents.llm import HFHubChatClient
from dapr_agents.types import UserMessage

from dotenv import load_dotenv

load_dotenv()

# Basic chat completion
llm = HFHubChatClient(model="microsoft/Phi-3-mini-4k-instruct")
response = llm.generate("Name a famous dog!")

if len(response.get_content()) > 0:
    print("Response: ", response.get_content())

# Chat completion using a prompty file for context
llm = HFHubChatClient.from_prompty("basic.prompty")
response = llm.generate(input_data={"question": "What is your name?"})

if len(response.get_content()) > 0:
    print("Response with prompty: ", response.get_content())

# Chat completion with user input
llm = HFHubChatClient()
response = llm.generate(messages=[UserMessage("hello")])


if len(response.get_content()) > 0 and "hello" in response.get_content().lower():
    print("Response with user input: ", response.get_content())
