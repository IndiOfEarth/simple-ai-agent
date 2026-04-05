# Mini AI research agent
# Flow:
# 1. Ask the user for a research question
# 2. Send that question to an LLM
# 3. Force the LLM to return the answer in a fixed structure
# 4. Parse the answer into a Python object
# 5. Print the clean structured result

# Loads environment variables from a .env file
# Example: API keys like ANTHROPIC_API_KEY
from dotenv import load_dotenv

# Lets us define a structured response format using a Python class
from pydantic import BaseModel

# LLM wrappers for OpenAI and Anthropic models
# Only ChatAnthropic is actually used in this code
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Used to build the prompt sent to the LLM
from langchain_core.prompts import ChatPromptTemplate

# Used to force / parse the LLM output into a Pydantic model
from langchain_core.output_parsers import PydanticOutputParser

# Tools for creating and running an agent
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Custom tools imported from another file
# These are likely functions the agent can use while researching
from tools import search_tool, wiki_tool, save_tool

# Load API keys and other environment variables from .env
load_dotenv()


# This class defines the exact structure we want back from the AI
class ResearchResponse(BaseModel):
    topic: str            # Main topic being researched
    summary: str          # Short summary of the answer
    sources: list[str]    # Sources used, such as URLs or references
    tools_used: list[str] # Names of tools the agent used


# Create the LLM we want to use
# This uses Anthropic's Claude Haiku model
llm = ChatAnthropic(model="claude-haiku-4-5")


# Create an output parser
# Its job is to take the raw LLM response and turn it into a ResearchResponse object
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# Build the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.

            Wrap the output in this format and provide no other text:
            {format_instructions}
            """,
        ),

        # Placeholder for past conversation messages
        # Not really used much in this simple version
        ("placeholder", "{chat_history}"),

        # The user's research question will be inserted here
        ("human", "{query}"),

        # Scratchpad used internally by the agent
        # This is where tool calls / intermediate reasoning are tracked
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Insert the parser's formatting instructions into the prompt
# This tells the LLM exactly what structure it must follow
prompt = prompt.partial(format_instructions=parser.get_format_instructions())


# List of tools the agent is allowed to use
tools = [search_tool, wiki_tool, save_tool]

# Create the tool-calling agent
# This combines:
# - the LLM
# - the prompt
# - the available tools
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


# Wrap the agent in an executor
# The executor actually runs the agent and manages tool use
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # Shows step-by-step logs in the terminal
)


# Ask the user for a research question
query = input("What can I help you research? ")


# Run the agent with the user's query
# The result is usually a dictionary containing the final output
raw_response = agent_executor.invoke({
    "query": query
})


# Try to parse the agent's output into the ResearchResponse structure
try:
    # Get the text output from the raw response
    # Then parse it into a Pydantic object
    structured_response = parser.parse(raw_response.get("output")[0]["text"])

    # Print the clean structured result
    print(structured_response)

except Exception as e:
    # If parsing fails, show the error and the raw response for debugging
    print("Error parsing response", e, "Raw response - ", raw_response)