# build a mini ai research agent
# takes a question -> sends it to an llm -> forces the response into a structured format -> returns it cleanly
# essentially: answer this, but ONLY in this structure

# Load environment variables from a .env file (API keys etc.)
from dotenv import load_dotenv

# Used to define structured data models
from pydantic import BaseModel

# LLM wrappers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Prompt building tools
from langchain_core.prompts import ChatPromptTemplate

# Parses LLM output into a structured format
from langchain_core.output_parsers import PydanticOutputParser

# Agent creation tools
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool

# Load environment variables (e.g. ANTHROPIC_API_KEY)
load_dotenv()


# Define the structure we want the AI to return
class ResearchResponse(BaseModel):
    topic: str              # The topic of the research
    summary: str            # A short explanation/summary
    sources: list[str]      # List of sources (URLs, references, etc.)
    tools_used: list[str]   # Tools the agent used (if any)


# Initialise the LLM (Claude model)
llm = ChatAnthropic(model="claude-haiku-4-5")


# Create a parser that forces the LLM output into the above structure
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# Create a prompt template
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

        # Placeholder for conversation history (not used much here)
        ("placeholder", "{chat_history}"),

        # The actual user query goes here
        ("human", "{query}"),

        # Internal reasoning / tool usage scratchpad
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Inject format instructions from the parser into the prompt
prompt = prompt.partial(format_instructions=parser.get_format_instructions())


tools = [search_tool]

# Create an agent that can call tools (none provided yet)
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools  # No tools yet
)


# Wrap the agent in an executor (handles running it)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,     # Again, no tools
    verbose=True  # Prints step-by-step logs
)

query = input("What can i help you research? ")
# Run the agent with a query
raw_response = agent_executor.invoke({
    "query": query
})


# parses into a python object
try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw response - ", raw_response)
