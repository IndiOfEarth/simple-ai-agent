# Import ready-made tools from LangChain community
# These let the agent search the web and Wikipedia
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun

# Wrapper for configuring how Wikipedia search behaves
from langchain_community.utilities import WikipediaAPIWrapper

# Tool class lets us define custom tools the agent can use
from langchain.tools import Tool

# Used for adding timestamps when saving files
from datetime import datetime


# Custom function to save research output to a text file
def save_to_txt(data: str, filename: str = "research_output.txt"):
    
    # Get current date + time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the text nicely before saving
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    # Open the file in "append" mode ("a")
    # This means it ADDS new data instead of overwriting the file
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    # Return a success message (this is what the agent sees)
    return f"Data successfully saved to {filename}"


# Wrap the save function as a tool the agent can use
save_tool = Tool(
    name="save_text_to_file",                     # Name the agent will call
    func=save_to_txt,                            # Function it runs
    description="Saves structured research data to a text file",
)


# Create a DuckDuckGo search tool (for general web search)
search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="search",                               # Tool name used by agent
    func=search.run,                             # Runs the search
    description="Search the web for information",
)


# Configure Wikipedia search behaviour
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,             # Only return the top result
    docs_content_chars_max=100   # Limit how much text is returned
)

# Create Wikipedia tool using the wrapper
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)