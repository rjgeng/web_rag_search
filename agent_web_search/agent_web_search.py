import os
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool

# Use DuckDuckGo as the web search tool
from duckduckgo_search import DDGS

# Load your OpenAI API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- Define a Web Search Tool Function ---
@function_tool
def web_search(query: str) -> str:
    """Searches the web for up-to-date information and returns top 3 results."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=3):
                results.append(f"{r['title']}\n{r['href']}\n{r['body']}\n")
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


# --- Create an Agent ---
research_agent = Agent(
    name="research_assistant",
    instructions="You are a research assistant who uses web search to answer questions. Always cite your sources with URLs when available.",
    tools=[web_search],  # Use tools parameter, not functions
    model="gpt-4o",  # or "gpt-4o-mini" for faster/cheaper responses
)


# --- Create Runner and Execute Query ---
def main():
    question = "Search Rongjun Geng information on line, focus on his career and professional scope"

    # Run the agent with the question using Runner.run_sync()
    result = Runner.run_sync(
        starting_agent=research_agent,
        input=question
    )

    # Print the response
    print(f"Assistant: {result.final_output}")


if __name__ == "__main__":
    main()