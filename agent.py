import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Define tools using the @tool decorator
@tool
def get_current_datetime() -> str:
    """Returns the current date and time in the format YYYY-MM-DD HH:MM:SS"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def reverse_string(s: str) -> str:
    """Reverses the input string."""
    return s[::-1]

@tool
def calculator(expression: str) -> str:
    """Evaluates a simple mathematical expression. Input should be a valid mathematical expression like '25 * 4 + 10'."""
    try:
        # WARNING: Using eval can be dangerous. In production, consider using a safe math parser.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize the OpenAI language model (using GitHub Models API)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=os.getenv("API_KEY"),
    base_url="https://models.inference.ai.azure.com"  # GitHub Models inference endpoint
)

# Collect all tools
tools = [get_current_datetime, reverse_string, calculator]

# Create the agent with tools
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a professional and helpful AI assistant. Provide succinct, accurate responses.",
    debug=False
)

def main():
    """Main function to run example queries."""
    print("🚀 Agent is ready to receive input...\n")

    if not os.getenv("API_KEY"):
        print("❌ Error: API_KEY not found in environment variables.")
        print("Please create a .env file with API_KEY for GitHub Models API.")
        return

    # Example queries
    queries = [
        "What time is it right now?",
        "What is 25 * 4 + 10?",
        "Reverse the string 'Hello World'",
    ]

    print("Running example queries:\n")
    for query in queries:
        print(f"📝 Query: {query}")
        try:
            # Invoke the agent with the query
            input_messages = [{"role": "user", "content": query}]
            response = agent.invoke({"messages": input_messages})
            
            # Extract the final response from messages
            if response and "messages" in response:
                final_message = response["messages"][-1]
                print(f"✅ Response: {final_message.content}\n")
            else:
                print(f"✅ Response: {response}\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

    # Interactive session
    print("--- Interactive Session ---")
    print("Type your query below (or type 'exit' or 'quit' to stop):")
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
                
            # Invoke the agent with the user query
            input_messages = [{"role": "user", "content": query}]
            response = agent.invoke({"messages": input_messages})
            
            # Extract and print the final response
            if response and "messages" in response:
                final_message = response["messages"][-1]
                print(f"✅ Response: {final_message.content}")
            else:
                print(f"✅ Response: {response}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# Example usage of the agent
if __name__ == "__main__":
    main()