# Python LangChain Agent (GitHub Models)

A professional AI assistant built using **LangChain** and **GitHub Models API**. This agent is equipped with several custom tools to perform specific tasks such as retrieving the current time, performing calculations, and manipulating strings.

## 🚀 Features

- **Custom Tools**:
  - `get_current_datetime`: Returns the current date and time.
  - `reverse_string`: Reverses any provided string.
  - `calculator`: Evaluates mathematical expressions safely.
- **Interactive Session**: Ask your own questions and interact with the agent in real-time.
- **Example Queries**: Automatically runs a set of test queries on startup to demonstrate tool usage.
- **GitHub Models Integration**: Uses the `gpt-4o` model via the GitHub Models inference endpoint.

## 🛠️ Prerequisites

- Python 3.8+
- A GitHub personal access token (with access to GitHub Models)

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/CQLOR/Python-Langchain.git
   cd Python-Langchain
   ```

2. **Install dependencies**:
   ```bash
   pip install langchain-openai langchain python-dotenv
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your GitHub API Key:
   ```env
   API_KEY=your_github_token_here
   ```

## 🖥️ Usage

Run the agent with the following command:

```bash
python3 agent.py
```

### Example Interaction

```text
🚀 Agent is ready to receive input...

Running example queries:

📝 Query: What time is it right now?
✅ Response: The current date and time is 2026-06-09 16:30:12.

--- Interactive Session ---
Type your query below (or type 'exit' or 'quit' to stop):

👤 You: What is 50 * 5?
✅ Response: 250
```

## 🧩 Tools Used

- **get_current_datetime**: Uses the `datetime` library to provide real-time updates.
- **calculator**: Uses a safe evaluation method for basic math.
- **reverse_string**: A simple string processing tool.

## ⚠️ Troubleshooting

### Rate Limits (Error code: 429)
The GitHub Models API has usage limits. If you see a `RateLimitReached` error, you will need to wait for the specified duration before making more requests.

---

Built by CQLOR
