---
mode: agent
model: gpt-4o
temperature: 0.7
api_provider: GitHub Models API
tools:
  - get_current_datetime
  - reverse_string
  - calculator
description: "A professional AI assistant that can get current date/time, perform calculations, and reverse strings"
system_prompt: "You are a professional and helpful AI assistant. Provide succinct, accurate responses."
---

# Python LangChain Agent

A multi-tool AI agent built with LangChain that can perform various tasks including:

## Available Tools

1. **get_current_datetime** - Returns the current date and time in the format YYYY-MM-DD HH:MM:SS
2. **calculator** - Evaluates simple mathematical expressions (e.g., "25 * 4 + 10")
3. **reverse_string** - Reverses input strings

## Configuration

- **Model**: GPT-4o via GitHub Models API
- **Temperature**: 0.7 (balanced between deterministic and creative responses)
- **Base URL**: https://models.inference.ai.azure.com

## Usage

```bash
python3 agent.py
```

The agent will process example queries and return responses using the appropriate tools.

## Environment Setup

Required environment variable in `.env`:
- `API_KEY` - GitHub Models API authentication token

## Best Practices

### Security
- **Never hardcode credentials** - Always use environment variables for API keys
- **Validate inputs** - Sanitize user inputs before passing to tools
- **Use restricted eval()** - The calculator tool uses `eval()` with restricted builtins for safety
- **Protect sensitive data** - Don't log or display API keys in output

### Tool Development
- **Use @tool decorator** - Modern LangChain approach for defining tools
- **Clear descriptions** - Each tool should have a clear docstring for the agent to understand its purpose
- **Type hints** - Use proper type annotations for tool parameters and return values
- **Error handling** - Wrap tool logic in try-except blocks to gracefully handle failures

### Agent Configuration
- **Appropriate temperature** - Use 0.7 for balanced responses (0.0 for deterministic, 1.0 for creative)
- **Concise prompts** - Keep system prompts clear and succinct
- **Tool selection** - Only include necessary tools to reduce token usage and confusion
- **Debug mode** - Enable `debug=False` in production to reduce verbose output

### Performance
- **Batch requests** - Process multiple queries efficiently
- **Cache credentials** - Load environment variables once at startup
- **Monitor token usage** - Track API costs and optimize prompts
- **Rate limiting** - Implement delays between requests if needed

### Testing
- **Test each tool independently** - Verify tools work before integrating with the agent
- **Test edge cases** - Handle division by zero, empty strings, invalid expressions
- **Validate responses** - Ensure agent selects the correct tool for each query
- **Monitor performance** - Track response times and error rates

