# YipYap - AI Tech News MCP Server

An MCP (Model Context Protocol) server that fetches AI tech news from Reddit and Hacker News.

## Features

Four tools to help you stay up-to-date with AI news:

- **`summarise_weekly`** - Get top AI news from the past week
- **`get_news`** - Search for news about specific AI topics
- **`get_trending`** - Find posts with high engagement (lots of discussion)
- **`get_drama`** - Discover controversial AI discussions

## Installation

```bash
pip install -e .
```

## Usage

### Testing with MCP Inspector

Test the server locally:

```bash
mcp dev yipyap.py
```

### Using with Claude Desktop

Add to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "yipyap": {
      "command": "python",
      "args": ["/path/to/yipyap/yipyap.py"]
    }
  }
}
```

Then restart Claude Desktop.

## Tools

### summarise_weekly()

Get the top 10 AI news posts from the past week across Reddit and Hacker News, sorted by score.

**Example:**
```
summarise_weekly()
```

### get_news(keyword, days=7)

Search for news about a specific AI topic.

**Parameters:**
- `keyword` (required) - Search term (e.g., "GPT-5", "Claude", "Gemini")
- `days` (optional) - Number of days to look back (default: 7)

**Example:**
```
get_news(keyword="Claude", days=3)
```

### get_trending(days=7)

Find posts with high engagement - lots of comments relative to upvotes.

**Parameters:**
- `days` (optional) - Number of days to look back (default: 7)

**Example:**
```
get_trending(days=7)
```

### get_drama(days=7)

Discover controversial AI discussions on Reddit.

**Parameters:**
- `days` (optional) - Number of days to look back (default: 7)

**Example:**
```
get_drama(days=3)
```

## Sources

**Reddit subreddits:**
- MachineLearning
- artificial
- LocalLLaMA
- OpenAI
- ClaudeAI
- PromptEngineering
- learnmachinelearning
- neuralnetworks
- singularity
- programming
- LangChain
- StableDiffusion
- ArtificialInteligence
- Futurology
- ControlProblem

**Hacker News:**
- All AI-related stories via Algolia API

## License

MIT
