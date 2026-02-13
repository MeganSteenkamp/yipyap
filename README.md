# YipYap - AI Tech News MCP Server

An MCP (Model Context Protocol) server that fetches AI tech news from Reddit and Hacker News, with AI-powered article summarization.

## Features

Four tools to help you stay up-to-date with AI news:

- **`summarise_weekly`** - Get top AI news from the past week with AI-generated TLDRs
- **`get_news`** - Search for news about specific AI topics with TLDRs
- **`get_trending`** - Find posts with high engagement (lots of discussion) with TLDRs
- **`get_drama`** - Discover controversial AI discussions with TLDRs

Each tool now includes AI-powered article summarization using Groq's Llama model, providing concise TLDRs for linked articles. Photo-only posts are automatically filtered out to focus on substantive content.

## Installation

```bash
pip install -e .
```

## Setup

1. Get a free Groq API key from [https://console.groq.com](https://console.groq.com)

2. Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
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

Get the top 10 AI news posts from the past week across Reddit and Hacker News, sorted by score. Each post includes an AI-generated TLDR of the linked article.

**Example:**
```
summarise_weekly()
```

### get_news(keyword, days=7)

Search for news about a specific AI topic with AI-generated TLDRs.

**Parameters:**
- `keyword` (required) - Search term (e.g., "GPT-5", "Claude", "Gemini")
- `days` (optional) - Number of days to look back (default: 7)

**Example:**
```
get_news(keyword="Claude", days=3)
```

### get_trending(days=7)

Find posts with high engagement - lots of comments relative to upvotes. Includes AI-generated TLDRs.

**Parameters:**
- `days` (optional) - Number of days to look back (default: 7)

**Example:**
```
get_trending(days=7)
```

### get_drama(days=7)

Discover controversial AI discussions on Reddit with AI-generated TLDRs.

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

## How It Works

1. Fetches posts from Reddit and Hacker News
2. Filters out photo-only posts
3. Fetches article content from URLs
4. Generates concise TLDRs using Groq's Llama 3.3 70B model
5. Returns formatted results with metadata and summaries

## License

MIT
