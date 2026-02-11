from datetime import datetime, timedelta
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("yipyap")

SUBREDDITS = [
    "MachineLearning",
    "artificial",
    "LocalLLaMA",
    "OpenAI",
    "ClaudeAI",
    "PromptEngineering",
    "learnmachinelearning",
    "neuralnetworks",
    "singularity",
    "programming",
    "LangChain",
    "StableDiffusion",
    "ArtificialInteligence",
    "Futurology",
    "ControlProblem"
]


async def fetch_reddit_posts(days: int = 7, limit: int = 5) -> list[dict[str, Any]]:
    seven_days_ago = datetime.now() - timedelta(days=days)
    seven_days_ago_timestamp = seven_days_ago.timestamp()
    
    all_posts = []
    
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        for subreddit in SUBREDDITS:
            response = await client.get(
                f"https://www.reddit.com/r/{subreddit}/top.json",
                params={"limit": 100, "t": "week"},
                headers={"User-Agent": "yipyap/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                created_utc = post.get("created_utc", 0)
                
                if created_utc >= seven_days_ago_timestamp:
                    all_posts.append({
                        "title": post.get("title", ""),
                        "url": post.get("url", ""),
                        "score": post.get("score", 0),
                        "comments": post.get("num_comments", 0),
                        "created": created_utc,
                        "source": "Reddit",
                        "subreddit": subreddit,
                    })
    
    all_posts.sort(key=lambda x: x["score"], reverse=True)
    return all_posts[:limit]


async def fetch_hn_posts(days: int = 7, limit: int = 5) -> list[dict[str, Any]]:
    seven_days_ago = datetime.now() - timedelta(days=days)
    seven_days_ago_timestamp = int(seven_days_ago.timestamp())
    
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        response = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "tags": "story",
                "numericFilters": f"created_at_i>{seven_days_ago_timestamp}",
                "hitsPerPage": limit,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        posts = []
        for hit in data.get("hits", []):
            posts.append({
                "title": hit.get("title", ""),
                "url": hit.get("url", "") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                "score": hit.get("points", 0),
                "comments": hit.get("num_comments", 0),
                "created": hit.get("created_at_i", 0),
                "source": "HackerNews",
            })
        
        return posts


@mcp.tool()
async def summarise_weekly() -> str:
    """Get top tech news from the past week across Reddit and Hacker News."""
    
    errors = []
    
    try:
        reddit_posts = await fetch_reddit_posts(days=7, limit=5)
    except Exception as e:
        reddit_posts = []
        errors.append(f"Reddit error: {str(e)}")
    
    try:
        hn_posts = await fetch_hn_posts(days=7, limit=5)
    except Exception as e:
        hn_posts = []
        errors.append(f"HN error: {str(e)}")
    
    all_posts = reddit_posts + hn_posts
    all_posts.sort(key=lambda x: x["score"], reverse=True)
    
    if not all_posts:
        error_msg = "No posts found in the past week."
        if errors:
            error_msg += "\n\nErrors encountered:\n" + "\n".join(errors)
        return error_msg
    
    result = "# Top Tech News - Past Week\n\n"
    
    for i, post in enumerate(all_posts[:10], 1):
        created_date = datetime.fromtimestamp(post["created"]).strftime("%Y-%m-%d")
        source = post["source"]
        if source == "Reddit":
            source_detail = f"{source} (r/{post['subreddit']})"
        else:
            source_detail = source
        
        result += f"## {i}. {post['title']}\n"
        result += f"**Source:** {source_detail}\n"
        result += f"**Score:** {post['score']} points | {post['comments']} comments\n"
        result += f"**Date:** {created_date}\n"
        result += f"**URL:** {post['url']}\n\n"
    
    return result