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


async def fetch_reddit_controversial(days: int = 7, limit: int = 5) -> list[dict[str, Any]]:
    days_ago = datetime.now() - timedelta(days=days)
    days_ago_timestamp = days_ago.timestamp()
    
    all_posts = []
    
    time_param = "day" if days <= 1 else "week" if days <= 7 else "month"
    
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        for subreddit in SUBREDDITS:
            response = await client.get(
                f"https://www.reddit.com/r/{subreddit}/controversial.json",
                params={"limit": 100, "t": time_param},
                headers={"User-Agent": "yipyap/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                created_utc = post.get("created_utc", 0)
                
                if created_utc >= days_ago_timestamp:
                    all_posts.append({
                        "title": post.get("title", ""),
                        "url": post.get("url", ""),
                        "score": post.get("score", 0),
                        "comments": post.get("num_comments", 0),
                        "created": created_utc,
                        "source": "Reddit",
                        "subreddit": subreddit,
                    })
    
    all_posts.sort(key=lambda x: x["comments"], reverse=True)
    return all_posts[:limit]


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


@mcp.tool()
async def get_drama(days: int = 7) -> str:
    """Get controversial/heated AI discussions from Reddit.
    
    Args:
        days: Number of days to look back (default: 7)
    """
    
    try:
        drama_posts = await fetch_reddit_controversial(days=days, limit=5)
    except Exception as e:
        return f"Error fetching controversial posts: {str(e)}"
    
    if not drama_posts:
        return f"No controversial posts found in the past {days} day(s)."
    
    result = f"# Controversial AI Discussions - Past {days} Day(s)\n\n"
    
    for i, post in enumerate(drama_posts, 1):
        created_date = datetime.fromtimestamp(post["created"]).strftime("%Y-%m-%d")
        
        result += f"## {i}. {post['title']}\n"
        result += f"**Source:** Reddit (r/{post['subreddit']})\n"
        result += f"**Score:** {post['score']} points | {post['comments']} comments\n"
        result += f"**Date:** {created_date}\n"
        result += f"**URL:** {post['url']}\n\n"
    
    return result


@mcp.tool()
async def get_trending(days: int = 7) -> str:
    """Get trending AI posts with high engagement (comment-to-score ratio).
    
    Args:
        days: Number of days to look back (default: 7)
    """
    
    errors = []
    
    try:
        reddit_posts = await fetch_reddit_posts(days=days, limit=100)
        for post in reddit_posts:
            post["ratio"] = post["comments"] / max(post["score"], 1)
        reddit_posts.sort(key=lambda x: x["ratio"], reverse=True)
        reddit_trending = reddit_posts[:5]
    except Exception as e:
        reddit_trending = []
        errors.append(f"Reddit error: {str(e)}")
    
    try:
        hn_posts = await fetch_hn_posts(days=days, limit=100)
        for post in hn_posts:
            post["ratio"] = post["comments"] / max(post["score"], 1)
        hn_posts.sort(key=lambda x: x["ratio"], reverse=True)
        hn_trending = hn_posts[:5]
    except Exception as e:
        hn_trending = []
        errors.append(f"HN error: {str(e)}")
    
    all_posts = reddit_trending + hn_trending
    all_posts.sort(key=lambda x: x["ratio"], reverse=True)
    
    if not all_posts:
        error_msg = f"No trending posts found in the past {days} day(s)."
        if errors:
            error_msg += "\n\nErrors encountered:\n" + "\n".join(errors)
        return error_msg
    
    result = f"# Trending AI Discussions - Past {days} Day(s)\n\n"
    result += "*Posts with high engagement (lots of discussion relative to upvotes)*\n\n"
    
    for i, post in enumerate(all_posts[:10], 1):
        created_date = datetime.fromtimestamp(post["created"]).strftime("%Y-%m-%d")
        source = post["source"]
        if source == "Reddit":
            source_detail = f"{source} (r/{post['subreddit']})"
        else:
            source_detail = source
        
        result += f"## {i}. {post['title']}\n"
        result += f"**Source:** {source_detail}\n"
        result += f"**Score:** {post['score']} points | {post['comments']} comments (ratio: {post['ratio']:.2f})\n"
        result += f"**Date:** {created_date}\n"
        result += f"**URL:** {post['url']}\n\n"
    
    return result