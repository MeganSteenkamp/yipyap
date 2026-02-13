import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import AsyncGroq
from mcp.server.fastmcp import FastMCP

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

logging.basicConfig(level=logging.INFO, format='%(message)s')

mcp = FastMCP("yipyap")

groq_api_key = os.environ.get("GROQ_API_KEY")
if groq_api_key:
    logging.info(f"✓ GROQ_API_KEY loaded (starts with: {groq_api_key[:10]}...)")
else:
    logging.error("✗ GROQ_API_KEY not found in environment variables")

groq_base_url = os.environ.get("GROQ_BASE_URL") or None
groq_timeout_seconds = float(os.environ.get("GROQ_TIMEOUT_SECONDS", "30"))
groq_max_retries = int(os.environ.get("GROQ_MAX_RETRIES", "4"))
groq_ssl_verify_raw = os.environ.get("GROQ_SSL_VERIFY", "true")
groq_ssl_verify = groq_ssl_verify_raw.strip().lower() not in {"0", "false", "no", "off"}
groq_ca_bundle = os.environ.get("GROQ_CA_BUNDLE") or None

logging.info(f"Groq base URL: {groq_base_url or 'https://api.groq.com'}")
logging.info(f"Groq timeout: {groq_timeout_seconds}s | max retries: {groq_max_retries}")
if groq_ca_bundle:
    logging.info(f"Groq TLS: using custom CA bundle at {groq_ca_bundle}")
else:
    logging.info(f"Groq TLS: verify={groq_ssl_verify}")

groq_http_client = (
    httpx.AsyncClient(
        timeout=groq_timeout_seconds,
        verify=groq_ca_bundle if groq_ca_bundle else groq_ssl_verify,
    )
    if groq_api_key
    else None
)

groq_client = (
    AsyncGroq(
        api_key=groq_api_key,
        base_url=groq_base_url,
        max_retries=groq_max_retries,
        http_client=groq_http_client,
    )
    if groq_api_key
    else None
)

SUMMARIZATION_PROMPT = """Summarize the following article in 2-3 concise sentences. Focus on the key points and main takeaways.

Article:
{content}

TLDR:"""

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
    "ControlProblem",
    "technology",
]


def is_photo_only_post(post: dict[str, Any]) -> bool:
    post_hint = post.get("post_hint", "")
    is_self = post.get("is_self", False)
    url = post.get("url", "")
    selftext = post.get("selftext", "")
    
    if post_hint == "image":
        return True
    
    if is_self and not selftext:
        return True
    
    image_domains = ["i.redd.it", "imgur.com", "i.imgur.com"]
    if any(domain in url for domain in image_domains) and not selftext:
        return True
    
    return False


async def fetch_article_content(url: str) -> str:
    logging.info(f"📥 Attempting to fetch: {url}")
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, verify=False) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            article = soup.find("article")
            if article:
                text = article.get_text(separator=" ", strip=True)
            else:
                main = soup.find("main") or soup.find("body")
                text = main.get_text(separator=" ", strip=True) if main else ""
            
            words = text.split()[:1000]
            content = " ".join(words)
            logging.info(f"✓ Fetched content from {url[:50]}... ({len(content)} chars)")
            return content
    except Exception as e:
        logging.error(f"✗ Failed to fetch {url[:50]}...: {type(e).__name__}: {str(e)}")
        return ""


async def generate_tldr(content: str) -> str:
    if not content or len(content) < 100:
        logging.warning(f"✗ Content too short for TLDR ({len(content)} chars)")
        return ""
    if groq_client is None:
        logging.error("✗ GROQ_API_KEY not configured; cannot generate TLDR")
        return ""
    
    try:
        logging.info(f"→ Generating TLDR for {len(content)} chars of content...")
        response = await groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": SUMMARIZATION_PROMPT.format(content=content[:8000])
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150,
        )
        tldr = response.choices[0].message.content.strip()
        logging.info(f"✓ Generated TLDR: {tldr[:50]}...")
        return tldr
    except Exception as e:
        logging.exception(f"✗ Failed to generate TLDR: {type(e).__name__}: {str(e)}")
        return ""


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
                
                if created_utc >= seven_days_ago_timestamp and not is_photo_only_post(post):
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


async def search_reddit_posts(keyword: str, days: int = 7, limit: int = 5) -> list[dict[str, Any]]:
    days_ago = datetime.now() - timedelta(days=days)
    days_ago_timestamp = days_ago.timestamp()
    
    all_posts = []
    
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        for subreddit in SUBREDDITS:
            response = await client.get(
                f"https://www.reddit.com/r/{subreddit}/search.json",
                params={
                    "q": keyword,
                    "restrict_sr": 1,
                    "sort": "top",
                    "t": "all",
                    "limit": 25,
                },
                headers={"User-Agent": "yipyap/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                created_utc = post.get("created_utc", 0)
                
                if created_utc >= days_ago_timestamp and not is_photo_only_post(post):
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


async def search_hn_posts(keyword: str, days: int = 7, limit: int = 5) -> list[dict[str, Any]]:
    days_ago = datetime.now() - timedelta(days=days)
    days_ago_timestamp = int(days_ago.timestamp())
    
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        response = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "query": keyword,
                "tags": "story",
                "numericFilters": f"created_at_i>{days_ago_timestamp}",
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
                
                if created_utc >= days_ago_timestamp and not is_photo_only_post(post):
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
    """Get top tech news from the past week across Reddit and Hacker News with AI-generated TLDRs."""
    
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
    
    logging.info(f"\n📊 Processing {len(all_posts[:10])} posts for TLDRs...\n")
    
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
        result += f"**URL:** {post['url']}\n"
        
        logging.info(f"\n[{i}/10] Processing: {post['title'][:60]}...")
        content = await fetch_article_content(post['url'])
        if content:
            tldr = await generate_tldr(content)
            if tldr:
                result += f"\n**TLDR:** {tldr}\n"
        
        result += "\n"
    
    return result


@mcp.tool()
async def get_drama(days: int = 7) -> str:
    """Get controversial/heated AI discussions from Reddit with AI-generated TLDRs.
    
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
        result += f"**URL:** {post['url']}\n"
        
        content = await fetch_article_content(post['url'])
        if content:
            tldr = await generate_tldr(content)
            if tldr:
                result += f"\n**TLDR:** {tldr}\n"
        
        result += "\n"
    
    return result


@mcp.tool()
async def get_trending(days: int = 7) -> str:
    """Get trending AI posts with high engagement (comment-to-score ratio) and AI-generated TLDRs.
    
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
        result += f"**URL:** {post['url']}\n"
        
        content = await fetch_article_content(post['url'])
        if content:
            tldr = await generate_tldr(content)
            if tldr:
                result += f"\n**TLDR:** {tldr}\n"
        
        result += "\n"
    
    return result


@mcp.tool()
async def get_news(keyword: str, days: int = 7) -> str:
    """Search for news about a specific AI topic or keyword with AI-generated TLDRs.
    
    Args:
        keyword: The topic or keyword to search for (e.g., "GPT-5", "Claude", "Gemini")
        days: Number of days to look back (default: 7)
    """
    
    errors = []
    
    try:
        reddit_posts = await search_reddit_posts(keyword=keyword, days=days, limit=5)
    except Exception as e:
        reddit_posts = []
        errors.append(f"Reddit error: {str(e)}")
    
    try:
        hn_posts = await search_hn_posts(keyword=keyword, days=days, limit=5)
    except Exception as e:
        hn_posts = []
        errors.append(f"HN error: {str(e)}")
    
    all_posts = reddit_posts + hn_posts
    all_posts.sort(key=lambda x: x["score"], reverse=True)
    
    if not all_posts:
        error_msg = f"No posts found for '{keyword}' in the past {days} day(s)."
        if errors:
            error_msg += "\n\nErrors encountered:\n" + "\n".join(errors)
        return error_msg
    
    result = f"# News about '{keyword}' - Past {days} Day(s)\n\n"
    
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
        result += f"**URL:** {post['url']}\n"
        
        content = await fetch_article_content(post['url'])
        if content:
            tldr = await generate_tldr(content)
            if tldr:
                result += f"\n**TLDR:** {tldr}\n"
        
        result += "\n"
    
    return result


@mcp.tool()
async def groq_healthcheck() -> str:
    """Check Groq connectivity and auth."""
    if groq_client is None:
        return "GROQ_API_KEY not configured."

    try:
        await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=5,
        )
        return "Groq request succeeded."
    except Exception as e:
        logging.exception(f"✗ Groq healthcheck failed: {type(e).__name__}: {str(e)}")
        return f"Groq request failed: {type(e).__name__}: {str(e)}"

if __name__ == "__main__":
    mcp.run()
