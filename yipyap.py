import os
import logfire
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Literal, NotRequired, TypedDict

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import AsyncGroq
from mcp.server.fastmcp import FastMCP

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

logfire.configure(
    service_name="yipyap",
    advanced=logfire.AdvancedOptions(base_url="https://logfire.data-platform.qa.ckotech.internal"),
)
logfire.instrument_httpx()

mcp = FastMCP("yipyap")

USER_AGENT = "yipyap/0.1.0"
FALSEY = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in FALSEY


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


HTTP_VERIFY = _env_bool("YIPYAP_HTTP_VERIFY", False)
HTTP_TIMEOUT_SECONDS = _env_float("YIPYAP_HTTP_TIMEOUT_SECONDS", 30.0)
ARTICLE_FETCH_TIMEOUT_SECONDS = _env_float("YIPYAP_ARTICLE_FETCH_TIMEOUT_SECONDS", 15.0)
HTTP_CA_BUNDLE = (
    os.environ.get("YIPYAP_HTTP_CA_BUNDLE")
    or os.environ.get("SSL_CERT_FILE")
    or None
)

HTTP_VERIFY_VALUE: str | bool = HTTP_CA_BUNDLE if HTTP_CA_BUNDLE else HTTP_VERIFY


class Post(TypedDict):
    title: str
    url: str
    score: int
    comments: int
    created: int
    source: Literal["Reddit", "HackerNews"]
    subreddit: NotRequired[str]
    ratio: NotRequired[float]


groq_api_key = os.environ.get("GROQ_API_KEY")
if groq_api_key:
    logfire.info("✓ GROQ_API_KEY loaded")
else:
    logfire.error("✗ GROQ_API_KEY not found in environment variables")

groq_base_url = os.environ.get("GROQ_BASE_URL") or None
groq_timeout_seconds = _env_float("GROQ_TIMEOUT_SECONDS", 30.0)
groq_max_retries = _env_int("GROQ_MAX_RETRIES", 4)
groq_ssl_verify = _env_bool("GROQ_SSL_VERIFY", False)
groq_ca_bundle = os.environ.get("GROQ_CA_BUNDLE") or None

groq_http_client = None
if groq_api_key:
    groq_http_client = httpx.AsyncClient(
        timeout=groq_timeout_seconds,
        verify=groq_ca_bundle if groq_ca_bundle else groq_ssl_verify,
    )

groq_client: AsyncGroq | None = None
if groq_api_key:
    groq_client = AsyncGroq(
        api_key=groq_api_key,
        base_url=groq_base_url,
        max_retries=groq_max_retries,
        http_client=groq_http_client,
    )

SUMMARIZATION_PROMPT = """Summarize the following article in 2-3 concise sentences. Focus on the key points and main takeaways.

Article:
{content}

TLDR:"""

SUBREDDITS: list[str] = [
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
    "ControlProblem",
    "technology",
]

def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_photo_only_post(post: dict[str, object]) -> bool:
    post_hint = _as_str(post.get("post_hint", ""))
    if post_hint == "image":
        return True

    is_self = bool(post.get("is_self", False))
    selftext = _as_str(post.get("selftext", ""))
    if is_self and not selftext:
        return True

    url = _as_str(post.get("url", ""))
    if not url or selftext:
        return False

    image_domains = ("i.redd.it", "imgur.com", "i.imgur.com")
    return any(domain in url for domain in image_domains)


def _reddit_post_to_post(post: dict[str, object], subreddit: str) -> Post:
    return {
        "title": _as_str(post.get("title", "")),
        "url": _as_str(post.get("url", "")),
        "score": _as_int(post.get("score", 0)),
        "comments": _as_int(post.get("num_comments", 0)),
        "created": _as_int(post.get("created_utc", 0)),
        "source": "Reddit",
        "subreddit": subreddit,
    }


def _hn_hit_to_post(hit: dict[str, object]) -> Post:
    object_id = _as_str(hit.get("objectID", ""))
    url = _as_str(hit.get("url", ""))
    if not url:
        url = f"https://news.ycombinator.com/item?id={object_id}"

    return {
        "title": _as_str(hit.get("title", "")),
        "url": url,
        "score": _as_int(hit.get("points", 0)),
        "comments": _as_int(hit.get("num_comments", 0)),
        "created": _as_int(hit.get("created_at_i", 0)),
        "source": "HackerNews",
    }


@logfire.instrument("fetch_article_content {url}")
async def fetch_article_content(url: str) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=ARTICLE_FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            verify=HTTP_VERIFY_VALUE,
            headers={"User-Agent": USER_AGENT},
        ) as client:
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
            logfire.info("✓ Fetched content from {url}", url=url[:50], content_length=len(content))
            return content
    except Exception as e:
        logfire.error("✗ Failed to fetch {url}", url=url[:50], exc_type=type(e).__name__, exc_msg=str(e))
        return ""


@logfire.instrument("generate_tldr")
async def generate_tldr(content: str) -> str:
    if not content or len(content) < 100:
        logfire.warn("✗ Content too short for TLDR", content_length=len(content))
        return ""
    try:
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
        logfire.info("✓ Generated TLDR: {preview}", preview=tldr[:50])
        return tldr
    except Exception as e:
        logfire.exception("✗ Failed to generate TLDR")
        return ""


async def _fetch_reddit(
    path: str,
    *,
    days: int,
    params: dict[str, str | int],
) -> list[Post]:
    threshold = datetime.now() - timedelta(days=days)
    threshold_ts = threshold.timestamp()

    posts: list[Post] = []
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        verify=HTTP_VERIFY_VALUE,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for subreddit in SUBREDDITS:
            response = await client.get(
                f"https://www.reddit.com/r/{subreddit}/{path}",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            children = data.get("data", {}).get("children", [])
            for child in children:
                raw = child.get("data", {})
                if not isinstance(raw, dict):
                    continue
                post = raw
                created_utc = post.get("created_utc", 0)
                if float(created_utc or 0) < threshold_ts:
                    continue
                if is_photo_only_post(post):
                    continue
                posts.append(_reddit_post_to_post(post, subreddit=subreddit))

    return posts


async def fetch_reddit_posts(days: int = 7, limit: int = 5) -> list[Post]:
    posts = await _fetch_reddit("top.json", days=days, params={"limit": 100, "t": "week"})
    posts.sort(key=lambda x: x["score"], reverse=True)
    return posts[:limit]


async def fetch_hn_posts(days: int = 7, limit: int = 5) -> list[Post]:
    seven_days_ago = datetime.now() - timedelta(days=days)
    seven_days_ago_timestamp = int(seven_days_ago.timestamp())

    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        verify=HTTP_VERIFY_VALUE,
        headers={"User-Agent": USER_AGENT},
    ) as client:
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

        posts: list[Post] = []
        for hit in data.get("hits", []):
            if not isinstance(hit, dict):
                continue
            posts.append(_hn_hit_to_post(hit))

        return posts


async def search_reddit_posts(keyword: str, days: int = 7, limit: int = 5) -> list[Post]:
    posts = await _fetch_reddit(
        "search.json",
        days=days,
        params={
            "q": keyword,
            "restrict_sr": 1,
            "sort": "top",
            "t": "all",
            "limit": 25,
        },
    )
    posts.sort(key=lambda x: x["score"], reverse=True)
    return posts[:limit]


async def search_hn_posts(keyword: str, days: int = 7, limit: int = 5) -> list[Post]:
    days_ago = datetime.now() - timedelta(days=days)
    days_ago_timestamp = int(days_ago.timestamp())

    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        verify=HTTP_VERIFY_VALUE,
        headers={"User-Agent": USER_AGENT},
    ) as client:
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

        posts: list[Post] = []
        for hit in data.get("hits", []):
            if not isinstance(hit, dict):
                continue
            posts.append(_hn_hit_to_post(hit))

        return posts


async def fetch_reddit_controversial(days: int = 7, limit: int = 5) -> list[Post]:
    if days <= 1:
        time_param = "day"
    elif days <= 7:
        time_param = "week"
    else:
        time_param = "month"

    posts = await _fetch_reddit(
        "controversial.json",
        days=days,
        params={"limit": 100, "t": time_param},
    )
    posts.sort(key=lambda x: x["comments"], reverse=True)
    return posts[:limit]


async def _safe_posts(label: str, call: Awaitable[list[Post]]) -> tuple[list[Post], str | None]:
    try:
        return await call, None
    except Exception as e:
        return [], f"{label} error: {str(e)}"


def _source_detail(post: Post) -> str:
    if post["source"] == "Reddit":
        subreddit = post.get("subreddit", "")
        return f"Reddit (r/{subreddit})" if subreddit else "Reddit"
    return "HackerNews"


def _score_line(post: Post, *, include_ratio: bool) -> str:
    line = f"**Score:** {post['score']} points | {post['comments']} comments"
    if include_ratio and "ratio" in post:
        line += f" (ratio: {post['ratio']:.2f})"
    return line


def _render_post_block(i: int, post: Post, *, include_ratio: bool) -> str:
    created_date = datetime.fromtimestamp(post["created"]).strftime("%Y-%m-%d")
    lines = [
        f"## {i}. {post['title']}\n",
        f"**Source:** {_source_detail(post)}\n",
        f"{_score_line(post, include_ratio=include_ratio)}\n",
        f"**Date:** {created_date}\n",
        f"**URL:** {post['url']}\n",
    ]
    return "".join(lines)


@mcp.tool()
async def summarise_weekly() -> str:
    """Get top tech news from the past week across Reddit and Hacker News with AI-generated TLDRs."""
    with logfire.span("summarise_weekly"):
        errors: list[str] = []
        reddit_posts, reddit_err = await _safe_posts("Reddit", fetch_reddit_posts(days=7, limit=5))
        hn_posts, hn_err = await _safe_posts("HN", fetch_hn_posts(days=7, limit=5))
        for err in (reddit_err, hn_err):
            if err:
                errors.append(err)

        all_posts: list[Post] = reddit_posts + hn_posts
        all_posts.sort(key=lambda x: x["score"], reverse=True)

        if not all_posts:
            error_msg = "No posts found in the past week."
            if errors:
                error_msg += "\n\nErrors encountered:\n" + "\n".join(errors)
            return error_msg

        parts: list[str] = ["# Top Tech News - Past Week\n\n"]

        for i, post in enumerate(all_posts[:10], 1):
            parts.append(_render_post_block(i, post, include_ratio=False))
            content = await fetch_article_content(post["url"])
            if content:
                tldr = await generate_tldr(content)
                if tldr:
                    parts.append(f"\n**TLDR:** {tldr}\n")
            parts.append("\n")

        return "".join(parts)


@mcp.tool()
async def get_drama(days: int = 7) -> str:
    """Get controversial/heated AI discussions from Reddit with AI-generated TLDRs."""
    with logfire.span("get_drama", days=days):
        try:
            drama_posts = await fetch_reddit_controversial(days=days, limit=5)
        except Exception as e:
            return f"Error fetching controversial posts: {str(e)}"

        if not drama_posts:
            return f"No controversial posts found in the past {days} day(s)."

        parts: list[str] = [f"# Controversial AI Discussions - Past {days} Day(s)\n\n"]
        for i, post in enumerate(drama_posts, 1):
            parts.append(_render_post_block(i, post, include_ratio=False))
            content = await fetch_article_content(post["url"])
            if content:
                tldr = await generate_tldr(content)
                if tldr:
                    parts.append(f"\n**TLDR:** {tldr}\n")
            parts.append("\n")

        return "".join(parts)


@mcp.tool()
async def get_trending(days: int = 7) -> str:
    """Get trending AI posts with high engagement and AI-generated TLDRs."""
    with logfire.span("get_trending", days=days):
        errors: list[str] = []

        reddit_posts, reddit_err = await _safe_posts("Reddit", fetch_reddit_posts(days=days, limit=100))
        if reddit_err:
            errors.append(reddit_err)
        for post in reddit_posts:
            post["ratio"] = post["comments"] / max(post["score"], 1)
        reddit_posts.sort(key=lambda x: x.get("ratio", 0.0), reverse=True)
        reddit_trending = reddit_posts[:5]

        hn_posts, hn_err = await _safe_posts("HN", fetch_hn_posts(days=days, limit=100))
        if hn_err:
            errors.append(hn_err)
        for post in hn_posts:
            post["ratio"] = post["comments"] / max(post["score"], 1)
        hn_posts.sort(key=lambda x: x.get("ratio", 0.0), reverse=True)
        hn_trending = hn_posts[:5]

        all_posts: list[Post] = reddit_trending + hn_trending
        all_posts.sort(key=lambda x: x.get("ratio", 0.0), reverse=True)

        if not all_posts:
            error_msg = f"No trending posts found in the past {days} day(s)."
            if errors:
                error_msg += "\n\nErrors encountered:\n" + "\n".join(errors)
            return error_msg

        parts: list[str] = [
            f"# Trending AI Discussions - Past {days} Day(s)\n\n",
            "*Posts with high engagement (lots of discussion relative to upvotes)*\n\n",
        ]

        for i, post in enumerate(all_posts[:10], 1):
            parts.append(_render_post_block(i, post, include_ratio=True))
            content = await fetch_article_content(post["url"])
            if content:
                tldr = await generate_tldr(content)
                if tldr:
                    parts.append(f"\n**TLDR:** {tldr}\n")
            parts.append("\n")

        return "".join(parts)


@mcp.tool()
async def get_news(keyword: str, days: int = 7) -> str:
    """Search for news about a keyword with AI-generated TLDRs."""
    with logfire.span("get_news", keyword=keyword, days=days):
        errors: list[str] = []

        reddit_posts, reddit_err = await _safe_posts(
            "Reddit", search_reddit_posts(keyword=keyword, days=days, limit=5)
        )
        hn_posts, hn_err = await _safe_posts("HN", search_hn_posts(keyword=keyword, days=days, limit=5))
        for err in (reddit_err, hn_err):
            if err:
                errors.append(err)

        all_posts: list[Post] = reddit_posts + hn_posts
        all_posts.sort(key=lambda x: x["score"], reverse=True)

        if not all_posts:
            error_msg = f"No posts found for '{keyword}' in the past {days} day(s)."
            if errors:
                error_msg += "\n\nErrors encountered:\n" + "\n".join(errors)
            return error_msg

        parts: list[str] = [f"# News about '{keyword}' - Past {days} Day(s)\n\n"]
        for i, post in enumerate(all_posts[:10], 1):
            parts.append(_render_post_block(i, post, include_ratio=False))
            content = await fetch_article_content(post["url"])
            if content:
                tldr = await generate_tldr(content)
                if tldr:
                    parts.append(f"\n**TLDR:** {tldr}\n")
            parts.append("\n")

        return "".join(parts)

if __name__ == "__main__":
    mcp.run()
