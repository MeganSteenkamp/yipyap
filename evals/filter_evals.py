from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

from yipyap import is_photo_only_post


async def task(post: dict[str, object]) -> bool:
    return is_photo_only_post(post)


dataset = Dataset(
    cases=[
        Case(
            name="post_hint_image",
            inputs={"post_hint": "image", "url": "https://i.redd.it/abc.jpg", "title": "Cool pic"},
            expected_output=True,
        ),
        Case(
            name="self_post_no_text",
            inputs={"is_self": True, "selftext": "", "title": "Discussion thread"},
            expected_output=True,
        ),
        Case(
            name="imgur_url",
            inputs={"url": "https://imgur.com/abc123", "title": "Screenshot"},
            expected_output=True,
        ),
        Case(
            name="i_redd_it_url",
            inputs={"url": "https://i.redd.it/xyz.png", "title": "Image post"},
            expected_output=True,
        ),
        Case(
            name="normal_article_url",
            inputs={"url": "https://techcrunch.com/2025/01/01/some-article", "title": "Article"},
            expected_output=False,
        ),
        Case(
            name="url_with_selftext",
            inputs={
                "url": "https://imgur.com/abc",
                "selftext": "Check out this image I found",
                "title": "Interesting find",
            },
            expected_output=False,
        ),
    ],
    evaluators=[EqualsExpected()],
)


if __name__ == "__main__":
    report = dataset.evaluate_sync(task)
    report.print(include_input=True, include_output=True, include_durations=False)
