import os
import re
from dataclasses import dataclass

import httpx
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    EqualsExpected,
    Evaluator,
    EvaluatorContext,
    IsInstance,
    LLMJudge,
)

from yipyap import generate_tldr

_groq_ssl_verify = os.environ.get("GROQ_SSL_VERIFY", "false").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_groq_ca_bundle = os.environ.get("GROQ_CA_BUNDLE") or None
_judge_model = GroqModel(
    "llama-3.3-70b-versatile",
    provider=GroqProvider(
        base_url=os.environ.get("GROQ_BASE_URL") or None,
        http_client=httpx.AsyncClient(
            verify=_groq_ca_bundle if _groq_ca_bundle else _groq_ssl_verify,
        ),
    ),
)

OPENAI_ARTICLE = """
OpenAI has released GPT-5, its most capable language model to date, featuring significant
improvements in reasoning, coding, and multimodal understanding. The model demonstrates
near-human performance on a wide range of benchmarks, including the Bar exam, medical
licensing tests, and advanced mathematics competitions. Unlike its predecessors, GPT-5 can
process and generate images, audio, and video natively without requiring separate models.
OpenAI claims the model is also significantly more efficient, reducing inference costs by
roughly 50% compared to GPT-4. The company plans to roll out access to ChatGPT Plus
subscribers first, with API access to follow within weeks. Researchers noted that GPT-5
shows markedly reduced hallucination rates, though they caution that the model is still
not fully reliable for high-stakes decisions. The training process incorporated a new
technique called constitutional distillation that helps align the model outputs with human
values more reliably than previous RLHF methods. Critics have raised concerns about the
environmental impact of training such large models, with estimates suggesting it consumed
as much electricity as a small city for several months.
""".strip()

RUST_ARTICLE = """
The Rust programming language has surpassed Go in the annual Stack Overflow developer
survey for the most-loved language for the ninth consecutive year, with 87% of respondents
expressing they would like to continue using it. Rust adoption in production environments
has grown substantially, with major companies including Google, Microsoft, and Amazon now
using it for performance-critical systems. Google has announced that new Android kernel
modules will be written in Rust rather than C to reduce memory-safety vulnerabilities,
which account for roughly 70% of all reported security bugs in the Android codebase.
Microsoft has also begun rewriting core Windows components in Rust, starting with drivers
and system services. Despite the enthusiasm, Rust's steep learning curve remains a
significant barrier, with many developers citing the borrow checker as difficult to master.
The Rust Foundation has responded by investing in improved documentation and beginner-friendly
learning resources to help widen the contributor base. Version 2.0 of the language is
expected to include ergonomic improvements aimed at reducing friction for new adopters.
""".strip()


@dataclass
class SentenceCountEvaluator(Evaluator[str, str]):
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if not ctx.output:
            return 0.0
        sentences = [s for s in re.split(r"[.!?]+\s+", ctx.output.strip()) if s.strip()]
        if 2 <= len(sentences) <= 3:
            return 1.0
        return 0.5


dataset = Dataset(
    cases=[
        Case(
            name="openai_gpt5_release",
            inputs=OPENAI_ARTICLE,
            metadata={"topic": "AI release", "words": len(OPENAI_ARTICLE.split())},
            evaluators=[
                SentenceCountEvaluator(),
                LLMJudge(
                    rubric=(
                        "The TLDR accurately captures the key points of the article "
                        "in 2-3 concise sentences without hallucinating details not "
                        "present in the input."
                    ),
                    model=_judge_model,
                    include_input=True,
                    score={"evaluation_name": "quality", "include_reason": True},
                    assertion={"evaluation_name": "accurate", "include_reason": True},
                ),
            ],
        ),
        Case(
            name="rust_adoption_article",
            inputs=RUST_ARTICLE,
            metadata={"topic": "programming language", "words": len(RUST_ARTICLE.split())},
            evaluators=[
                SentenceCountEvaluator(),
                LLMJudge(
                    rubric=(
                        "The TLDR accurately captures the key points of the article "
                        "in 2-3 concise sentences without hallucinating details not "
                        "present in the input."
                    ),
                    model=_judge_model,
                    include_input=True,
                    score={"evaluation_name": "quality", "include_reason": True},
                    assertion={"evaluation_name": "accurate", "include_reason": True},
                ),
            ],
        ),
        Case(
            name="short_content_returns_empty",
            inputs="Too short.",
            expected_output="",
            evaluators=[EqualsExpected()],
        ),
    ],
    evaluators=[IsInstance(type_name="str")],
)


if __name__ == "__main__":
    report = dataset.evaluate_sync(generate_tldr)
    report.print(include_input=False, include_output=True, include_durations=True)
