"""
Shared fixtures and configuration for deepeval LLM evaluation tests.
"""

import os
import re

import pytest
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCaseParams

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_JUDGE_MODEL_NAME = os.getenv("OLLAMA_JUDGE_MODEL", "glm-5.3-flash:cloud")


def _strip_markdown_code_fence(value):
    if isinstance(value, tuple) and value:
        return (_strip_markdown_code_fence(value[0]), *value[1:])
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    match = re.fullmatch(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return value


class StableOllamaModel(OllamaModel):
    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        return _strip_markdown_code_fence(result)

    async def a_generate(self, *args, **kwargs):
        result = await super().a_generate(*args, **kwargs)
        return _strip_markdown_code_fence(result)


OLLAMA_JUDGE_MODEL = StableOllamaModel(
    model=OLLAMA_JUDGE_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
)


# ---------------------------------------------------------------------------
# Reusable GEval metric factories
# ---------------------------------------------------------------------------

def json_schema_metric(schema_description: str):
    """Creates a GEval metric that checks JSON schema compliance."""
    return GEval(
        name="JSON Schema Compliance",
        criteria=(
            "Evaluate whether the actual output is valid JSON that conforms to "
            "the required schema. Only check structure, key names, and data "
            "types — do NOT penalize for specific values. "
            + schema_description
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=OLLAMA_JUDGE_MODEL,
    )


def output_correctness_metric():
    """Creates a GEval metric that checks factual/logical correctness."""
    return GEval(
        name="Output Correctness",
        criteria=(
            "Determine whether the actual output is logically correct and "
            "reasonable given the input text. The analysis should make sense "
            "for the provided input."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=OLLAMA_JUDGE_MODEL,
    )


def answer_relevancy_metric():
    """Creates a GEval metric that checks whether the output is topically
    relevant to the input.  Unlike AnswerRelevancyMetric (which assumes a
    Q&A format), this works for classification and analysis endpoints where
    the output is structured metadata about the input text."""
    return GEval(
        name="Answer Relevancy",
        criteria=(
            "Evaluate whether the actual output is topically relevant to the "
            "input text. The labels, categories, or analysis in the output "
            "should directly relate to the subject matter of the input. "
            "Structured metadata (labels, categories, confidence scores) that "
            "accurately describes the input text should be considered relevant."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=OLLAMA_JUDGE_MODEL,
    )
