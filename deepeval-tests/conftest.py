"""
Shared fixtures and configuration for deepeval LLM evaluation tests.
"""

import os
import re
from typing import Any
from typing import Sequence

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCaseParams
from pydantic import ValidationError

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_JUDGE_MODEL_NAME = os.getenv("OLLAMA_JUDGE_MODEL", "glm-5.3-flash:cloud")


def _strip_markdown_code_fence(value):
    if isinstance(value, tuple) and value:
        return (_strip_markdown_code_fence(value[0]), *value[1:])
    if not isinstance(value, str):
        return value
    return _strip_markdown_code_fence_text(value)


def _strip_markdown_code_fence_text(value: str) -> str:
    stripped = value.strip()
    match = re.fullmatch(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return value


def _remove_trailing_json_commas(value: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", value)


class _SchemaProxy:
    def __init__(self, schema):
        self._schema = schema

    def __getattr__(self, attr):
        schema = object.__getattribute__(self, "_schema")
        return getattr(schema, attr)

    def model_validate_json(self, content):
        if isinstance(content, (bytes, bytearray)):
            content = content.decode("utf-8")
        if not isinstance(content, str):
            raise TypeError("model_validate_json content must be str/bytes")
        sanitized = _strip_markdown_code_fence_text(content)
        try:
            return self._schema.model_validate_json(sanitized)
        except ValidationError as exc:
            if not any(error.get("type") == "json_invalid" for error in exc.errors()):
                raise
            repaired = _remove_trailing_json_commas(sanitized)
            if repaired != sanitized:
                try:
                    return self._schema.model_validate_json(repaired)
                except ValidationError as repair_exc:
                    raise exc from repair_exc
            raise


def _schema_with_sanitized_json(schema):
    if schema is None:
        return None
    return _SchemaProxy(schema)


class StableOllamaModel(OllamaModel):
    def generate(self, prompt, schema=None, **kwargs):
        result = super().generate(
            prompt=prompt,
            schema=_schema_with_sanitized_json(schema),
            **kwargs,
        )
        return _strip_markdown_code_fence(result)

    async def a_generate(self, prompt, schema=None, **kwargs):
        result = await super().a_generate(
            prompt=prompt,
            schema=_schema_with_sanitized_json(schema),
            **kwargs,
        )
        return _strip_markdown_code_fence(result)


def assert_test_with_retry(test_case: Any, metrics: Sequence[Any], retries: int = 1) -> None:
    for attempt in range(retries + 1):
        try:
            assert_test(test_case, metrics)
            return
        except AssertionError:
            if attempt == retries:
                raise


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
