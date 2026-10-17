import conftest
import pytest
from pydantic import BaseModel
from pydantic import ValidationError


class _Steps(BaseModel):
    steps: list[str]


def test_strip_markdown_code_fence_handles_json_fence():
    raw = '```json\n{"steps": ["a"]}\n```'
    assert conftest._strip_markdown_code_fence(raw) == '{"steps": ["a"]}'


def test_strip_markdown_code_fence_handles_tuple_result():
    raw = ('```json\n{"ok": true}\n```', 0.01)
    assert conftest._strip_markdown_code_fence(raw) == ('{"ok": true}', 0.01)


def test_strip_markdown_code_fence_keeps_plain_text():
    raw = '{"steps": ["a"]}'
    assert conftest._strip_markdown_code_fence(raw) == raw


def test_strip_markdown_code_fence_keeps_plain_tuple_text():
    raw = ('{"ok": true}', 0.01)
    assert conftest._strip_markdown_code_fence(raw) == raw


def test_schema_proxy_handles_fenced_json_for_schema():
    raw = '```json\n{"steps": ["first", "second"]}\n```'
    parsed = conftest._schema_with_sanitized_json(_Steps).model_validate_json(raw)
    assert parsed.steps == ["first", "second"]


def test_schema_proxy_none_passthrough():
    assert conftest._schema_with_sanitized_json(None) is None


def test_schema_proxy_rejects_non_string_json_content():
    with pytest.raises(TypeError):
        conftest._schema_with_sanitized_json(_Steps).model_validate_json(123)


def test_schema_proxy_repairs_trailing_comma_json():
    raw = '{\n  "steps": ["first", "second"],\n}'
    parsed = conftest._schema_with_sanitized_json(_Steps).model_validate_json(raw)
    assert parsed.steps == ["first", "second"]


def test_schema_proxy_raises_when_repaired_json_still_invalid():
    raw = '{"steps": ["first", ], "extra": }'
    with pytest.raises(ValidationError):
        conftest._schema_with_sanitized_json(_Steps).model_validate_json(raw)


def test_schema_proxy_raises_when_invalid_json_has_no_trailing_commas():
    raw = '{"steps": ["first", "second"'
    with pytest.raises(ValidationError):
        conftest._schema_with_sanitized_json(_Steps).model_validate_json(raw)


def test_assert_test_with_retry_retries_once_and_succeeds(monkeypatch):
    calls = {"count": 0}

    def flaky_assert_test(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise AssertionError("flaky first failure")

    monkeypatch.setattr(conftest, "assert_test", flaky_assert_test)
    conftest.assert_test_with_retry(object(), [object()], retries=1)

    assert calls["count"] == 2


def test_assert_test_with_retry_raises_after_retries(monkeypatch):
    calls = {"count": 0}

    def always_fail_assert_test(*_args, **_kwargs):
        calls["count"] += 1
        raise AssertionError("persistent failure")

    monkeypatch.setattr(conftest, "assert_test", always_fail_assert_test)
    with pytest.raises(AssertionError, match="persistent failure"):
        conftest.assert_test_with_retry(object(), [object()], retries=1)

    assert calls["count"] == 2
