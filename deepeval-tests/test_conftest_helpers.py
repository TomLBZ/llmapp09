import conftest


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
