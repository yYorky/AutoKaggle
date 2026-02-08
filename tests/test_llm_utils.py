from autokaggle.llm_utils import extract_json


def test_extract_json_handles_code_fences() -> None:
    payload = """```json
    {"status": "ok"}
    ```"""
    assert extract_json(payload) == {"status": "ok"}


def test_extract_json_repairs_missing_commas_in_arrays() -> None:
    payload = '{"files":[{"path":"a","content":"x"}{"path":"b","content":"y"}]}'
    result = extract_json(payload)
    assert result["files"][1]["path"] == "b"


def test_extract_json_repairs_trailing_commas() -> None:
    payload = '{"a": 1, "b": [1, 2,],}'
    assert extract_json(payload) == {"a": 1, "b": [1, 2]}
