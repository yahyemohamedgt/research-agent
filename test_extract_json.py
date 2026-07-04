from dotenv import load_dotenv

load_dotenv()

from graph import _extract_json

assert _extract_json('{"items": []}') == {"items": []}
assert _extract_json('Here is the JSON:\n{"items": [{"text": "hi"}]}') == {"items": [{"text": "hi"}]}
assert _extract_json('```json\n{"items": []}\n```') == {"items": []}
assert _extract_json('no json here') == {}
assert _extract_json('') == {}

print("ok")
