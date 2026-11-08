
from typing import Any
from pydantic import BaseModel

_GEMINI_JSON_SCHEMA_KEYS = frozenset({
    '$id', '$defs', '$ref', '$anchor',
    'type', 'format', 'title', 'description', 'enum',
    'items', 'prefixItems', 'minItems', 'maxItems',
    'minimum', 'maximum', 'anyOf', 'oneOf',
    'properties', 'additionalProperties', 'required',
})


def _filter_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_filter_schema(item) for item in value]

    if not isinstance(value, dict):
        return value

    clean: dict[str, Any] = {}

    for key, item in value.items():
        if key == 'const':
            clean['enum'] = [item]

        elif key in ('$defs', 'properties') and isinstance(item, dict):
            clean[key] = {
                name: _filter_schema(child)
                for name, child in item.items()
            }

        elif key in _GEMINI_JSON_SCHEMA_KEYS:
            clean[key] = _filter_schema(item)

    return clean


def gemini_json_schema(
    model: type[BaseModel]
) -> dict[str, Any]:
    return _filter_schema(model.model_json_schema())
