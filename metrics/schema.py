"""Shared CAD JSON schemas."""
from __future__ import annotations

from typing import Any, Dict

OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "object",
        "function",
        "global_shape",
        "components",
        "interfaces",
        "working_principle",
        "dimensions",
        "materials_guess",
        "manufacturing",
        "assembly_hypothesis",
        "assumptions",
        "uncertainties",
        "risks_limits",
    ],
    "properties": {
        "object": {"type": "string"},
        "function": {"type": "string"},
        "global_shape": {"type": "string"},
        "components": {"type": "array", "items": {"type": "string"}},
        "interfaces": {"type": "array", "items": {"type": "string"}},
        "working_principle": {"type": "string"},
        "dimensions": {
            "type": "object",
            "required": ["H", "W", "D", "unit"],
            "properties": {
                "H": {"type": "number"},
                "W": {"type": "number"},
                "D": {"type": "number"},
                "unit": {"type": "string"},
            },
        },
        "materials_guess": {"type": "array", "items": {"type": "string"}},
        "manufacturing": {"type": "array", "items": {"type": "string"}},
        "assembly_hypothesis": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "risks_limits": {"type": "array", "items": {"type": "string"}},
    },
}

CAD_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["images", "context_text", "instruction", "output"],
    "properties": {
        "images": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "context_text": {"type": "string"},
        "instruction": {"type": "string"},
        "output": OUTPUT_SCHEMA,
    },
}
