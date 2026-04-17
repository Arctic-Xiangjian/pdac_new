from __future__ import annotations

from typing import Any, Dict, Mapping


def sanitize_state_dict_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for key, value in state_dict.items():
        clean_key = key
        while clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod.") :]
        clean_key = clean_key.replace("._orig_mod.", ".")
        sanitized[clean_key] = value
    return sanitized


def unwrap_state_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Checkpoint payload is not a state_dict dictionary.")


def extract_prefixed_state_dict(payload: Any, prefix: str) -> Dict[str, Any]:
    state_dict = sanitize_state_dict_keys(unwrap_state_dict(payload))
    prefix = prefix.rstrip(".")
    prefix_with_dot = f"{prefix}."
    extracted = {
        key[len(prefix_with_dot) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix_with_dot)
    }
    return extracted or state_dict
