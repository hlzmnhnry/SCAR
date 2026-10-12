"""Strict construction of (nested) dataclasses from YAML/JSON mappings."""

from __future__ import annotations

import dataclasses
import difflib
import types
import typing
from datetime import date
from typing import Any, Union


class ConfigError(ValueError):
    pass


def _is_optional(tp) -> tuple[bool, Any]:
    origin = typing.get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) < len(typing.get_args(tp)):
            return True, (args[0] if len(args) == 1 else Union[tuple(args)])  # noqa: UP007
    return False, tp


def _convert(value: Any, tp: Any, path: str) -> Any:
    optional, inner = _is_optional(tp)
    if value is None:
        if optional or tp is Any:
            return None
        raise ConfigError(f"{path}: must not be empty")
    tp = inner
    origin = typing.get_origin(tp)

    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        return from_dict(tp, value, path)
    if origin in (Union, types.UnionType):
        errors = []
        for option in typing.get_args(tp):
            try:
                return _convert(value, option, path)
            except (ConfigError, TypeError, ValueError) as error:
                errors.append(str(error))
        raise ConfigError(f"{path}: {value!r} matches none of {typing.get_args(tp)}")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list")
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{path}: expected {len(args)} values, got {len(value)}")
        return tuple(_convert(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args, strict=True)))
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list")
        return [_convert(v, typing.get_args(tp)[0], f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict or tp is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        return dict(value)
    if tp is date:
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value))
        except ValueError as error:
            raise ConfigError(f"{path}: expected a date (YYYY-MM-DD), got {value!r}") from error
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return str(value)
    return value


def from_dict(cls, data: dict, path: str = ""):
    """Build dataclass ``cls`` from ``data``; unknown keys and wrong types raise :class:`ConfigError`."""
    hints = typing.get_type_hints(cls)
    fields = {field.name: field for field in dataclasses.fields(cls) if field.init}
    unknown = [key for key in data if key not in fields]
    if unknown:
        key = unknown[0]
        suggestion = difflib.get_close_matches(key, fields, n=1)
        hint = f" (did you mean '{suggestion[0]}'?)" if suggestion else f" (allowed: {', '.join(fields)})"
        raise ConfigError(f"{path + '.' if path else ''}{key}: unknown option{hint}")

    values = {}
    for name, field in fields.items():
        field_path = f"{path}.{name}" if path else name
        if name in data:
            values[name] = _convert(data[name], hints[name], field_path)
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{field_path}: required option missing")
    try:
        return cls(**values)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{path or cls.__name__}: {error}") from error
