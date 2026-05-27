from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Awaitable, Callable
from types import UnionType
from typing import TypeVar, get_args, get_origin, get_type_hints

from pydantic import BaseModel

ActivityFunction = TypeVar("ActivityFunction", bound=Callable[..., Awaitable[object]])


def durable_activity(
    retries: int = 0,
    timeout: float = 60.0,
    backoff_seconds: float = 5.0,
) -> Callable[[ActivityFunction], ActivityFunction]:
    try:
        from temporal_light import activity as temporal_activity
    except ImportError:
        return _identity_decorator

    def decorator(activity_function: ActivityFunction) -> ActivityFunction:
        signature = inspect.signature(activity_function)
        type_hints = get_type_hints(activity_function)
        return_type = type_hints.get("return")

        @functools.wraps(activity_function)
        async def serialized_activity_function(
            *positional_arguments: object,
            **keyword_arguments: object,
        ) -> object:
            restored_arguments = _restore_arguments(
                signature=signature,
                type_hints=type_hints,
                positional_arguments=positional_arguments,
                keyword_arguments=keyword_arguments,
            )
            result = await activity_function(
                *restored_arguments.args,
                **restored_arguments.kwargs,
            )
            return _json_safe_value(result)

        temporal_wrapped_function = temporal_activity(
            retries=retries,
            timeout=timeout,
            backoff_seconds=backoff_seconds,
        )(serialized_activity_function)

        @functools.wraps(activity_function)
        async def typed_activity_function(
            *positional_arguments: object,
            **keyword_arguments: object,
        ) -> object:
            serialized_result = await temporal_wrapped_function(
                *[_json_safe_value(argument) for argument in positional_arguments],
                **{key: _json_safe_value(value) for key, value in keyword_arguments.items()},
            )
            return _restore_value(serialized_result, return_type)

        typed_activity_function.__activity_policy__ = temporal_wrapped_function.__activity_policy__  # type: ignore[attr-defined]
        typed_activity_function.__is_activity__ = True  # type: ignore[attr-defined]
        return typed_activity_function  # type: ignore[return-value]

    return decorator


def _identity_decorator(activity_function: ActivityFunction) -> ActivityFunction:
    return activity_function


@dataclasses.dataclass(frozen=True)
class RestoredArguments:
    args: tuple[object, ...]
    kwargs: dict[str, object]


def _restore_arguments(
    signature: inspect.Signature,
    type_hints: dict[str, object],
    positional_arguments: tuple[object, ...],
    keyword_arguments: dict[str, object],
) -> RestoredArguments:
    bound_arguments = signature.bind_partial(*positional_arguments, **keyword_arguments)
    restored_values: dict[str, object] = {}
    for argument_name, argument_value in bound_arguments.arguments.items():
        annotation = type_hints.get(argument_name)
        restored_values[argument_name] = _restore_value(argument_value, annotation)
    return RestoredArguments(args=(), kwargs=restored_values)


def _json_safe_value(value: object) -> object:
    match value:
        case BaseModel():
            return value.model_dump(mode="json")
        case list():
            return [_json_safe_value(item) for item in value]
        case tuple():
            return [_json_safe_value(item) for item in value]
        case dict():
            return {str(key): _json_safe_value(item) for key, item in value.items()}
        case _:
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return {
                    "__dataclass_type__": type(value).__name__,
                    "fields": dataclasses.asdict(value),
                }
            return value


def _restore_value(value: object, annotation: object | None) -> object:
    if annotation is None:
        return value
    if _is_base_model_type(annotation) and isinstance(value, dict):
        return annotation.model_validate(value)
    if dataclasses.is_dataclass(annotation) and isinstance(value, dict):
        return annotation(**_dataclass_fields(value))
    restored_union_value = _restore_union_value(value, annotation)
    if restored_union_value is not None:
        return restored_union_value
    return value


def _is_base_model_type(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _restore_union_value(value: object, annotation: object) -> object | None:
    if get_origin(annotation) is not UnionType and not isinstance(annotation, UnionType):
        return None
    if not isinstance(value, dict):
        return None
    dataclass_type_name = value.get("__dataclass_type__")
    if not isinstance(dataclass_type_name, str):
        return None
    for union_member in get_args(annotation):
        if not dataclasses.is_dataclass(union_member):
            continue
        if union_member.__name__ == dataclass_type_name:
            return union_member(**_dataclass_fields(value))
    return None


def _dataclass_fields(value: dict[object, object]) -> dict[str, object]:
    fields = value.get("fields", value)
    if not isinstance(fields, dict):
        raise ValueError("Serialized dataclass fields must be an object")
    return {str(key): field_value for key, field_value in fields.items()}
