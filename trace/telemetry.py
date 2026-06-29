"""
Console tracing helpers.

Tracing is disabled by default and is enabled only inside trace_context().
The decorator keeps instrumentation local to function boundaries so normal
non-trace runs remain quiet.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Callable


_trace_enabled: ContextVar[bool] = ContextVar("re0rag_trace_enabled", default=False)
_trace_depth: ContextVar[int] = ContextVar("re0rag_trace_depth", default=0)

AttributeExtractor = Callable[[tuple[Any, ...], dict[str, Any], Any], dict[str, Any]]


@contextmanager
def trace_context(enabled: bool = True):
    """Enable console trace printing within the current execution context."""
    enabled_token = _trace_enabled.set(enabled)
    depth_token = _trace_depth.set(0)
    try:
        yield
    finally:
        _trace_depth.reset(depth_token)
        _trace_enabled.reset(enabled_token)


def is_trace_enabled() -> bool:
    return _trace_enabled.get()


def trace_span(
    name: str | None = None,
    attributes: AttributeExtractor | None = None,
):
    """
    Decorate a function and print its elapsed time when tracing is enabled.

    attributes receives (args, kwargs, result) after a successful call and may
    return a small dict of values to append to the trace line.
    """
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not is_trace_enabled():
                    return await func(*args, **kwargs)
                token = _trace_depth.set(_trace_depth.get() + 1)
                start = perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    _print_trace(span_name, start, args, kwargs, result, attributes)
                    return result
                except Exception as exc:
                    _print_trace(span_name, start, args, kwargs, None, attributes, exc)
                    raise
                finally:
                    _trace_depth.reset(token)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_trace_enabled():
                return func(*args, **kwargs)
            token = _trace_depth.set(_trace_depth.get() + 1)
            start = perf_counter()
            try:
                result = func(*args, **kwargs)
                _print_trace(span_name, start, args, kwargs, result, attributes)
                return result
            except Exception as exc:
                _print_trace(span_name, start, args, kwargs, None, attributes, exc)
                raise
            finally:
                _trace_depth.reset(token)

        return wrapper

    return decorator


def _print_trace(
    span_name: str,
    start: float,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    attributes: AttributeExtractor | None,
    exc: Exception | None = None,
) -> None:
    elapsed_ms = (perf_counter() - start) * 1000
    depth = max(_trace_depth.get() - 1, 0)
    indent = "  " * depth
    status = "ERROR" if exc else "OK"
    attrs = _safe_extract_attributes(attributes, args, kwargs, result)
    if exc:
        attrs["error.type"] = exc.__class__.__name__
    attr_text = _format_attributes(attrs)
    print(f"[trace] {indent}{span_name} {status} {elapsed_ms:.2f}ms{attr_text}")


def _safe_extract_attributes(
    attributes: AttributeExtractor | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    if attributes is None:
        return _default_attributes(result)
    try:
        return {**_default_attributes(result), **(attributes(args, kwargs, result) or {})}
    except Exception as exc:
        return {"trace.attributes_error": exc.__class__.__name__}


def _default_attributes(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    attrs: dict[str, Any] = {}
    if "selected_tool" in result:
        attrs["selected_tool"] = result.get("selected_tool")
    if "retry_count" in result:
        attrs["retry_count"] = result.get("retry_count")
    if "documents" in result:
        attrs["documents.count"] = len(result.get("documents") or [])
    if "evidence" in result:
        attrs["evidence.count"] = len(result.get("evidence") or [])
    if "sources" in result:
        attrs["sources.count"] = len(result.get("sources") or [])
    if "answer" in result:
        attrs["answer.length"] = len(result.get("answer") or "")
    return attrs


def _format_attributes(attrs: dict[str, Any]) -> str:
    if not attrs:
        return ""
    parts = []
    for key, value in attrs.items():
        if isinstance(value, float):
            value = round(value, 4)
        parts.append(f"{key}={value}")
    return " " + " ".join(parts)

