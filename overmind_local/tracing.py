"""Decorator-based tracing that writes spans to local SQLite — zero cloud."""
import functools
import inspect
import json
import time
import uuid
import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

from .storage import write_span

_local = threading.local()
_agent_name_global: Optional[str] = None


# ── Context helpers ────────────────────────────────────────────────────────────

def set_agent_name(name: str, *, global_default: bool = False):
    """Set the current agent name (thread-local).

    Pass global_default=True to also set a process-wide fallback that
    background threads (callbacks, async workers) will see when they haven't
    called set_agent_name themselves.
    """
    global _agent_name_global
    _local.agent_name = name
    if global_default:
        _agent_name_global = name


def get_agent_name() -> Optional[str]:
    return getattr(_local, "agent_name", _agent_name_global)


def get_trace_id() -> Optional[str]:
    return getattr(_local, "trace_id", None)


def get_span_id() -> Optional[str]:
    return getattr(_local, "span_id", None)


@contextmanager
def _span_ctx(span_id: str, trace_id: str):
    prev_span = getattr(_local, "span_id", None)
    prev_trace = getattr(_local, "trace_id", None)
    _local.span_id = span_id
    _local.trace_id = trace_id
    try:
        yield
    finally:
        _local.span_id = prev_span
        _local.trace_id = prev_trace


# ── Core span writer ───────────────────────────────────────────────────────────

def _record(name: str, span_type: str, input_data: Any, output_data: Any,
            error: Optional[str], start: float, end: float,
            metadata: Optional[dict] = None,
            span_id: Optional[str] = None,
            trace_id: Optional[str] = None,
            parent_id: Optional[str] = None):
    write_span({
        "id": span_id or str(uuid.uuid4()),
        "trace_id": trace_id or get_trace_id() or str(uuid.uuid4()),
        "parent_id": parent_id or get_span_id(),
        "name": name,
        "span_type": span_type,
        "agent_name": get_agent_name(),
        "input": _safe(input_data),
        "output": _safe(output_data),
        "error": error,
        "start_time": start,
        "end_time": end,
        "duration_ms": (end - start) * 1000,
        "metadata": metadata or {},
    })


def _safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _capture_inputs(fn: Callable, args: tuple, kwargs: dict) -> Any:
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return {k: v for k, v in bound.arguments.items() if k != "self"}
    except Exception:
        return {"args": _safe(args), "kwargs": _safe(kwargs)}


# ── Decorator factory ──────────────────────────────────────────────────────────

def _make_decorator(span_type: str):
    def decorator_factory(name_or_fn=None):
        def decorator(fn: Callable) -> Callable:
            span_name = (name_or_fn if isinstance(name_or_fn, str) else None) or fn.__name__

            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def wrapper(*args, **kwargs):
                    span_id = str(uuid.uuid4())
                    trace_id = get_trace_id() or str(uuid.uuid4())
                    input_data = _capture_inputs(fn, args, kwargs)
                    start = time.time()
                    result = None
                    error_msg = None
                    with _span_ctx(span_id, trace_id):
                        try:
                            result = await fn(*args, **kwargs)
                            return result
                        except Exception as exc:
                            error_msg = str(exc)
                            raise
                        finally:
                            _record(
                                name=span_name,
                                span_type=span_type,
                                input_data=input_data,
                                output_data=result,
                                error=error_msg,
                                start=start,
                                end=time.time(),
                                span_id=span_id,
                                trace_id=trace_id,
                            )
            else:
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    span_id = str(uuid.uuid4())
                    trace_id = get_trace_id() or str(uuid.uuid4())
                    input_data = _capture_inputs(fn, args, kwargs)
                    start = time.time()
                    result = None
                    error_msg = None
                    with _span_ctx(span_id, trace_id):
                        try:
                            result = fn(*args, **kwargs)
                            return result
                        except Exception as exc:
                            error_msg = str(exc)
                            raise
                        finally:
                            _record(
                                name=span_name,
                                span_type=span_type,
                                input_data=input_data,
                                output_data=result,
                                error=error_msg,
                                start=start,
                                end=time.time(),
                                span_id=span_id,
                                trace_id=trace_id,
                            )

            return wrapper

        # Allow @observe, @observe(), and @observe("name")
        if callable(name_or_fn):
            return decorator(name_or_fn)
        return decorator

    return decorator_factory


observe = _make_decorator("function")
tool = _make_decorator("tool")
workflow = _make_decorator("workflow")
entry_point = _make_decorator("entry_point")
