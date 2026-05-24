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

def set_agent_name(name: str):
    """Set the current agent name (thread-local, affects all spans on this thread)."""
    _local.agent_name = name


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


# ── Decorator factory ──────────────────────────────────────────────────────────

def _make_decorator(span_type: str):
    def decorator_factory(name: Optional[str] = None):
        def decorator(fn: Callable) -> Callable:
            span_name = name or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                span_id = str(uuid.uuid4())
                trace_id = get_trace_id() or str(uuid.uuid4())

                # Capture inputs
                try:
                    sig = inspect.signature(fn)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    input_data = {k: v for k, v in bound.arguments.items() if k != "self"}
                except Exception:
                    input_data = {"args": _safe(args), "kwargs": _safe(kwargs)}

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

        # Allow both `@observe` and `@observe()` and `@observe("name")`
        if callable(name):
            fn, name = name, None
            return decorator(fn)
        return decorator

    return decorator_factory


observe = _make_decorator("function")
tool = _make_decorator("tool")
workflow = _make_decorator("workflow")
entry_point = _make_decorator("entry_point")
