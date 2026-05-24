"""Monkey-patch OpenAI and Anthropic clients to auto-capture LLM spans."""
import time
import uuid
from typing import Any

from .tracing import _record, get_trace_id, _safe


def instrument_all():
    """Instrument all available LLM providers (safe to call even if not installed)."""
    instrument_openai()
    instrument_anthropic()


def instrument_openai():
    try:
        import openai.resources.chat.completions as _mod
        _patch_cls(_mod, "Completions", is_async=False, extract_fn=_openai_output, name_prefix="openai")
        _patch_cls(_mod, "AsyncCompletions", is_async=True, extract_fn=_openai_output, name_prefix="openai")
    except (ImportError, AttributeError):
        pass


def instrument_anthropic():
    try:
        import anthropic.resources.messages as _mod
        _patch_cls(_mod, "Messages", is_async=False, extract_fn=_anthropic_output, name_prefix="anthropic")
        _patch_cls(_mod, "AsyncMessages", is_async=True, extract_fn=_anthropic_output, name_prefix="anthropic")
    except (ImportError, AttributeError):
        pass


# ── Patch factory ──────────────────────────────────────────────────────────────

def _patch_cls(mod, cls_name: str, *, is_async: bool, extract_fn, name_prefix: str):
    cls = getattr(mod, cls_name, None)
    if cls is None or getattr(cls, "_overmind_patched", False):
        return
    original = cls.create

    def _emit(kwargs, result, error_msg, start, sid, tid):
        input_data = {"model": kwargs.get("model"), "messages": kwargs.get("messages", [])}
        if "system" in kwargs:
            input_data["system"] = kwargs["system"]
        _record(
            name=f"{name_prefix}/{kwargs.get('model', 'unknown')}",
            span_type="llm",
            input_data=input_data,
            output_data=extract_fn(result),
            error=error_msg,
            start=start,
            end=time.time(),
            metadata={
                "model": kwargs.get("model"),
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
            },
            span_id=sid,
            trace_id=tid,
        )

    if is_async:
        async def patched(self, **kwargs):
            sid, tid = str(uuid.uuid4()), get_trace_id() or str(uuid.uuid4())
            start = time.time()
            result = error_msg = None
            try:
                result = await original(self, **kwargs)
                return result
            except Exception as exc:
                error_msg = str(exc)
                raise
            finally:
                _emit(kwargs, result, error_msg, start, sid, tid)
    else:
        def patched(self, **kwargs):
            sid, tid = str(uuid.uuid4()), get_trace_id() or str(uuid.uuid4())
            start = time.time()
            result = error_msg = None
            try:
                result = original(self, **kwargs)
                return result
            except Exception as exc:
                error_msg = str(exc)
                raise
            finally:
                _emit(kwargs, result, error_msg, start, sid, tid)

    cls.create = patched
    cls._overmind_patched = True


# ── Output extractors ──────────────────────────────────────────────────────────

def _openai_output(result: Any) -> dict:
    if result is None:
        return {}
    try:
        choice = result.choices[0]
        msg = choice.message
        return {
            "content": msg.content,
            "tool_calls": [
                {"name": tc.function.name, "args": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
            "finish_reason": choice.finish_reason,
            "usage": result.usage.model_dump() if result.usage else None,
            "model": result.model,
        }
    except Exception:
        return _safe(result)


def _anthropic_output(result: Any) -> dict:
    if result is None:
        return {}
    try:
        return {
            "content": next((b.text for b in result.content if hasattr(b, "text")), None),
            "stop_reason": result.stop_reason,
            "usage": {"input_tokens": result.usage.input_tokens,
                      "output_tokens": result.usage.output_tokens},
            "model": result.model,
        }
    except Exception:
        return _safe(result)
