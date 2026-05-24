"""Monkey-patch OpenAI and Anthropic clients to auto-capture LLM spans."""
import time
import uuid
import json
from typing import Any

from .tracing import _record, get_trace_id, get_span_id, get_agent_name, _safe


def instrument_all():
    """Instrument all available LLM providers (safe to call even if not installed)."""
    instrument_openai()
    instrument_anthropic()


def instrument_openai():
    try:
        import openai.resources.chat.completions as _mod
        _patch_openai_sync(_mod)
    except (ImportError, AttributeError):
        pass

    try:
        import openai.resources.chat.completions as _mod
        _patch_openai_async(_mod)
    except (ImportError, AttributeError):
        pass


def instrument_anthropic():
    try:
        import anthropic.resources.messages as _mod
        _patch_anthropic_sync(_mod)
    except (ImportError, AttributeError):
        pass


# ── OpenAI sync ────────────────────────────────────────────────────────────────

def _patch_openai_sync(mod):
    if getattr(mod.Completions, "_overmind_patched", False):
        return
    original = mod.Completions.create

    def patched(self, **kwargs):
        sid = str(uuid.uuid4())
        tid = get_trace_id() or str(uuid.uuid4())
        start = time.time()
        result = error_msg = None
        try:
            result = original(self, **kwargs)
            return result
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            output = _openai_output(result)
            _record(
                name=f"openai/{kwargs.get('model', 'unknown')}",
                span_type="llm",
                input_data={"model": kwargs.get("model"), "messages": kwargs.get("messages", [])},
                output_data=output,
                error=error_msg,
                start=start,
                end=time.time(),
                metadata={"model": kwargs.get("model"), "temperature": kwargs.get("temperature")},
                span_id=sid,
                trace_id=tid,
            )

    mod.Completions.create = patched
    mod.Completions._overmind_patched = True


def _patch_openai_async(mod):
    if getattr(mod.AsyncCompletions, "_overmind_patched", False):
        return
    original = mod.AsyncCompletions.create

    async def patched(self, **kwargs):
        sid = str(uuid.uuid4())
        tid = get_trace_id() or str(uuid.uuid4())
        start = time.time()
        result = error_msg = None
        try:
            result = await original(self, **kwargs)
            return result
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            _record(
                name=f"openai/{kwargs.get('model', 'unknown')}",
                span_type="llm",
                input_data={"model": kwargs.get("model"), "messages": kwargs.get("messages", [])},
                output_data=_openai_output(result),
                error=error_msg,
                start=start,
                end=time.time(),
                metadata={"model": kwargs.get("model")},
                span_id=sid,
                trace_id=tid,
            )

    mod.AsyncCompletions.create = patched
    mod.AsyncCompletions._overmind_patched = True


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


# ── Anthropic sync ─────────────────────────────────────────────────────────────

def _patch_anthropic_sync(mod):
    if getattr(mod.Messages, "_overmind_patched", False):
        return
    original = mod.Messages.create

    def patched(self, **kwargs):
        sid = str(uuid.uuid4())
        tid = get_trace_id() or str(uuid.uuid4())
        start = time.time()
        result = error_msg = None
        try:
            result = original(self, **kwargs)
            return result
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            output = _anthropic_output(result)
            _record(
                name=f"anthropic/{kwargs.get('model', 'unknown')}",
                span_type="llm",
                input_data={"model": kwargs.get("model"), "messages": kwargs.get("messages", []),
                            "system": kwargs.get("system")},
                output_data=output,
                error=error_msg,
                start=start,
                end=time.time(),
                metadata={"model": kwargs.get("model"), "max_tokens": kwargs.get("max_tokens")},
                span_id=sid,
                trace_id=tid,
            )

    mod.Messages.create = patched
    mod.Messages._overmind_patched = True


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
