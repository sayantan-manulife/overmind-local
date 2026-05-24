# overmind-local

Self-hosted alternative to [Overmind](https://github.com/overmind-core/overmind).  
Zero cloud. Everything lives in a local SQLite file.

## What it does

| Feature | Overmind (cloud) | overmind-local |
|---------|-----------------|----------------|
| Decorator-based tracing | ✓ | ✓ |
| Auto-instrument OpenAI / Anthropic | ✓ | ✓ |
| Production trace storage | api.overmindlab.ai | `.overmind/traces.db` (SQLite) |
| LLM-powered optimization loop | ✓ | ✓ (your API key, direct) |
| Policy constraints | ✓ | ✓ |
| Test dataset management | ✓ | ✓ |
| API key required | Yes | No |
| Internet required | Yes | No |
| Dashboard UI | console.overmindlab.ai | CLI (`overmind-local traces`) |

## Install

```bash
pip install overmind-local          # core + CLI
pip install "overmind-local[openai]"      # + OpenAI auto-instrumentation
pip install "overmind-local[anthropic]"   # + Anthropic auto-instrumentation
pip install "overmind-local[all]"         # everything
```

## Quick start

```python
import overmind_local as om

# 1. Init local storage (creates .overmind/traces.db)
om.init()

# 2. (Optional) auto-instrument your LLM clients
om.instrument_all()

# 3. Decorate your agent functions
om.set_agent_name("my-agent")

@om.entry_point()
def run_agent(query: str) -> str:
    return search(query)

@om.tool()
def search(query: str) -> str:
    # your tool logic
    ...
```

Every call is written to SQLite. No network traffic.

## CLI

```bash
# Initialise
overmind-local init

# View recent spans
overmind-local traces
overmind-local traces --agent my-agent --errors-only

# Summary stats across all agents
overmind-local stats

# Add policies (constraints the optimizer must respect)
overmind-local policy add my-agent \
  --name "no-hallucination" \
  --description "Never fabricate citations. If unsure, say so."

# Add test cases
overmind-local dataset add my-agent \
  --input '{"query": "What is the capital of France?"}' \
  --expected "Paris"

# Run the optimization loop (uses your OPENAI_API_KEY / ANTHROPIC_API_KEY)
overmind-local optimize my-agent --model gpt-4o
```

## How the optimizer works

1. Reads the last N traces from SQLite
2. Loads your policies and test dataset
3. Sends a structured analysis prompt to the LLM (via [litellm](https://github.com/BerriAI/litellm) — supports OpenAI, Anthropic, Gemini, local Ollama, etc.)
4. Returns a ranked list of concrete changes: revised system prompts, tool description fixes, retry logic improvements

The LLM call goes **directly** to your provider. No Overmind backend involved.

## Export to Langfuse / Phoenix (optional)

If you want a web UI, you can forward spans to any OpenTelemetry-compatible backend:

```python
# Use the OpenTelemetry SDK directly alongside overmind-local
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# ... configure your preferred backend
```

overmind-local writes to SQLite; you can run a separate OTEL collector pointing at
Langfuse self-hosted, Jaeger, Phoenix, Grafana Tempo, etc.

## License

MIT
