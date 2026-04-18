"""
inference.py — Pocket-Agent grader interface.

Exposes:  run(prompt: str, history: list[dict]) -> str

Loads the quantized model from ./artifacts/ on first import.
No network calls are made at any point.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Paths — set these to match the notebook's ARTIFACTS_DIR
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).parent
_ARTIFACTS_DIR = _BASE_DIR / "artifacts"
_QUANTIZED_DIR = _ARTIFACTS_DIR / "quantized_model"

# ---------------------------------------------------------------------------
# System prompt (must match what the model was trained on)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Pocket-Agent, a concise on-device assistant with access to exactly five tools. \
When the user's request clearly matches a tool, respond ONLY with a JSON object wrapped in <tool_call>...</tool_call> tags. \
When no tool fits (chitchat, impossible requests, or ambiguous references with no conversation history), respond with plain natural language — no tool call.

Available tools and their required arguments:
  weather  : {"location": "<city>", "unit": "C" or "F"}
  calendar : {"action": "list" or "create", "date": "YYYY-MM-DD", "title": "<string, create only>"}
  convert  : {"value": <number>, "from_unit": "<unit>", "to_unit": "<unit>"}
  currency : {"amount": <number>, "from": "<ISO3>", "to": "<ISO3>"}
  sql      : {"query": "<SQL string>"}

Tool call format (emit exactly this, nothing else before or after):
<tool_call>{"tool": "tool_name", "args": {...}}</tool_call>"""

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_device = "cpu"


def _load_model():
    global _tokenizer, _model

    if _tokenizer is not None:
        return

    model_path = str(_QUANTIZED_DIR)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Artifacts not found at {_QUANTIZED_DIR}. "
            "Run all cells in pocket_agent_colab.ipynb to generate artifacts."
        )

    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    _model.eval()

    # Apply int8 dynamic quantization for faster CPU inference
    _model = torch.quantization.quantize_dynamic(
        _model, {torch.nn.Linear}, dtype=torch.qint8
    )


def _format_messages(history: list[dict], prompt: str) -> list[dict]:
    """Build a messages list suitable for apply_chat_template."""
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


def run(prompt: str, history: list[dict]) -> str:
    """
    Generate a response for the given prompt given conversation history.

    Args:
        prompt:  The latest user message.
        history: List of prior turns as dicts with 'role' and 'content' keys.
                 Roles must be 'user' or 'assistant'.

    Returns:
        Raw model output — either a <tool_call>...</tool_call> JSON string
        or a plain-text refusal.
    """
    _load_model()

    messages = _format_messages(history, prompt)

    enc = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(enc, torch.Tensor):
        enc = enc.to(_device)
        prompt_len = enc.shape[1]
        with torch.no_grad():
            output_ids = _model.generate(
                enc,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )
    else:
        enc = enc.to(_device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = _model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )

    new_tokens = output_ids[0, prompt_len:]
    response = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        ("What's the weather in Paris?", []),
        ("Convert 100 miles to kilometers", []),
        ("How are you doing?", []),
        (
            "Convert that to euros",
            [
                {"role": "user", "content": "I have 500 dollars."},
                {"role": "assistant", "content": "Got it — 500 USD noted. Want to convert?"},
            ],
        ),
    ]
    for prompt, history in test_cases:
        print(f"\nUser: {prompt}")
        if history:
            print(f"(History: {len(history)} prior turns)")
        result = run(prompt, history)
        print(f"Agent: {result}")
