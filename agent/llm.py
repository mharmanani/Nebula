"""Local coordinator language models (Hugging Face transformers; no API calls), greedy decoding, one tool call per turn.

Presets: qwen (Qwen3.6-27B, the paper's coordinator, thinking off unless `thinking=True`), gpt-oss (gpt-oss-20b, low reasoning effort) and mistral
(Mistral-Small-3.2-24B-Instruct). Each emits tool calls in its own format; `parse` returns (text, call or None).
Only the qwen preset was used for the paper's results.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, replace


@dataclass
class Preset:
    model: str  # Hugging Face id or a local folder
    chat_kwargs: dict = field(default_factory=dict)
    raw_history: bool = False  # keep the model's own tool-call text in the history (as the paper did for Qwen)


PRESETS = {
    "qwen": Preset("Qwen/Qwen3.6-27B", {"enable_thinking": False}, raw_history=True),
    "gpt-oss": Preset("openai/gpt-oss-20b", {"reasoning_effort": "low"}),
    "mistral": Preset("mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
}
THINKING_TOKENS = 4000  # extra generation budget per turn when thinking is on (the thinking is stripped by `parse`)

_QWEN_XML = re.compile(r"<tool_call>\s*<function=(?P<name>[^>]+)>(?P<body>.*?)</function>\s*</tool_call>", re.S)
_QWEN_PARAM = re.compile(r"<parameter=(?P<key>[^>]+)>\s*(?P<value>.*?)\s*</parameter>", re.S)
_QWEN_JSON = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_HARMONY_CALL = re.compile(r"to=functions\.(?P<name>[\w\-]+).*?<\|message\|>(?P<args>.*?)(?:<\|call\|>|$)", re.S)
_HARMONY_FINAL = re.compile(r"<\|channel\|>final<\|message\|>(?P<text>.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)


def _arguments(text: str) -> dict:
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def parse_qwen(raw: str):
    text = raw.split("</think>")[-1].strip()
    m = _QWEN_XML.search(text)
    if m:
        return text, {"name": m.group("name").strip(), "arguments": dict(_QWEN_PARAM.findall(m.group("body")))}
    m = _QWEN_JSON.search(text)
    d = _arguments(m.group(1)) if m else {}
    return text, ({"name": str(d["name"]).strip(), "arguments": d.get("arguments") or {}} if "name" in d else None)


def parse_gpt_oss(raw: str):
    m = _HARMONY_CALL.search(raw)
    if m:
        return "", {"name": m.group("name"), "arguments": _arguments(m.group("args").strip())}
    f = _HARMONY_FINAL.search(raw)
    return (f.group("text") if f else re.sub(r"<\|[^|]+\|>", " ", raw)).strip(), None


def parse_mistral(raw: str):
    text, _, call = raw.partition("[TOOL_CALLS]")
    if not call:
        return text.strip(), None
    if "[ARGS]" in call:  # name[ARGS]{...}
        name, _, args = call.partition("[ARGS]")
        return text.strip(), {"name": name.strip(), "arguments": _arguments(args.strip().split("</s>")[0])}
    try:  # [{"name": ..., "arguments": {...}}]
        d = json.loads(call.strip().split("</s>")[0])[0]
        return text.strip(), {"name": d["name"], "arguments": d.get("arguments") or {}}
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return text.strip(), None


PARSERS = {"qwen": parse_qwen, "gpt-oss": parse_gpt_oss, "mistral": parse_mistral}


class LocalLLM:
    def __init__(self, preset: str = "qwen", model: str | None = None, thinking: bool = False):
        """`thinking`: turn the model's thinking on (qwen only; off by default, as in the paper)."""
        import torch
        import transformers

        self.name, self.preset, self.thinking = preset, PRESETS[preset], thinking
        if thinking:
            if "enable_thinking" not in self.preset.chat_kwargs:
                raise ValueError(f"thinking can only be switched on for the qwen preset, not {preset}")
            self.preset = replace(self.preset, chat_kwargs={**self.preset.chat_kwargs, "enable_thinking": True})
        path = model or self.preset.model
        self.tok = transformers.AutoTokenizer.from_pretrained(path, padding_side="left")
        try:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto").eval()
        except ValueError:  # vision-language checkpoints (Mistral Small 3.2)
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(
                path, dtype=torch.bfloat16, device_map="auto"
            ).eval()

    def generate(self, messages: list[dict], tools: list[dict] | None = None, max_new_tokens: int = 500) -> str:
        import torch

        prompt = self.tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, **self.preset.chat_kwargs
        )
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        if self.thinking:
            max_new_tokens += THINKING_TOKENS
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id
            )
        text = self.tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=False)
        for stop in (self.tok.eos_token, "<|im_end|>", "<|endoftext|>"):
            if stop and stop in text:
                text = text.split(stop)[0]
        return text.strip()

    def parse(self, raw: str):
        return PARSERS[self.name](raw)

    def assistant_turn(self, text: str, call: dict | None) -> dict:
        if call is None or self.preset.raw_history:
            return {"role": "assistant", "content": text}
        call_id = uuid.uuid4().hex[:9]
        return {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
            ],
        }

    def tool_turn(self, name: str, content: str, assistant: dict) -> dict:
        turn = {"role": "tool", "name": name, "content": content}
        if "tool_calls" in assistant:
            turn["tool_call_id"] = assistant["tool_calls"][0]["id"]
        return turn
