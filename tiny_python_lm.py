"""A tiny, transparent Python-specialized language model.

This module intentionally uses only the Python standard library. It is a
research prototype, not a replacement for a neural coding model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import tokenize as py_tokenize
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Iterable

SPECIAL = {
    py_tokenize.INDENT: "<INDENT>",
    py_tokenize.DEDENT: "<DEDENT>",
    py_tokenize.NEWLINE: "<NEWLINE>",
}
KEYWORDS = {
    "and", "as", "assert", "async", "await", "break", "case", "class",
    "continue", "def", "del", "elif", "else", "except", "finally", "for",
    "from", "global", "if", "import", "in", "is", "lambda", "match", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
    "True", "False", "None",
}
BUILTINS = {"print", "len", "range", "int", "str", "float", "list", "dict", "set", "sum", "min", "max"}


def tokenize_python(source: str) -> list[str]:
    """Convert Python into compact structural and lexical tokens."""
    result: list[str] = ["<BOS>"]
    try:
        stream = py_tokenize.generate_tokens(StringIO(source).readline)
        for token in stream:
            token_type, text = token.type, token.string
            if token_type == py_tokenize.ERRORTOKEN and text.isspace():
                continue
            if token_type == py_tokenize.ENDMARKER:
                continue
            if token_type in SPECIAL:
                result.append(SPECIAL[token_type])
            elif token_type in (py_tokenize.ENCODING, py_tokenize.NL, py_tokenize.COMMENT):
                continue
            elif token_type == py_tokenize.NAME:
                if text in KEYWORDS:
                    result.append(f"KW:{text}")
                elif text in BUILTINS:
                    result.append(f"FN:{text}")
                else:
                    result.append("NAME:<id>")
            elif token_type == py_tokenize.STRING:
                result.append("LIT:<str>")
            elif token_type == py_tokenize.NUMBER:
                result.append("LIT:<num>")
            else:
                result.append(f"OP:{text}")
    except (IndentationError, py_tokenize.TokenError):
        result.extend(["<INVALID>", "<END>"])
    if source.endswith("\n") or (source and source[-1].isspace()):
        trailing_indent = len(source.rsplit("\n", 1)[-1]) // 4
        while trailing_indent > 0:
            result.append("<INDENT>")
            trailing_indent -= 1
    return result


def detokenize(tokens: Iterable[str]) -> str:
    """Render a conservative readable Python-like representation."""
    lines: list[str] = []
    current: list[str] = []
    indent = 0
    for token in tokens:
        if token in {"<BOS>", "<END>"}:
            continue
        if token == "<INDENT>":
            indent += 1
            continue
        if token == "<DEDENT>":
            if current:
                lines.append("    " * indent + " ".join(current).strip())
                current = []
            indent = max(0, indent - 1)
            continue
        if token == "<NEWLINE>":
            if current:
                lines.append("    " * indent + " ".join(current).strip())
                current = []
            continue
        if token.startswith("KW:"):
            current.append(token[3:])
        elif token.startswith("FN:"):
            current.append(token[3:])
        elif token.startswith("OP:"):
            op = token[3:]
            if op in {",", ":", ")", "]", "}"} and current:
                current[-1] += op
            elif op in {"(", "[", "{"}:
                current.append(op)
            else:
                current.append(op)
        elif token == "NAME:<id>":
            current.append("value")
        elif token == "LIT:<str>":
            current.append('"text"')
        elif token == "LIT:<num>":
            current.append("0")
        else:
            current.append(token)
    if current:
        lines.append("    " * indent + " ".join(current).strip())
    return "\n".join(lines)


class TinyPythonLM:
    def __init__(self, order: int = 3) -> None:
        if order < 2:
            raise ValueError("order must be at least 2")
        self.order = order
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, examples: Iterable[str], epochs: int = 1) -> "TinyPythonLM":
        for _ in range(max(1, epochs)):
            for source in examples:
                tokens = tokenize_python(source) + ["<END>"]
                self.vocab.update(tokens)
                for index in range(len(tokens)):
                    start = max(0, index - self.order + 1)
                    context = "\u241f".join(tokens[start:index]) or "<BOS>"
                    self.counts[context][tokens[index]] += 1
        return self

    def next_token(self, history: list[str], temperature: float = 0.0) -> str:
        context_tokens = history[-(self.order - 1):]
        context = "\u241f".join(context_tokens) or "<BOS>"
        candidates = self.counts.get(context)
        if not candidates:
            candidates = self.counts.get(context_tokens[-1], Counter()) if context_tokens else Counter()
        if not candidates:
            return "<END>"
        if temperature <= 0:
            return candidates.most_common(1)[0][0]
        weighted = [(token, count ** (1 / max(temperature, 0.05))) for token, count in candidates.items()]
        return random.choices([token for token, _ in weighted], [weight for _, weight in weighted])[0]

    def generate(self, prompt: str, max_tokens: int = 48, temperature: float = 0.0) -> list[str]:
        history = tokenize_python(prompt)
        generated: list[str] = []
        for _ in range(max_tokens):
            token = self.next_token(history, temperature)
            generated.append(token)
            history.append(token)
            if token == "<END>":
                break
        return generated

    def to_dict(self) -> dict:
        transitions = {context: dict(counter) for context, counter in self.counts.items()}
        return {"format": "pypocket-ngram-v1", "order": self.order, "vocab": sorted(self.vocab), "transitions": transitions}

    @classmethod
    def from_dict(cls, payload: dict) -> "TinyPythonLM":
        model = cls(order=int(payload["order"]))
        model.vocab = set(payload.get("vocab", []))
        model.counts = defaultdict(Counter, {context: Counter(values) for context, values in payload["transitions"].items()})
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TinyPythonLM":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def perplexity(model: TinyPythonLM, examples: Iterable[str]) -> float:
    total = 0
    negative_log_likelihood = 0.0
    for source in examples:
        history = tokenize_python(source)
        for expected in tokenize_python(source)[1:] + ["<END>"]:
            candidates = model.counts.get("\u241f".join(history[-(model.order - 1):]), Counter())
            probability = (candidates[expected] + 1) / (sum(candidates.values()) + max(1, len(model.vocab)))
            negative_log_likelihood -= math.log(probability)
            total += 1
            history.append(expected)
    return math.exp(negative_log_likelihood / max(1, total))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/examples.py")
    parser.add_argument("--output", default="web/model.json")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    namespace: dict = {}
    exec(Path(args.data).read_text(encoding="utf-8"), namespace)
    examples = namespace["TRAINING_EXAMPLES"]
    model = TinyPythonLM().fit(examples, epochs=args.epochs)
    model.save(args.output)
    print(f"trained examples={len(examples)} vocab={len(model.vocab)} output={args.output}")


if __name__ == "__main__":
    main()
