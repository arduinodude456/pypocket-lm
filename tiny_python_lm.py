"""PyPocket LM: a tiny Python-specialized model with a prompt-to-code layer.

The baseline remains transparent and standard-library-only. It combines:
1. a Python-aware token n-gram model for code-prefix continuation, and
2. a tiny trained intent/template layer for natural-language prompts.

This is still a small research model, not a general neural LLM.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
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
WORD_RE = re.compile(r"[\wäöüß]+", re.IGNORECASE)
STOPWORDS = {"a", "an", "the", "in", "on", "of", "to", "and", "ein", "eine", "einen", "der", "die", "das", "für", "mit", "und", "zu"}


def normal_tokenize(text: str) -> list[str]:
    """Tokenize natural-language prompts into reusable lowercase word units."""
    return [word.casefold() for word in WORD_RE.findall(text)]


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
        if token.startswith("KW:") or token.startswith("FN:"):
            current.append(token.split(":", 1)[1])
        elif token.startswith("OP:"):
            op = token[3:]
            if op in {",", ":", ")", "]", "}"} and current:
                current[-1] += op
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
        self.prompt_vocab: set[str] = set()
        self.intent_patterns: dict[str, list[str]] = {}
        self.templates: dict[str, list[str]] = defaultdict(list)

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

    def fit_prompts(self, prompt_examples: Iterable[dict]) -> "TinyPythonLM":
        """Learn a compact natural-language lexicon and intent keyword map."""
        pattern_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for example in prompt_examples:
            intent = str(example["intent"])
            words = normal_tokenize(str(example["prompt"]))
            self.prompt_vocab.update(words)
            for word in set(words):
                pattern_counts[intent][word] += 1
            template = str(example["template"])
            if template not in self.templates[intent]:
                self.templates[intent].append(template)
        self.intent_patterns = {intent: [word for word, _ in counts.most_common()] for intent, counts in pattern_counts.items()}
        return self

    def recognize_intent(self, prompt: str) -> str | None:
        words = set(normal_tokenize(prompt)) - STOPWORDS
        best_intent, best_score = None, 0.0
        for intent, pattern in self.intent_patterns.items():
            matches = words.intersection(pattern)
            if not matches:
                continue
            score = sum(1.0 / (1.0 + pattern.index(word)) for word in matches)
            if score > best_score:
                best_intent, best_score = intent, score
        return best_intent

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

    def generate_for_prompt(self, prompt: str, max_tokens: int = 48) -> dict[str, str | None]:
        """Generate code from natural language when an intent is recognized."""
        intent = self.recognize_intent(prompt)
        if intent and intent in self.templates:
            candidates = self.templates[intent]
            words = set(normal_tokenize(prompt))
            german = words.intersection({"hallo", "welt", "schreibe", "beispiel", "begrüße", "begrüßen", "summe", "schleife"})
            preferred = [candidate for candidate in candidates if ("Hallo" in candidate) == bool(german)]
            return {"intent": intent, "code": (preferred or candidates)[0]}
        return {"intent": None, "code": detokenize(self.generate(prompt, max_tokens=max_tokens))}

    def to_dict(self) -> dict:
        transitions = {context: dict(counter) for context, counter in self.counts.items()}
        return {
            "format": "pypocket-ngram-v2",
            "order": self.order,
            "vocab": sorted(self.vocab),
            "prompt_vocab": sorted(self.prompt_vocab),
            "intent_patterns": self.intent_patterns,
            "templates": self.templates,
            "transitions": transitions,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TinyPythonLM":
        model = cls(order=int(payload["order"]))
        model.vocab = set(payload.get("vocab", []))
        model.prompt_vocab = set(payload.get("prompt_vocab", []))
        model.intent_patterns = payload.get("intent_patterns", {})
        model.templates = defaultdict(list, payload.get("templates", {}))
        model.counts = defaultdict(Counter, {context: Counter(values) for context, values in payload["transitions"].items()})
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

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
    model.fit_prompts(namespace.get("PROMPT_EXAMPLES", []))
    model.save(args.output)
    print(f"trained code_examples={len(examples)} intents={len(model.intent_patterns)} vocab={len(model.vocab)} prompt_vocab={len(model.prompt_vocab)} output={args.output}")


if __name__ == "__main__":
    main()
