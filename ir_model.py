from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from ir import tokenize_ir
from tiny_python_lm import normal_tokenize


def concept_tokens(prompt: str) -> list[str]:
    """Map words to compositional symbols; this is tokenization, not output logic."""
    words = normal_tokenize(prompt)
    tokens: list[str] = []
    for word in words:
        if word in {"setze", "speichere", "set", "save"}: tokens.append("CONCEPT:SET")
        elif word in {"addiert", "addiere", "add", "plus"}: tokens.append("CONCEPT:ADD")
        elif word in {"subtrahiert", "subtrahiere", "sub", "minus"}: tokens.append("CONCEPT:SUB")
        elif word in {"multipliziert", "multipliziere", "multiply", "mul", "times"}: tokens.append("CONCEPT:MUL")
        elif word in {"teilt", "teile", "divide", "div", "geteilt"}: tokens.append("CONCEPT:DIV")
        elif word in {"schleife", "loop", "for"}: tokens.append("CONCEPT:FOR_RANGE")
        elif word in {"aus", "gib", "gibt", "print", "output", "ausgabe"}: tokens.append("CONCEPT:PRINT")
        elif word.isdigit(): tokens.append(f"NUM:{word}")
        elif word in {"x", "total", "wert", "i", "number", "item"}: tokens.append(f"VAR:{word}")
    return tokens


class IRLanguageModel:
    """Tiny autoregressive baseline over prompt+IR token sequences.

    It uses add-one cross-entropy counts rather than pretending to be a neural
    network. This makes the generalization limits measurable and reproducible.
    """
    def __init__(self, order: int = 20):
        self.order = order
        self.transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def sequence(self, prompt: str, ir: str) -> list[str]:
        return ["<BOS>", "<PROMPT>", *concept_tokens(prompt), "<IR>", *tokenize_ir(ir)[1:]]

    def fit(self, tasks: list[dict[str, str]], epochs: int = 2) -> "IRLanguageModel":
        for _ in range(max(1, epochs)):
            for task in tasks:
                sequence = self.sequence(task["prompt"], task["ir"])
                self.vocab.update(sequence)
                for index in range(1, len(sequence)):
                    for width in range(1, min(self.order - 1, index) + 1):
                        context = "\u241f".join(sequence[index - width:index])
                        self.transitions[context][sequence[index]] += 1
        return self

    def _next(self, history: list[str], temperature: float = 0.7, top_k: int = 8) -> str:
        candidates = Counter()
        for width in range(min(self.order - 1, len(history)), 0, -1):
            context = "\u241f".join(history[-width:])
            candidates = self.transitions.get(context, Counter())
            if candidates:
                break
        if not candidates:
            return "<END>"
        ranked = candidates.most_common(top_k)
        if temperature <= 0:
            return ranked[0][0]
        weights = [count ** (1 / max(temperature, 0.05)) for _, count in ranked]
        return random.choices([token for token, _ in ranked], weights)[0]

    def generate_ir(self, prompt: str, max_tokens: int = 48, temperature: float = 0.35, top_k: int = 8) -> list[str]:
        history = ["<BOS>", "<PROMPT>", *concept_tokens(prompt), "<IR>"]
        output: list[str] = []
        for _ in range(max_tokens):
            token = self._next(history, temperature, top_k)
            output.append(token)
            history.append(token)
            if token == "<END>":
                break
        return output

    def cross_entropy(self, tasks: list[dict[str, str]]) -> float:
        loss = 0.0
        total = 0
        for task in tasks:
            sequence = self.sequence(task["prompt"], task["ir"])
            for index in range(1, len(sequence)):
                context = "\u241f".join(sequence[max(0, index - self.order + 1):index])
                counts = self.transitions.get(context, Counter())
                probability = (counts[sequence[index]] + 1) / (sum(counts.values()) + len(self.vocab))
                loss -= math.log(probability)
                total += 1
        return loss / max(1, total)

    def save(self, path: str | Path) -> None:
        payload = {"format": "pypocket-ir-ngram-v1", "order": self.order, "vocab": sorted(self.vocab), "transitions": {key: dict(value) for key, value in self.transitions.items()}}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
