"""Synthetic compositional tasks; test splits use unseen combinations."""
from __future__ import annotations

import random

OPS = [("ADD", "addiert"), ("SUB", "subtrahiert"), ("MUL", "multipliziert"), ("DIV", "teilt")]


def build_tasks(seed: int = 7, count: int = 120) -> list[dict[str, str]]:
    rng = random.Random(seed)
    tasks: list[dict[str, str]] = []
    for _ in range(count):
        variable = rng.choice(["x", "total", "wert"])
        value = rng.choice([2, 3, 5, 7, 11])
        operation, german = rng.choice(OPS)
        second = rng.choice([2, 4, 6, 8])
        prompt = f"setze {value} in {variable}, {german} {variable} mit {second}, und gib das Ergebnis aus"
        ir = f"SET VAR:{variable} NUM:{value}\n{operation} VAR:{variable} NUM:{second}\nPRINT VAR:{variable}"
        tasks.append({"prompt": prompt, "ir": ir, "family": "arithmetic"})
    for _ in range(max(8, count // 5)):
        variable = rng.choice(["i", "number", "item"])
        limit = rng.choice([3, 5, 10])
        prompt = f"gib die Zahlen von eins bis {limit} aus mit einer Schleife"
        ir = f"FOR_RANGE VAR:{variable} NUM:{limit}\nPRINT VAR:{variable}\nEND_FOR"
        tasks.append({"prompt": prompt, "ir": ir, "family": "loop"})
    return tasks


TRAIN_TASKS = build_tasks()
