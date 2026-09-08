from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from data.ir_tasks import build_tasks
from ir import compile_ir, tokenize_ir, validate_ir
from ir_model import IRLanguageModel


def evaluate(model: IRLanguageModel, train: list[dict], validation: list[dict]) -> dict:
    def score(tasks: list[dict]) -> dict:
        exact = valid = runnable = 0
        for task in tasks:
            generated = model.generate_ir(task["prompt"], temperature=0)
            ok, _ = validate_ir(generated)
            valid += int(ok)
            if " ".join(generated) == " ".join(tokenize_ir(task["ir"])[1:]):
                exact += 1
            if ok:
                try:
                    ast.parse(compile_ir(generated))
                    runnable += 1
                except (SyntaxError, ValueError):
                    pass
        total = max(1, len(tasks))
        return {"exact_match": exact / total, "valid_ir": valid / total, "syntactically_valid_python": runnable / total}

    return {
        "training_cross_entropy": model.cross_entropy(train),
        "validation_cross_entropy": model.cross_entropy(validation),
        "train": score(train),
        "novel_combinations": score(validation),
        "validation_tasks": [task["prompt"] for task in validation],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="web/ir_model.json")
    parser.add_argument("--metrics", default="web/metrics.json")
    args = parser.parse_args()
    all_tasks = build_tasks(seed=19, count=150)
    train, validation = all_tasks[:120], all_tasks[120:]
    model = IRLanguageModel(order=20).fit(train, epochs=3)
    metrics = evaluate(model, train, validation)
    model.save(args.output)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
