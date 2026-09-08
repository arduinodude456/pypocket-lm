"""Compact Python IR used by the compositional experiment.

IR is deliberately small and deterministic. The model predicts IR tokens; this
module validates and compiles them. It never executes generated code.
"""
from __future__ import annotations

from dataclasses import dataclass

OPS = {"SET", "ADD", "SUB", "MUL", "DIV", "PRINT", "RETURN", "FOR_RANGE", "END_FOR", "FUNCTION", "END_FUNCTION"}


def tokenize_ir(text: str) -> list[str]:
    tokens = ["<BOS>"]
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        tokens.extend(parts)
        tokens.append("<NL>")
    tokens.append("<END>")
    return tokens


def validate_ir(tokens: list[str]) -> tuple[bool, str]:
    depth = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"<BOS>", "<NL>", "<END>"}:
            index += 1
            continue
        if token == "FOR_RANGE":
            if index + 2 >= len(tokens) or not tokens[index + 1].startswith("VAR:") or not tokens[index + 2].startswith("NUM:"):
                return False, "FOR_RANGE requires VAR and NUM"
            depth += 1
            index += 3
            continue
        if token == "END_FOR":
            if depth == 0:
                return False, "unexpected END_FOR"
            depth -= 1
            index += 1
            continue
        if token in {"SET", "ADD", "SUB", "MUL", "DIV"}:
            if index + 2 >= len(tokens) or not tokens[index + 1].startswith("VAR:"):
                return False, f"{token} requires a variable and value"
            index += 3
            continue
        if token == "PRINT":
            if index + 1 >= len(tokens) or not (tokens[index + 1].startswith(("VAR:", "STR:", "NUM:"))):
                return False, "PRINT requires a value"
            index += 2
            continue
        if token == "RETURN":
            if index + 1 >= len(tokens) or not tokens[index + 1].startswith(("VAR:", "NUM:")):
                return False, "RETURN requires a value"
            index += 2
            continue
        if token == "FUNCTION":
            if index + 1 >= len(tokens) or not tokens[index + 1].startswith("NAME:"):
                return False, "FUNCTION requires a name"
            index += 2
            continue
        if token == "END_FUNCTION":
            index += 1
            continue
        return False, f"unknown IR token: {token}"
    if depth:
        return False, "unclosed loop"
    return True, "ok"


def _value(token: str) -> str:
    kind, value = token.split(":", 1)
    if kind == "VAR":
        return value
    if kind == "NUM":
        return value
    if kind == "STR":
        return repr(value.replace("_", " "))
    return "value"


def compile_ir(tokens: list[str]) -> str:
    valid, reason = validate_ir(tokens)
    if not valid:
        raise ValueError(reason)
    lines: list[str] = []
    indent = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"<BOS>", "<NL>", "<END>"}:
            index += 1
            continue
        prefix = "    " * indent
        if token == "SET":
            lines.append(f"{prefix}{tokens[index + 1].split(':', 1)[1]} = {_value(tokens[index + 2])}"); index += 3
        elif token in {"ADD", "SUB", "MUL", "DIV"}:
            symbol = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}[token]
            var = tokens[index + 1].split(':', 1)[1]; value = _value(tokens[index + 2])
            lines.append(f"{prefix}{var} = {var} {symbol} {value}"); index += 3
        elif token == "PRINT":
            lines.append(f"{prefix}print({_value(tokens[index + 1])})"); index += 2
        elif token == "FOR_RANGE":
            var = tokens[index + 1].split(':', 1)[1]; limit = _value(tokens[index + 2])
            lines.append(f"{prefix}for {var} in range({limit}):"); indent += 1; index += 3
        elif token == "END_FOR":
            indent = max(0, indent - 1); index += 1
        elif token == "FUNCTION":
            lines.append(f"{prefix}def {tokens[index + 1].split(':', 1)[1]}():"); indent += 1; index += 2
        elif token == "END_FUNCTION":
            indent = max(0, indent - 1); index += 1
        elif token == "RETURN":
            lines.append(f"{prefix}return {_value(tokens[index + 1])}"); index += 2
        else:
            index += 1
    return "\n".join(lines) + ("\n" if lines else "")
