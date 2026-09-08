# Roadmap

## Phase 1 — current

Transparent n-gram baseline with Python-aware structural tokens, JSON export, browser inference, and GitHub Actions.

## Phase 2 — compact neural model

Replace transition counts with a small character- or token-level GRU/Transformer trained with a pure-Python or WebGPU-compatible implementation. Export quantized weights and keep inference in the browser.

## Phase 3 — user adaptation

Add an opt-in dataset editor that stores examples locally, validates Python syntax, and lets the user export a training bundle. Never upload personal code by default.

## Phase 4 — safe evaluation

Add held-out Python tasks, syntax validity rate, exact-match rate, and regression reports to each GitHub Actions run. Generated code must remain inert text and never execute automatically.
