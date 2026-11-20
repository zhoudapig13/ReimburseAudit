# Project Architecture

The project separates model policy from training orchestration and command-line data adapters.

```clean-architecture
layer domain      = audit_engine/models/**
layer policy      = audit_engine/policy/**
layer adapters    = audit_engine/readers/**
layer application = audit_engine/training/**
layer main        = code/scripts/**
```

- `audit_engine/models` contains only tensor-level model architecture and imports no data loaders, paths, or command-line code.
- `audit_engine/training` owns losses and stage policies and may depend on the model layer.
- `audit_engine/policy` owns typed deterministic comparison operations and imports no labels, model readers, or filesystem code.
- `audit_engine/readers` owns local Qwen loading, source text views, and frozen feature extraction; it never reads supervision or evaluation answers.
- `code/scripts` owns filesystem access, JSONL conversion, reports, manifests, and command-line wiring.
- `data_raw` is read-only. Generated artifacts are written below `data`.
- `labels_private` is available only to training-label and evaluation processes, never production inference.
