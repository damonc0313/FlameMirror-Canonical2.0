# Core Component Base

This repository now includes a shared `BaseComponent` class in `src/core/base.py`. It provides:

- Standard initialization and cleanup behavior
- A `health_check` method reporting component status and timestamp
- Shared configuration via `ComponentConfig`

`GitManager` and `CodeGenerator` leverage this base to reduce duplication and improve maintainability.
