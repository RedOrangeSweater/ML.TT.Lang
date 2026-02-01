# TASK: PyTorch-like модульный UX (питч рефакторинга)

- **Статус**: In progress.
- **Контекст**: [Direction.md](../00_Status/Direction.md), [08_PythonDialectLayersAndDataFlow.md](../02_Architecture/08_PythonDialectLayersAndDataFlow.md), [20_IdealDataFlowAndModuleStructure.md](../02_Architecture/20_IdealDataFlowAndModuleStructure.md).

## Цель

Сделать код tt-lang максимально удобным в духе PyTorch/PyTorch Lightning: модульно, понятно, расширяемо. Подготовить воспроизводимый «питч» рефакторинга: все тесты проходят, backward compatible по публичному API, без просадки по performance, с инструкцией запуска и baseline по commit hash main.

## Definition of Done

- [x] Новый модульный API (`ttl.nn`): Module, Sequential, Pipeline; Pydantic-конфиги на границах.
- [x] Реализация опирается на существующий engine (`ttl.run`, compile, runtime); внутренние импорты можно менять.
- [x] 1–2 эталонных примера в `examples/` (композиция, минимальный boilerplate).
- [x] Тесты: lit (IR) и pytest/unit для нового API.
- [x] Run configs в `.vscode/` для запуска тестов и примеров одним кликом.
- [x] Perf smoke + baseline (commit hash main), правило «не хуже» baseline.
- [ ] Все текущие проверки проходят: `check-ttlang-mlir`, `check-ttlang-python-lit`, выбранный поднабор pytest (проверить после сборки).
- [x] Статусные документы обновлены: [01_Kanban.md](../../01_tt_metal/00_Status/01_Kanban.md), [00_projects_graph.md](../../01_tt_metal/00_Status/00_projects_graph.md).

## Публичный API (зафиксирован)

- **Модуль**: `ttl.nn` (по аналогии с `torch.nn`). Путь в коде: `python/ttl/nn/`.
- **Экспорт из пакета** (`python/ttl/__init__.py`): добавляем в `__all__` и реэкспорт:
  - `Module` — базовый интерфейс (PyTorch-style).
  - `Sequential` — композиция модулей по порядку (по умолчанию).
  - `Pipeline` — синоним/обёртка для цепочки программ с единым run.
  - Опционально: `PipelineConfig` / `RunConfig` (Pydantic) для опций компиляции и запуска.
- **Использование**: `import ttl; ttl.nn.Sequential(...)` или `from ttl import nn; nn.Sequential(...)`. В `__init__.py` реэкспортируем `nn` как подмодуль и при необходимости `Module`, `Sequential`, `Pipeline` на верхний уровень для краткости.

## Ссылки

- Направление: [docs/sdlc/00_Main/00_Status/Direction.md](../00_Status/Direction.md).
- LLD слоёв: [08_PythonDialectLayersAndDataFlow.md](../02_Architecture/08_PythonDialectLayersAndDataFlow.md).
- Целевая структура и сравнение с PyTorch: [20_IdealDataFlowAndModuleStructure.md](../02_Architecture/20_IdealDataFlowAndModuleStructure.md).
