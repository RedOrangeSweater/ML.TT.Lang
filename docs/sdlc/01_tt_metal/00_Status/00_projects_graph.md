# Граф проектов (tt-metal)

Точка входа для статуса и навигации по SDLC.

- **SDLC tt-metal**: [README](../README.md) — status, ideas, BA, architecture, specs, QA, operations.
- **Статус-дашборд**: [01_Kanban.md](01_Kanban.md).
- **Пайплайн Doc–MLIR–GraphDB**: [00_Doc_MLIR_GraphDB_Pipeline.md](../02_Architecture/00_Doc_MLIR_GraphDB_Pipeline.md).

**Связанные блоки SDLC**: tt-lang main: [00_Main/](../../00_Main/). LLK: [02_llk/](../../02_llk/).

- **Scheduler-viz** (док 16, 17): UI визуализации планировщика + Gym env; следующий шаг — реестр алгоритмов и Run configs в .vscode.
- **Next**: документ «Python dialect layers and data flow» ([08_PythonDialectLayersAndDataFlow.md](../../00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md)) — выполнен; документ «Идеальный data flow и модульная структура» ([20_IdealDataFlowAndModuleStructure.md](../../00_Main/02_Architecture/20_IdealDataFlowAndModuleStructure.md)) — добавлен; далее — рефакторинг ttl по слоям (program/compile/runtime/graph), расширение Pydantic на kernel_runner, затем E2E планировщик (Фазы 3–4 roadmap).

Обновлять при изменении структуры репо или приоритетов.
