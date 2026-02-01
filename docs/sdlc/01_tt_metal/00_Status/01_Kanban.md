# Kanban (tt-metal)

Снимок задач и приоритетов. Связан с [00_projects_graph.md](00_projects_graph.md).

| Колонка | Описание |
|---------|----------|
| Backlog | Идеи, захваченные в [00_Ideas/](../00_Ideas/) |
| Next | Запланированные таски (03_Specs, при необходимости TASK_*) |
| In progress | Текущие работы |
| Done | Завершённые |

Обновлять по мере движения задач.

---

## Примеры задач (scheduler-viz, док 16/17)

| Backlog | Next | In progress | Done |
|---------|------|-------------|------|
| 17 flexible scheduler framework UX (идея зафиксирована в [00_Main/00_Ideas/17_flexible_scheduler_framework_ux.md](../../00_Main/00_Ideas/17_flexible_scheduler_framework_ux.md)) | Фазы 3–4 roadmap: E2E планировщик (граф операций, заглушка RCW, материализация) | — | Док 16: идея UI + Gym + динамика; док 17: расширенное видение UX |
| scheduler-viz: фаза динамики (симулятор → метрики → reward, отображение в UI) | scheduler-viz: реестр алгоритмов, сравнение планов | — | scheduler-viz MVP (placement env, 3 вида UI, step/run) |
| — | Python dialect layers: док 08_PythonDialectLayersAndDataFlow; Pydantic kernel_runner (KernelDescriptorBuildRequest, CBDescriptorBuildRequest) | — | Док 08: Python dialect layers and data flow |
| — | Рефакторинг ttl по слоям (program/compile/runtime/graph) | ttl.nn PyTorch-like API: Module, Sequential, Pipeline, примеры, тесты, run configs, baseline ([TASK_pytorch_like_modules_pitch.md](../../00_Main/06_Development/TASK_pytorch_like_modules_pitch.md)) | — |
