# Снимок статуса (read-only)

Сгенерирован по фактическим источникам. Точки дрейфа перечислены в конце.

## SoT + Start here

- **docs/README.md**: навигация по документации; start here для tt-lang — [00_Main/00_Status/00_Quickstart.md](00_Quickstart.md); статус — [01_tt_metal/00_Status/00_projects_graph.md](../../01_tt_metal/00_Status/00_projects_graph.md).
- **docs/sdlc/00_Main/00_Status/README.md**: дашборд; статус и Kanban — в [01_tt_metal/00_Status/](../../01_tt_metal/00_Status/).

## Direction (1–3 буллета + Evidence)

- Явные слои (Program / Graph / Compile / Runtime), Pydantic-типы и прокси на границе с ttnn ([Direction.md](Direction.md), [20_IdealDataFlowAndModuleStructure.md](../02_Architecture/20_IdealDataFlowAndModuleStructure.md)).
- Иерархические request-модели вместо длинных списков параметров ([08_PythonDialectLayersAndDataFlow.md](../02_Architecture/08_PythonDialectLayersAndDataFlow.md)).
- Документация и статус в согласии с кодом (SDLC-first, sdlc.audit).

## Status snapshot

| | Содержимое |
|---|------------|
| **Now** | ttl.nn PyTorch-like API (Module, Sequential, Pipeline) — in progress; рефакторинг ttl по слоям — в Next. |
| **Next** | Рефакторинг ttl по слоям (program/compile/runtime/graph); Python dialect layers + Pydantic kernel_runner; E2E планировщик (Фазы 3–4). |
| **Blockers** | (Не зафиксированы в текущих доках.) |
| **Risks** | Дрейф между .cursor/commands/sdlc.status и фактическими путями к статусу. |

Источники: [00_projects_graph.md](../../01_tt_metal/00_Status/00_projects_graph.md), [01_Kanban.md](../../01_tt_metal/00_Status/01_Kanban.md).

## Top priorities (3)

1. Выровнять команду sdlc.status и sdlc.audit с фактическими путями (см. Точки дрейфа).
2. Рефакторинг ttl_api в тонкий фасад; слои program/compile/runtime с чистыми контрактами.
3. Ввести реестр стадий пайплайна; фичи включаются по пайплайну.

## Where to put new info

- Idea → `docs/sdlc/00_Main/00_Ideas/` (или inbox при наличии).
- Bug → [01_tt_metal/00_Status/02_Problems_Log.md](../../01_tt_metal/00_Status/02_Problems_Log.md).
- Execution → `docs/sdlc/00_Main/06_Development/TASK_*.md`.

## Consistency hint

При расхождении обновить `.cursor/commands/sdlc.status.md` и при необходимости `sdlc.audit` (источники и пути).

---

## Точки дрейфа (для sdlc.audit)

1. **.cursor/commands/sdlc.status.md** — в разделе «Источники» указано:
   - «Статус: `docs/sdlc/00_Main/00_Status/00_projects_graph.md`, `01_Kanban.md`»  
   Факт: файлов `00_projects_graph.md` и `01_Kanban.md` в `docs/sdlc/00_Main/00_Status/` нет; они находятся в `docs/sdlc/01_tt_metal/00_Status/`. Нужно заменить на корректные пути.

2. **.cursor/commands/sdlc.status.md** — указано «Контекст: свежие `docs/06_Development/TASK_*.md`».  
   Факт: в tt-lang путь к TASK — `docs/sdlc/00_Main/06_Development/TASK_*.md`. Нужно уточнить путь.

3. **.cursor/commands/sdlc.status.md** — указано «При необходимости: `docs/03_BA/*`, `docs/01_Architecture/*`».  
   Факт: в tt-lang структура — `docs/sdlc/00_Main/01_BA/`, `docs/sdlc/00_Main/02_Architecture/`. Нужно привести к фактическим путям.

4. **CLAUDE.md / AGENTS.md** — упоминается `docs/BUILD_SYSTEM.md`.  
   Факт: файл находится в `docs/sdlc/00_Main/03_Specs/BUILD_SYSTEM.md`. Добавить корректную ссылку или алиас.
