## /release.aaa — путь к релизу (AAA: docs → gates → код)

Довести выбранный проект до AAA-качества и состояния «можно релизить» через SDLC.

### Input

Опционально: релиз (`REL-*` или путь), подпроект (`apps/<name>/`, `packages/<name>/`), пожелания по UX. **Пусто** → фокус из `docs/sdlc/00_Main/00_Status/00_projects_graph.md`, `01_Kanban.md`; ближайший `REL-*` в `docs/**`.

### AAA-критерии

Time-to-first-success минимален; 2 клика / 1 подтверждение; понятные affordances; Undo/confirm, без silent overwrite; gates пройдены, зафиксированы.

### Порядок

1. SoT + цель релиза: `docs/README.md`, `docs/sdlc/00_Main/00_Status/00_projects_graph.md`, `01_Kanban.md`, `REL-*`.
2. SDLC-доки сначала: P0/P1 gates, задачи в статусе «ready», минимальные правки при расхождениях.
3. Release path: 3–10 P0 шагов со ссылками на TASK и чеклисты.
4. Имплементация по шагам; run configs при необходимости.
5. Прогон P0 чеклистов, Notes в релиз-доке.

### Output

SoT + Release target; AAA Direction; Release path (P0); Now/Next/Blockers/Risks; What I changed.
