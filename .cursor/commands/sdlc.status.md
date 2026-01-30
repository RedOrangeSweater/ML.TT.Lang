## /sdlc.status — снимок направления и статуса (read-only)

Ответы: куда движемся, Now/Next/Blockers/Risks, приоритеты, куда записать идею/баг. Русский, структура. **Read-only**: не создавать/не редактировать файлы без явного запроса.

### Input

Опционально: «в чём запутался», гипотеза направления, путь к `TASK_*`/`REQ`/`ADR` или список файлов. **Пусто** → снимок по документам.

### Источники

1. SoT + start here: `docs/README.md`.
2. Статус: `docs/sdlc/00_Main/00_Status/00_projects_graph.md`, `01_Kanban.md`.
3. Контекст: свежие `docs/06_Development/TASK_*.md`.
4. При необходимости: `docs/03_BA/*`, `docs/01_Architecture/*`.

Если статус пустой/расходится — явно сказать и предложить next actions.

### Output

- **SoT + Start here**; **Direction** (1–3 буллета + Evidence).
- **Status snapshot**: Now, Next, Blockers, Risks.
- **Top priorities**: 3 пункта.
- **Where to put new info**: Idea → `docs/sdlc/01_GameIdeas/inbox/` (/ttl-idea); Bug → `02_Problems_Log.md`; Execution → `TASK_*`.
- **Consistency hint**: при расхождении — обновить `docs/sdlc/00_Main/00_Status/*` и/или `/sdlc.audit`.

### Constraints

Без широких рефакторов. Не угадывать детали; недостающее — помечать как предположения и перечислить, что создать/обновить.
