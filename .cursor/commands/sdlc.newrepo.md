## /sdlc.newrepo — пустой проект с полной структурой папок и скриптов

Один сценарий: скопировать Cursor-артефакты из template в целевой репо, затем создать полную SDLC-структуру одной командой.

Цель: из «голого» репо за два шага получить `docs/sdlc/00_Main/` (00_Status, 01_BA, 02_Architecture, 03_Specs, 04_QA), README, реестр и при необходимости перечисленные sub‑SDLC.

### Input (опционально)

- **target** — путь к корню целевого репо (куда копировать). По умолчанию: текущий репо (где вызывается команда).
- **path** — корень SDLC для последующего `/sdlc.main` (по умолчанию `docs/sdlc`).
- **subProjects** — список под‑проектов для создания сразу (формат как в `/sdlc.main`: через запятую, например `01_Ideas,02_ReleaseGates`).

### Output (шаги, apply)

1. **Подготовить целевой репо**  
   Скопировать в него Cursor-артефакты из template (в репо cursorrules):
   - `docs/projects/sdlc-lifecycle/.cursor/rules/` → `<target>/.cursor/rules/`
   - `docs/projects/sdlc-lifecycle/.cursor/commands/` → `<target>/.cursor/commands/`  
   Если template недоступен (целевой репо не cursorrules) — взять из переданного архива/папки `sdlc-lifecycle` или указать пользователю скопировать вручную по `docs/projects/sdlc-lifecycle/README.md`.

2. **В целевом репо выполнить одну команду** `/sdlc.main`:
   - Без параметров — создаётся структура по умолчанию: `docs/sdlc/00_Main/` (00_Status, 01_BA, 02_Architecture, 03_Specs, 04_QA) с README в каждой папке, `docs/sdlc/README.md`, `docs/sdlc/00_registry.json`.
   - С параметрами: **path** (корень SDLC, по умолчанию `docs/sdlc`), **subProjects** (список под‑проектов для создания сразу).

### Next step

После выполнения — вести документацию по SDLC-циклу; при необходимости добавлять sub‑SDLC через `/sdlc.create`.

### Constraints

- Не перезаписывать без подтверждения существующие `.cursor/rules` или `.cursor/commands`, если там уже есть содержимое.
- Копирование и создание структуры — атомарные шаги; коммиты по правилам репо (02_atomic_git).

### См. также

- `/sdlc.main` — создание/обновление основного SDLC и перечисленных sub‑SDLC (path, subProjects).
- Template: `docs/projects/sdlc-lifecycle/README.md` в репо cursorrules.
