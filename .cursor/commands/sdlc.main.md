## /sdlc.main — основной SDLC (00_Main) + пронумерованные sub‑SDLC

Стандартный layout SDLC в этом репо:

- **`docs/sdlc/00_Main/`** — основной SDLC проекта (базовая структура: status → BA → architecture → specs → QA).
- **`docs/sdlc/01_*`, `docs/sdlc/02_*`, …** — дополнительные sub‑SDLC этого же проекта (например `01_GameIdeas`, `02_MlirDialects`).

Цель: единый "main SDLC" рядом с независимыми подпроцессами, упорядоченными по номеру (как "main" + "ветки", но на уровне папок).

### Input (опционально)

- **path** — корень SDLC. По умолчанию `docs/sdlc`. Все создаваемые папки и реестр — относительно этого пути (например `<path>/00_Main/`, `<path>/00_registry.json`).
- **subProjects** — список под‑проектов для создания сразу (вместо последующих вызовов `/sdlc.create`). Формат: через запятую или по одному на строку. Каждый элемент — короткое имя (например `GameIdeas`, `ReleaseGates`) или уже с номером `NN_Name` (например `01_GameIdeas`). Для каждого: определить номер NN (если не задан — следующий свободный), создать `<path>/NN_<name>/` с README и подпапками по шаблону `default` (как в `/sdlc.create`), добавить запись в `<path>/00_registry.json`. При отсутствии goal_title/goal_description — использовать имя папки как goal_title и нейтральное описание цели.

### Output (шаги, apply)

Все пути ниже — относительно **path** (по умолчанию `docs/sdlc`). Подставлять значение path везде, где указано `<path>`.

1. **Создать основной SDLC**:
   - `<path>/00_Main/README.md` (H1 + блок **Цель** + навигация на sub‑SDLC и реестр).
   - Подпапки по шаблону `default`:
     - `<path>/00_Main/00_Status/`
     - `<path>/00_Main/01_BA/`
     - `<path>/00_Main/02_Architecture/`
     - `<path>/00_Main/03_Specs/`
     - `<path>/00_Main/04_QA/`
     - В каждой подпапке — краткий `README.md` ("что сюда кладём").

2. **Создать перечисленные sub‑SDLC** (если заданы **subProjects**):
   - Для каждого элемента списка: определить целевую папку `<path>/NN_<name>/` (следующий свободный NN, если не задан).
   - Создать `<path>/NN_<name>/README.md` (H1, блок **Цель**), подпапки по шаблону `default` с кратким README в каждой.
   - Добавить запись в `<path>/00_registry.json`: sdlcId (`sdlc.<slug>`), slug, path, goal, goalShort, при необходимости domainTags.

3. **Пронумеровать существующие sub‑SDLC** (если уже есть папки без префикса NN_):
   - Переименовать `<path>/<name>/` → `<path>/NN_<name>/` (минимальный churn, сохранять смысл имени).
   - Пример: `game_ideas` → `01_GameIdeas`, `mlir_dialects` → `02_MlirDialects`.

4. **Навигация и реестр**:
   - Обновить `<path>/README.md`: описать `00_Main` + `NN_*`, убрать устаревшие входные точки.
   - Обновить `<path>/00_registry.json`: запись `sdlc.main` (path `<path>/00_Main/`), обновить path для переименованных sub‑SDLC.
   - Обновить ссылки по репо (`docs/README.md`, `.cursor/INDEX.md`, `.cursor/registry.yaml`, skills/commands и т.п.), если path = `docs/sdlc`.

5. **Совместимость**:
   - Если существовали "legacy" входы (например `<path>/00_Ideas/`) — удалить или заменить ссылками на соответствующий sub‑SDLC (`01_GameIdeas/…`), чтобы не плодить параллельные "main".

### Constraints

- Переезды делать через `git mv`, чтобы сохранялась история и не ломались диффы.
- Коммиты атомарные: отдельно SDLC‑layout/реестр/ссылки и отдельно код, если затрагивается.

### См. также

- `/sdlc.create` — добавить один sub‑SDLC позже по одному.

### Референс

- Корень SDLC: `<path>/README.md` (по умолчанию `docs/sdlc/README.md`)
- Реестр: `<path>/00_registry.json` (по умолчанию `docs/sdlc/00_registry.json`)
- Основной SDLC: `<path>/00_Main/` (по умолчанию `docs/sdlc/00_Main/`)
