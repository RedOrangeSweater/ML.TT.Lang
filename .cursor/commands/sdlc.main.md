## /sdlc.main — основной SDLC (00_Main) + пронумерованные sub‑SDLC

Стандартный layout SDLC в этом репо:

- **`docs/sdlc/00_Main/`** — основной SDLC проекта (базовая структура: status → BA → architecture → specs → QA).
- **`docs/sdlc/01_*`, `docs/sdlc/02_*`, …** — дополнительные sub‑SDLC этого же проекта (например `01_GameIdeas`, `02_MlirDialects`).

Цель: единый “main SDLC” рядом с независимыми подпроцессами, упорядоченными по номеру (как “main” + “ветки”, но на уровне папок).

### Output (шаги, apply)

1. **Создать основной SDLC**:
   - `docs/sdlc/00_Main/README.md` (H1 + блок **Цель** + навигация на sub‑SDLC и реестр).
   - Подпапки по шаблону `default`:
     - `docs/sdlc/00_Main/00_Status/`
     - `docs/sdlc/00_Main/01_BA/`
     - `docs/sdlc/00_Main/02_Architecture/`
     - `docs/sdlc/00_Main/03_Specs/`
     - `docs/sdlc/00_Main/04_QA/`
     - В каждой подпапке — краткий `README.md` (“что сюда кладём”).

2. **Пронумеровать существующие sub‑SDLC** (если уже есть):
   - Переименовать `docs/sdlc/<name>/` → `docs/sdlc/NN_<name>/` (минимальный churn, сохранять смысл имени).
   - Пример: `game_ideas` → `01_GameIdeas`, `mlir_dialects` → `02_MlirDialects`.

3. **Навигация и реестр**:
   - Обновить `docs/sdlc/README.md`: описать `00_Main` + `NN_*`, убрать устаревшие входные точки.
   - Обновить `docs/sdlc/00_registry.json`:
     - добавить запись `sdlc.main` (path `docs/sdlc/00_Main/`)
     - обновить `path` для переименованных sub‑SDLC
   - Обновить ссылки по репо (`docs/README.md`, `.cursor/INDEX.md`, `.cursor/registry.yaml`, skills/commands и т.п.).

4. **Совместимость**:
   - Если существовали “legacy” входы (например `docs/sdlc/00_Ideas/`) — удалить или заменить ссылками на соответствующий sub‑SDLC (`01_GameIdeas/…`), чтобы не плодить параллельные “main”.

### Constraints

- Переезды делать через `git mv`, чтобы сохранялась история и не ломались диффы.
- Коммиты атомарные: отдельно SDLC‑layout/реестр/ссылки и отдельно код, если затрагивается.

### Референс

- Корень SDLC: `docs/sdlc/README.md`
- Реестр: `docs/sdlc/00_registry.json`
- Основной SDLC: `docs/sdlc/00_Main/`
