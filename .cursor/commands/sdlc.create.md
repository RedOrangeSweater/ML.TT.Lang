## /sdlc.create — создать новый SDLC-подпроект

Быстро завести новый **sub‑SDLC** (отдельный процесс) с консистентной структурой и регистрацией в реестре.

Конвенция:

- основной SDLC проекта: `docs/sdlc/00_Main/` (создаётся/обновляется через `/sdlc.main`)
- sub‑SDLC: `docs/sdlc/NN_<folder>/` (например `01_GameIdeas`, `02_MlirDialects`)

### Input

- **folder** — имя подпроцесса (без префикса `NN_`, например `game_ideas`, `mlir_dialects`, `Release_Gates`). Папка будет создана как `docs/sdlc/NN_<folder>/`.
- **orderNN** — опционально: номер для префикса (две цифры, например `03`). Если не задан — выбрать следующий свободный номер среди `docs/sdlc/NN_*`.
- **goal_title** — краткое название процесса (H1 в README).
- **goal_description** — 1–2 предложения цели (блок «Цель» в README). Чётко и конкретно.
- **slug** — опционально: машиночитаемый id для реестра (lowercase, ASCII, подчёркивания). Если не задан — нормализовать из folder (пробелы → `_`, lowercase).
- **template** — опционально: `default` (00_Status, 01_BA, 02_Architecture, 03_Specs, 04_QA) или `minimal` (00_Status, 01_Work, 02_Output) или свой список подпапок через запятую.
- **domainTags** — опционально: теги для реестра (например `ideas`, `mlir`, `release`).

### Output (шаги)

1. Определить целевую папку `docs/sdlc/NN_<folder>/` (00_Main зарезервирована и не используется здесь).
2. Создать `docs/sdlc/NN_<folder>/README.md`: H1 = goal_title, блок **Цель** (goal_description), таблица подпапок по выбранному шаблону.
2. Создать подпапки с кратким README в каждой (назначение папки).
3. Добавить запись в `docs/sdlc/00_registry.json`: sdlcId (`sdlc.<slug>`), slug, path (`docs/sdlc/NN_<folder>/`), goal, goalShort (из goal_title или первой фразы goal_description), domainTags.

### Constraints

- Папка с таким именем не должна уже существовать: не создавать, если `docs/sdlc/NN_<folder>/` уже есть.
- После выполнения — атомарный коммит по правилам репо (02_atomic_git, 15_commit_push_shortcuts).

### Референс

- Корень и реестр: `docs/sdlc/README.md`, `docs/sdlc/00_registry.json`.
- Примеры: `docs/sdlc/01_GameIdeas/`, `docs/sdlc/02_MlirDialects/`, `docs/sdlc/00_Main/`.
