## /sdlc.migrate — перейти со старой плоской структуры docs на docs/sdlc/00_Main

Цель: аккуратно перенести документы из **плоской** структуры `docs/NN_Name/` в **новую** структуру `docs/sdlc/00_Main/NN_Name/` с сохранением истории (`git mv`) и без смешивания с другими изменениями.

### Когда использовать

- В репо ещё **старая** структура: `docs/00_Status/`, `docs/01_Architecture/`, `docs/03_BA/`, `docs/06_Development/`, `docs/07_Testing/`, `docs/09_Operations/` и т.п.
- Нужно перейти на **новую** структуру как в RulesEngineUniverse: `docs/sdlc/00_Main/` с подпапками 00_Status, 01_BA, 02_Architecture, 03_Specs, 04_QA (и при необходимости 00_Ideas, 05_Operations и др.).

### Стандартное соответствие (old → new)

| Было (плоско) | Стало (под 00_Main) |
|---------------|---------------------|
| `docs/00_Status/` | `docs/sdlc/00_Main/00_Status/` |
| `docs/00_Ideas/` | `docs/sdlc/00_Main/00_Ideas/` (или отдельный sub‑SDLC `01_Ideas` — по решению проекта) |
| `docs/01_Architecture/` | `docs/sdlc/00_Main/02_Architecture/` |
| `docs/03_BA/` | `docs/sdlc/00_Main/01_BA/` |
| `docs/06_Development/` | `docs/sdlc/00_Main/03_Specs/` |
| `docs/07_Testing/` | `docs/sdlc/00_Main/04_QA/` |
| `docs/09_Operations/` | `docs/sdlc/00_Main/05_Operations/` или `09_Operations/` (по конвенции проекта) |

Папки `docs/common/`, `docs/projects/` и т.п. **не переносить** — они остаются на верхнем уровне `docs/`.

### Output (шаги, выполнять по очереди)

1. **Создать скелет новой структуры** (если ещё нет):
   - Вызвать `/sdlc.main` и выполнить шаги создания `docs/sdlc/00_Main/` с подпапками 00_Status, 01_BA, 02_Architecture, 03_Specs, 04_QA и README.
   - Либо вручную создать `docs/sdlc/`, `docs/sdlc/00_Main/`, `docs/sdlc/00_registry.json`, `docs/sdlc/README.md` по образцу из команды `/sdlc.main`.

2. **Перенести содержимое только через `git mv`** (чтобы сохранить историю):
   - Для каждой существующей папки из таблицы: если целевая папка под `00_Main` уже создана и пуста — переместить содержимое старых папок в соответствующие новые (например `git mv docs/00_Status/* docs/sdlc/00_Main/00_Status/` и затем удалить пустую `docs/00_Status`).
   - Либо: `git mv docs/00_Status docs/sdlc/00_Main/00_Status` и т.д., если переименование 1:1 допустимо (для 01_Architecture → 02_Architecture и 03_BA → 01_BA потребуется сначала создать папку, перенести файлы, затем удалить старую).
   - Не трогать `docs/common/`, `docs/README.md` (обновить ссылки в шаге 4).

3. **Удалить пустые старые папки** после переноса (если остались).

4. **Обновить навигацию и ссылки**:
   - `docs/sdlc/00_registry.json` — запись `sdlc.main` с path `docs/sdlc/00_Main/` (если ещё нет).
   - `docs/sdlc/README.md` — описать 00_Main и входные точки.
   - `docs/README.md` — заменить ссылки на старые пути (`docs/01_Architecture/` и т.д.) на `docs/sdlc/00_Main/02_Architecture/` и т.д.
   - По репо: правила `.cursor/rules`, команды, упоминания путей в доках — заменить старые пути на новые.

5. **Коммиты**:
   - Атомарно: один коммит на миграцию layout (git mv + реестр + README), при необходимости отдельно — правки правил/ссылок.
   - После проверки: закоммитить, по запросу — пуш.

### Constraints

- Использовать только `git mv` для переноса файлов/папок, чтобы не терять историю.
- Не смешивать миграцию структуры с содержательными изменениями в доках.
- Если в репо уже есть `docs/sdlc/00_Main/` и часть контента — только дополнить по таблице и обновить ссылки.

### Референс

- Новая структура: `docs/sdlc/00_Main/`, реестр: `docs/sdlc/00_registry.json`.
- Команды: `/sdlc.main`, `/sdlc.create`, `/sdlc.status`, `/sdlc.audit`.
