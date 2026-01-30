## /curs.apply — применить правки по результатам аудита

Применить группы правок (diff-surface) из `/curs.audit` или `/sdlc.audit` строго по описанным шагам.

### Input

Опционально: путь к TASK, список групп (напр. `A, B`) или пусто → все группы из последнего TASK аудита в `docs/sdlc/00_Main/00_Status/`.

### Output

- **Applied**: группы (A–D или из sdlc.audit).
- **Files changed**: список.
- **Skipped**: группа + причина.

### Next step

Коммит — atomic workflow (`02_atomic_git`, `15_commit_push_shortcuts`). Пуш только по запросу.

### Constraints

Только файлы из выбранных групп (`.cursor/`, `.vscode/`, указанные `docs/`). Без логики приложения. Шаги только из TASK; если не описано — в выводе «требуется уточнение».
