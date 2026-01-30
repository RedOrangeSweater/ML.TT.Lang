## /curs.audit — аудит .cursor/ (rules, commands, skills)

Русский, структура без воды. Цель: проблемы в правилах/командах/скиллах, консистентность доков по правилам, экспорт в SDLC (TASK + план в месяцах).

### Input

Опционально: подмножество (`rules` | `commands` | `skills`), путь к TASK. **Пусто** → полный layered audit.

### Слои (без параметров)

1. **Rules**: description/frontmatter, нумерация, ссылки на docs, дубли.
2. **Commands**: Input/Output/Next step/Constraints, имена, битые ссылки.
3. **Skills**: name/description, триггеры, deprecated.
4. **Cross-layer**: правила ↔ команды ↔ навыки ↔ docs.
5. **Feedback**: этап оценки/доков — предложить правки правил, если пропущен.

### Output

- **Scope**; **Findings** (🔴/🟡/🟢); **Группы правок (diff-surface)**; **Предложения по правилам**; **Экспорт в SDLC** (TASK в `docs/sdlc/00_Main/00_Status/`, план в месяцах).

### Constraints

Только `.cursor/`, `.vscode/`, выбранные `docs/`. Применение правок — **только через** `/curs.apply` (правило `08_cursor_actions_only.mdc`). Коммит/пуш — atomic, без auto-push.
