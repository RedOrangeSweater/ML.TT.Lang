# Canonical .cursor/commands (shared across portfolio)

Shared slash-commands set for portfolio projects. Source: RulesEngineUniverse workflow (commands-first: curs.*, sdlc.*, hat.*, release.aaa). Project-specific commands (e.g. game.extract for REU only) live in `docs/projects/<project>/.cursor/commands/` and are overlaid when syncing that project.

- **Propagation**: `tools/sync_portfolio_to_upstreams.py` copies this folder to each upstream `.cursor/commands`.
- **Update**: When REU or another project improves commands, snapshot into `docs/projects/<project>/commands.upstream-*`, then copy the chosen set here and re-run propagation.

## Naming and scale

- **Format**: `domain.action[.subaction].md` — two or three segments, no spaces, dot-separated.
- **Domains**: sdlc.*, code.*, git.*, rules.*, curs.*, hat.*, release.*. Use 2–3 segments so the set remains scannable as it grows (e.g. toward hundreds or thousands of commands).
- **Examples**: sdlc.audit, sdlc.change.preflight, code.validation.pydantic, rules.update.from-feedback.

## SDLC layout (новая структура `docs/sdlc/00_Main/`)

- `/sdlc.newrepo` — пустой проект с полной структурой: копировать template в целевой репо, затем вызвать `/sdlc.main` (target, path, subProjects).
- `/sdlc.main` — создать/привести основной SDLC (00_Main) и sub‑SDLC.
- `/sdlc.create` — завести новый sub‑SDLC (01_*, 02_*, …).
- `/sdlc.migrate` — перейти со старой плоской структуры `docs/NN_Name/` на `docs/sdlc/00_Main/` (запускать вручную по репо, затем коммит/пуш).
- `/sdlc.status`, `/sdlc.audit` — снимок статуса и аудит согласованности.
- `/sdlc.change.preflight` — после изменения кода/контрактов: проверить SDLC-документы, влияние на заказчика, обновить доки/правила при необходимости, напомнить commit+push.

## Code and rules

- `/code.validation.pydantic` — убедиться, что валидация только в Pydantic; найти validate_* в бизнес-логике и предложить перенос в модели.
- `/rules.update.from-feedback` — когда пользователь поправил или напомнил — зафиксировать в правилах/командах, чтобы не повторять.
