# Canonical .cursor/commands (shared across portfolio)

Shared slash-commands set for portfolio projects. Source: RulesEngineUniverse workflow (commands-first: curs.*, sdlc.*, hat.*, release.aaa). Project-specific commands (e.g. game.extract for REU only) live in `docs/projects/<project>/.cursor/commands/` and are overlaid when syncing that project.

- **Propagation**: `tools/sync_portfolio_to_upstreams.py` copies this folder to each upstream `.cursor/commands`.
- **Update**: When REU or another project improves commands, snapshot into `docs/projects/<project>/commands.upstream-*`, then copy the chosen set here and re-run propagation.

**SDLC layout** (новая структура `docs/sdlc/00_Main/`):
- `/sdlc.main` — создать/привести основной SDLC (00_Main) и sub‑SDLC.
- `/sdlc.create` — завести новый sub‑SDLC (01_*, 02_*, …).
- `/sdlc.migrate` — перейти со старой плоской структуры `docs/NN_Name/` на `docs/sdlc/00_Main/` (запускать вручную по репо, затем коммит/пуш).
- `/sdlc.status`, `/sdlc.audit` — снимок статуса и аудит согласованности.
