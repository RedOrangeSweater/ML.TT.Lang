# Трансформация: reu.sdlc_kg → reu.sdlc_doc_intent

**Вход**: MLIR слой `reu.sdlc_kg` (kg_core).  
**Выход**: MLIR слой `reu.sdlc_doc_intent` (lowering/selection to doc intent).

Описание: по материализованному подграфу формируется claim для генерации — ссылка на kg-слой (kg_ref), язык, стиль. Текста абзаца нет; LLM восстановит по графу.

Реестр: [../README.md](../README.md).
