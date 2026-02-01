# GraphDB: схема и примеры (cursorrules)

**В репо канон** — этот каталог (`.cursor/artifacts_mlir_graphdb/graphdb/`). В портфолио-репо опционально: `docs/sdlc/_KG_MLIR/common/graphdb/` (синхронизация в проекты через propagate).

## Схема и примеры

- **schema.ttl** — схема KG (типы, отношения).
- **examples/** — примеры .ttl (факты) и .rq (SPARQL-запросы).

## Соответствие запросов этапам/диалектам

| Запрос (examples/) | Этап / слой | Диалект MLIR |
|--------------------|-------------|--------------|
| intro_section.rq, intro_para_1.rq, intro_chain.rq | intro / цепочка | reu.sdlc_kg, reu.sdlc_doc_intent |
| doc_layer.rq, mlir_doc_layer.rq, mlir_passes.rq, graphdb_layer.rq, next_mlir.rq, final_llm.rq | representation_layers (§2) | reu.sdlc_kg, reu.sdlc_doc_intent |
| links_section.rq | Ссылки | reu.sdlc_kg, reu.sdlc_doc_intent |

Реестр MLIR: [../mlir/README.md](../mlir/README.md). Пайплайн: [../../../00_Main/02_Architecture/00_Doc_MLIR_GraphDB_Pipeline.md](../../../00_Main/02_Architecture/00_Doc_MLIR_GraphDB_Pipeline.md).
