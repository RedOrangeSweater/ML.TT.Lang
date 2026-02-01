# SDLC tt_llk (LLK)

**Цель**: доки по низкоуровневым ядрам LLK; репо tt_llk внутри tt-metal (`tt_metal/third_party/tt_llk/`); связь с tt-lang и tt-metal.

**С чего начать (tt-llk SDLC)**: [02_Architecture/00_DirectionAndLayers.md](02_Architecture/00_DirectionAndLayers.md) — направление, слои и контракты Python test infra (Pydantic).

| Подпапка | Назначение |
|----------|------------|
| `00_Status/` | Dashboard, Kanban, статус-снимки по LLK |
| `01_BA/` | Требования (REQ, AC) |
| `02_Architecture/` | ADR, архитектура ядер |
| `03_Specs/` | Спецификации, таски |
| `04_QA/` | Чеклисты, тест-планы |

Связанные процессы: [00_Main](../00_Main/) (tt-lang), [01_tt_metal](../01_tt_metal/) (tt-metal).

**Связанные доки**: tt-lang lowering/specs: [00_Main/03_Specs](../00_Main/03_Specs/) (LOWERING_MULTITILE, DST_Allocation). Pipeline Doc–MLIR–GraphDB: [01_tt_metal/02_Architecture/00_Doc_MLIR_GraphDB_Pipeline.md](../01_tt_metal/02_Architecture/00_Doc_MLIR_GraphDB_Pipeline.md).

Каноническое расположение SDLC tt_llk — **здесь** (tt-lang). В репо tt_llk только стаб: `docs/sdlc/README.md` указывает сюда.
