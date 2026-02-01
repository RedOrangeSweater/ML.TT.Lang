# Трансформация: SPARQL → reu.sdlc_kg

**Вход**: результат SPARQL (подграф/факты из GraphDB).
**Выход**: MLIR слой `reu.sdlc_kg` (materialize query result).

Описание: запрос к GraphDB возвращает подграф (Intent, Effect, концепты); результат отображается в операции `sdlc_kg.intent`. Без CONTAINS по фразам — только по типам и IRIs.

Реестр: [../README.md](../README.md).
