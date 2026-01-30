# Реестр MLIR-слоёв и трансформаций (cursorrules)

Канонический **SoT** — KG (GraphDB) + MLIR intent/effect. Текст генерируется только на последнем шаге (LLM из финального MLIR). См. [00_Doc_MLIR_GraphDB_Pipeline.md](../00_Doc_MLIR_GraphDB_Pipeline.md) и `Artifacts/graphdb/`.

**Структура**: по путям видно этап → диалект → трансформация.

- **Диалекты**: [dialects/](dialects/) — один файл на диалект (описание, операции/типы).
- **Трансформации**: [transformations/](transformations/) — один файл на пасс/пайплайн (входной/выходной диалект, описание).

## Диалекты (слои)

| Диалект | Назначение |
|---------|------------|
| `reu.sdlc_kg` (kg_core) | Представление KG-структур в MLIR: концепты, отношения, эффекты (без текстовых фраз). |
| `reu.sdlc_doc_intent` (doc_intent) | Intent/effect для генерации документа: ссылки на KG ids, стиль/язык; без текстового content. |
| `reu.sdlc_doc` | Устаревший doc-слой с content-строками; вытесняется KG-first (kg_core + doc_intent). |

## Трансформации (KG-first)

| Пасс / пайплайн | Вход | Выход |
|-----------------|------|-------|
| GraphDB/SPARQL → reu.sdlc_kg | SPARQL result (подграф/факты) | `reu.sdlc_kg` (materialize query result) |
| reu.sdlc_kg → reu.sdlc_doc_intent | MLIR kg_core | `reu.sdlc_doc_intent` (lowering/selection to doc intent) |
| reu.sdlc_doc_intent → Text | MLIR doc_intent | LLM generation (контракт) |

Обновлять при добавлении новых пассов или диалектов.

## Генерация из MLIR и проверка консистентности через GraphDB

- **Генерация из MLIR (LLM)**: финальный MLIR (`reu.sdlc_doc_intent`) является входом для генерации литературно-технического текста; в KG и MLIR **не хранятся фразы** — только понятия, отношения, эффекты.
- **Консистентность**: загрузка в GraphDB по SPARQL/триплам, затем сверка графа с MLIR; при добавлении новых уровней — объявлять диалекты и трансформации в этом реестре.

## Проверки (для будущего tooling)

- **KG→MLIR completeness**: каждый doc-intent элемент имеет покрытие в KG (есть подграф/ответ на запрос).
- **No-phrase-in-KG**: в KG нет текстовых literal-ов для content/phrase (только ids, типы, отношения).
