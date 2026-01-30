# Doc–MLIR–GraphDB pipeline: уровни представления и финальная генерация текста

**Цель**: описать пайплайн документации с учётом трансляции в MLIR и GraphDB: после каждого логического куска — представление в MLIR (сноска); цепочка MLIR-трансформаций; GraphDB-запросы как следующий уровень MLIR; финальная стадия — генерация литературно-технического текста LLM из MLIR.

<!-- mlir: .cursor/artifacts_mlir_graphdb/mlir/doc_pipeline.mlir [#intro] dialect=ttm.sdlc_doc -->

---

## 1. Введение

Документация ведётся с расчётом на трансляцию в MLIR и GraphDB. После каждого логического куска (абзац, раздел, список требований) добавляется **представление в MLIR** — сноска или комментарий с путём к MLIR-файлу и (по возможности) якорем и именем диалекта.

Цепочка представлений: **Doc ↔ MLIR** (двунаправленно); **MLIR-трансформации** (серия пассов); **GraphDB-запросы** → следующий уровень MLIR; **финальная стадия** — генерация литературно-технического текста LLM из MLIR.

<!-- mlir: .cursor/artifacts_mlir_graphdb/mlir/doc_pipeline.mlir [#intro_chain] dialect=ttm.sdlc_doc -->

---

## 2. Уровни представления

### 2.1. Документ (Markdown / структура)

Человекочитаемый текст и структура: заголовки, абзацы, списки. Каждый логический кусок имеет **сноску** на соответствующий MLIR (файл + опционально якорь, диалект). Такая сноска обеспечивает прослеживаемость: из какого MLIR можно сгенерировать этот абзац и в какой MLIR он транслируется.

<!-- mlir: .cursor/artifacts_mlir_graphdb/mlir/doc_pipeline.mlir [#doc_layer] dialect=ttm.sdlc_doc -->

### 2.2. MLIR (документ-как-структура)

Слой MLIR, в который транслируется структура документа (секции, абзацы, требования). Для этого слоя объявляется **диалект** `ttm.sdlc_doc`. Этот MLIR — **источник** для обратной генерации абзаца LLM: doc ↔ MLIR bidirectional.

<!-- mlir: .cursor/artifacts_mlir_graphdb/mlir/doc_pipeline.mlir [#mlir_doc_layer] dialect=ttm.sdlc_doc -->

### 2.3. Цепочка MLIR-трансформаций

Все преобразования до финального текста выполняются как **серия MLIR-пассов** (lowering, специализация, оптимизация и т.д.). Каждый пасс описан; потребляемый и производимый диалекты зафиксированы в **реестре трансформаций**. Таким образом, до этапа генерации текста все преобразования — детерминированные MLIR-трансформации.

<!-- mlir: .cursor/artifacts_mlir_graphdb/mlir/doc_pipeline.mlir [#mlir_passes] dialect=ttm.sdlc_doc -->

### 2.4. GraphDB

Граф сущностей, связей и атрибутов. **Запросы к GraphDB** порождают следующий уровень MLIR: запросы возвращают структурированный результат, который отображается в MLIR следующего слоя. Необходимо явно описать: как запросы к GraphDB сопоставляются с операциями и типами в MLIR (см. мэппинг GraphDB→MLIR в Negative SDLC PreGraphDB).

<!-- mlir: .cursor/artifacts_mlir_graphdb/mlir/doc_pipeline.mlir [#graphdb] dialect=ttm.sdlc_doc -->

### 2.5. Следующий уровень MLIR

Результат GraphDB-запросов и/или предыдущих MLIR-пассов. Должен быть описан так же явно: объявление **диалекта** для слоя, регистрация в **реестре трансформаций** (какой пасс/запрос потребляет какие диалекты и что производит).

<!-- mlir: .cursor/artifacts_mlir_graphdb/mlir/doc_pipeline.mlir [#next_mlir] dialect=ttm.sdlc_doc -->

### 2.6. Финальная стадия генерации

В будущем **LLM** использует финальный MLIR для генерации **литературно-технического текста** (один или несколько проходов). До этой стадии все преобразования — детерминированные MLIR-трансформации и GraphDB; LLM применяется только на выходе: MLIR → текст.

<!-- mlir: .cursor/artifacts_mlir_graphdb/mlir/doc_pipeline.mlir [#final_llm] dialect=ttm.sdlc_doc -->

---

## 3. Правила ведения

- **После каждого абзаца/раздела**: сноска в едином формате (HTML-комментарий или markdown-сноска): путь к MLIR (канон: `.cursor/artifacts_mlir_graphdb/mlir/`), опционально якорь, диалект. Пример: `<!-- mlir: .cursor/artifacts_mlir_graphdb/mlir/doc_pipeline.mlir [#anchor] dialect=ttm.sdlc_doc -->`.
- **Реестр диалектов**: для каждого MLIR-слоя/файла объявлять диалект (в атрибутах модуля или в `.cursor/artifacts_mlir_graphdb/mlir/README.md`; см. также [Artifacts/mlir/README.md](Artifacts/mlir/README.md) в этом каталоге).
- **Реестр трансформаций**: перечень пассов/пайплайнов (doc→MLIR, GraphDB→MLIR, lowering, …) с указанием потребляемых и производимых диалектов.

---

## 4. Ссылки

- Common SDLC Core §3.0.1 (Doc→MLIR/GraphDB traceability).
- Negative SDLC PreGraphDB: мэппинг GraphDB→MLIR (референс в cursorrules/docs).
- Правило Doc–MLIR footnotes (REU): сноски после логических кусков, трек диалектов и трансформаций.

**Related**: tt-lang architecture: [00_Main/02_Architecture](../../00_Main/02_Architecture/). LLK: [02_llk/02_Architecture](../../02_llk/02_Architecture/).
