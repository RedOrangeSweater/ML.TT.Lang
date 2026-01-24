# TT-Lang Architecture — Low Level Design: Compiler Pipeline

## 1. Назначение

Этот документ детализирует компиляционный пайплайн tt‑lang: от Python DSL к TTL IR, далее к TTKernel/EmitC и к C++ артефактам. Документ “приземляет” High‑Level дизайн на конкретные файлы и точки расширения.

## 2. Артефакты и стадии (L1)

Ниже — логическая последовательность стадий (не обязательно 1:1 по бинарям):

```mermaid
flowchart LR
  Py["Python_DSL"] --> TTL["TTL_IR_(MLIR)"]
  TTL --> P0["TTL_Pass_Pipeline"]
  P0 --> TTKernel["TTKernel_IR"]
  TTKernel --> P1["TTKernel_to_EmitC"]
  P1 --> EmitC["EmitC_IR"]
  EmitC --> CPP["C++_Output"]
```

## 3. Где задаётся пайплайн

- TTL pipeline: `lib/Dialect/TTL/Pipelines/TTLPipelines.cpp`
  - регистрирует pipeline `ttl-to-ttkernel-pipeline` (см. `registerTTLPipelines()`).
- Трассировка примера multi‑tile lowering: `docs/LOWERING_MULTITILE.md`

## 4. Ключевые passes (TTL)

Набор “опорных” пасов, которые обычно встречаются в pipeline (точные пайплайны могут отличаться):

| Pass | Роль | Код |
|---|---|---|
| `convert-ttl-to-compute` | перевод tensor‑ops в `ttl.compute` регионы | `lib/Dialect/TTL/Transforms/ConvertTTLToCompute.cpp` |
| `ttl-assign-dst` | аллокация DST и вставка `ttl.copy_tile` (когда нужно) | `lib/Dialect/TTL/Transforms/TTLAssignDST.cpp` |
| `ttl-insert-tile-regs-sync` | синхронизация/жизненный цикл DST regs | `lib/Dialect/TTL/Transforms/TTLInsertTileRegsSync.cpp` |
| `ttl-lower-to-loops` | lowering compute к `scf.for` | `lib/Dialect/TTL/Transforms/ConvertTTLComputeToSCF.cpp` |
| `convert-ttl-to-ttkernel` | lowering TTL ops к TTKernel ops | `lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp` |

Примечание: порядок и набор зависит от конкретного pipeline (см. `TTLPipelines.cpp`).

## 5. Драйверы инструментов (CLI)

- `ttlang-opt`: `tools/ttlang-opt/ttlang-opt.cpp`
  - MLIR opt‑драйвер (регистрирует диалекты и TTL pipelines, затем вызывает `MlirOptMain`).
- `ttlang-translate`: `tools/ttlang-translate/ttlang-translate.cpp`
  - MLIR translate‑драйвер (включая регистрацию `--ttkernel-to-cpp`).

Примеры (как “payload”):

- TTL -> TTKernel:
  - `ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize input.mlir -o out.ttkernel.mlir`
- TTL -> EmitC (через опцию pipeline):
  - `ttlang-opt --ttl-to-ttkernel-pipeline="lower-to-emitc=1" input.mlir -o out.emitc.mlir`
- EmitC/TTKernel -> C++:
  - `ttlang-translate --ttkernel-to-cpp -o out.cpp out.emitc.mlir`

## 6. Типовые ошибки и где они возникают

### 6.1 Verification/Legality ошибки

Симптомы:

- “failed to legalize operation …”
- “verification failed …”

Где смотреть:

- конверсия/легализация: `lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp`
- тесты преобразования: `test/ttlang/Conversion/`

### 6.2 Ошибки трансляции (TTKernel/EmitC -> C++)

Симптомы:

- “unsupported op …”
- “translation failed …”

Где смотреть:

- `ttlang-translate` + связанные переводчики
- тесты `test/ttlang/Translate/TTLToCpp/`

## 7. Точки расширения (как развивать)

- Добавить новый pass:
  - объявление/регистрация в соответствующем `Passes.td` (если используется TableGen паттерн),
  - реализация в `lib/Dialect/.../Transforms/`.
- Добавить pipeline:
  - в `lib/Dialect/TTL/Pipelines/TTLPipelines.cpp`.
- Добавить тест:
  - `test/ttlang/Conversion/*` или `test/ttlang/Translate/*`.

