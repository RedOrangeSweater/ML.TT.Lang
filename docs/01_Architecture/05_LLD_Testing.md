# TT-Lang Architecture — Low Level Design: Testing

## 1. Назначение

Этот документ описывает тестовую инфраструктуру tt‑lang: какие типы тестов существуют, что они проверяют и где их искать.

## 2. Основные категории тестов

### 2.1 MLIR lit tests (компиляторные)

Назначение:

- проверить диалекты,
- проверить конверсии/легализацию,
- проверить трансляции (например, TTL -> C++ через TTKernel/EmitC).

Где:

- `test/ttlang/` (Dialects / Conversion / Translate)

### 2.2 Python tests (DSL / runtime)

Назначение:

- проверить Python API,
- проверить интеграцию с runtime,
- (часто) требуют устройство.

Где:

- `test/python/`

### 2.3 Simulator tests

Назначение:

- функциональная проверка симулятора (без железа).

Где:

- `test/sim/`

### 2.4 ME2E (модельные E2E)

Назначение:

- “более реальная” проверка end‑to‑end сценариев.

Где:

- `test/me2e/`

## 3. Типовой пайплайн lit теста (примерно)

```mermaid
sequenceDiagram
  participant Lit as llvm-lit
  participant Opt as ttlang-opt
  participant Tr as ttlang-translate
  participant FC as FileCheck

  Lit->>Opt: "run_pipeline(input.mlir)"
  Opt-->>Lit: "output_stage.mlir"
  Lit->>Tr: "translate(output_stage.mlir)"
  Tr-->>Lit: "output.cpp"
  Lit->>FC: "FileCheck(patterns, output.cpp)"
  FC-->>Lit: "pass_or_fail"
```

## 4. Где чаще всего ломается (и что делать)

- Изменился lowering → сломались FileCheck ожидания в `test/ttlang/Translate/`.
- Изменился legality/type conversion → `failed to legalize operation`.
- Изменился порядок строк/вставились дополнительные вызовы → править `CHECK-NEXT` vs `CHECK`.

## 5. Правила поддерживаемости

- тесты должны проверять **инварианты**, а не случайный порядок,
- избегать слишком хрупких `CHECK-NEXT`, если вставки возможны по дизайну,
- добавлять отдельные маленькие регрессии под критические кейсы.

