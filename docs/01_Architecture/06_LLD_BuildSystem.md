# TT-Lang Architecture — Low Level Design: Build System

## 1. Назначение

Этот документ кратко фиксирует архитектуру сборки tt‑lang и привязки к tt‑mlir toolchain.

Источник истины по механике сборки: `docs/BUILD_SYSTEM.md`.
Здесь — LLD‑конспект с “крючками”, куда смотреть при проблемах.

## 2. Основная идея

tt‑lang использует CMake сборку, которая **переиспользует окружение и toolchain tt‑mlir** (LLVM/MLIR, Python venv, зависимости), чтобы:

- не дублировать сборку LLVM,
- гарантировать совместимость версий,
- стабилизировать поведение пайплайна.

## 3. Сценарии интеграции с tt‑mlir (L1)

См. `docs/BUILD_SYSTEM.md`:

- **Scenario 1: Pre-built tt-mlir (Dev Mode)**
  - указать `TTMLIR_BUILD_DIR`
- **Scenario 2: Pre-installed tt-mlir (Recommended)**
  - использовать `TTMLIR_TOOLCHAIN_DIR` (по умолчанию `/opt/ttmlir-toolchain`)
- **Scenario 3: Automatic build**
  - взять commit из `third-party/tt-mlir.commit`, собрать локально в build

## 4. Ключевые директории build артефактов

Типично:

- `build/bin/` — `ttlang-opt`, `ttlang-translate`
- `build/tt-mlir-install/` — локальная установка tt‑mlir (для scenario 3)
- `build/_deps/tt-mlir-src/` — исходники tt‑mlir (FetchContent)

## 5. Типовые проблемы

- “не найден toolchain” → проверить `TTMLIR_TOOLCHAIN_DIR` / `TTMLIR_BUILD_DIR`.
- “несовместимые символы/линковка” → проверить, что toolchain и tt‑mlir совпадают по версии и собраны с нужными биндингами.
- “python/venv mismatch” → убедиться, что активирован правильный env из toolchain.

