# Результаты проверки тестов (check tests)

## Состояние на 2026-02-01

### 1. Ninja / CMake

- **`cmake --build build --target check-ttlang`** и **`ninja check-ttlang-pytest`** падают с ошибкой:
  ```text
  ninja: error: build.ninja:35: loading 'CMakeFiles/rules.ninja': No such file or directory
  ```
- В `build/` нет `CMakeFiles/rules.ninja` — сборка в неконсистентном состоянии. Нужна переконфигурация: из корня репо `cmake -G Ninja -B build .` (и при необходимости пересборка).

### 2. Pytest при ручном запуске

- Запуск: `source build/env/activate` затем `python -m pytest test/python/test_dialect_layers.py -v --tb=short` (из корня или из `test/python`).
- **Ошибка при сборе тестов:** `ModuleNotFoundError: No module named 'ttl._mlir_libs._ttlang'`.
- **Причина:** в `build/python_packages/ttl/` часть файлов — симлинки на `python/ttl/` (в т.ч. `__init__.py`). При загрузке пакета `ttl` из `build` интерпретатор разрешает симлинк и считает корнем пакета каталог `python/ttl/`. Тогда `ttl.__path__` указывает на `python/ttl/`, где нет `_mlir_libs` (он есть только в `build/python_packages/ttl/`), поэтому импорт `ttl._mlir_libs._ttlang` не находит модуль.
- **Дополнительно:** в build-дереве не хватало модулей:
  - `build/python_packages/ttl/boundary/` — пустая директория (в источнике есть `boundary/__init__.py`, `ttnn_types.py`, `mlir_types.py`).
  - `build/python_packages/ttl/runtime/runner.py` — отсутствовал (есть только симлинк `__init__.py`).
- Вручную добавлены симлинки: `boundary` -> `../../../python/ttl/boundary`, `runtime/runner.py` -> `../../../../../python/ttl/runtime/runner.py`. После этого импорт доходит до `ttl._mlir_libs._ttlang`, но при другом порядке путей снова подхватывается пакет из `python/ttl/` и ошибка `_mlir_libs` повторяется.

### 3. Рекомендации

- **Пересобрать дерево сборки:** из корня репо выполнить `cmake -G Ninja -B build .` (и при необходимости `cmake --build build`), чтобы появился `CMakeFiles/rules.ninja` и цели вроде `check-ttlang`/`check-ttlang-pytest` были доступны.
- **Запуск pytest через CMake:** после исправления сборки запускать тесты через цель:
  ```bash
  cmake --build build --target check-ttlang-pytest
  ```
  (рабочая директория и `PYTHONPATH` задаются так, чтобы использовался пакет из `build`).
- **Run configs (VSCode/Cursor):** в `.vscode/launch.json` и `.vscode/tasks.json` для pytest и примеров задаётся `PYTHONPATH=build/python_packages:python:test`. Это позволяет запускать тесты и примеры из IDE при наличии собранного дерева; при проблемах с симлинками надёжный способ — цель `check-ttlang-pytest`.
- **Устранить зависимость от симлинков для корня пакета:** чтобы при загрузке `ttl` из `build/python_packages/ttl` пакет не «переезжал» в `python/ttl` из-за разрешения симлинка на `__init__.py`, в CMake стоит либо:
  - копировать (а не симлинчить) `python/ttl/*` в `build/python_packages/ttl/`, либо
  - собирать в build один самодостаточный пакет `ttl` (с `_mlir_libs` и всеми модулями), без симлинков, ведущих в `python/ttl`, для файлов, по которым вычисляется `__path__`.

После выполнения этих шагов повторно проверить:  
`cmake --build build --target check-ttlang-pytest` и при необходимости `test/python/test_dialect_layers.py`.
