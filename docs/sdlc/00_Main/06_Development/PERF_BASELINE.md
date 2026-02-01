# Baseline и правило «без регресса» (питч рефакторинга)

- **Назначение**: зафиксировать baseline по commit hash main и набору тестов; после рефакторинга убедиться, что те же тесты проходят (и при необходимости perf не хуже).
- **Связано**: [TASK_pytorch_like_modules_pitch.md](TASK_pytorch_like_modules_pitch.md).

## Как зафиксировать baseline (main)

1. На ветке `main`: `git rev-parse HEAD` — сохранить хеш (например в `scripts/baseline_commit.txt` или в описании питча).
2. Запустить набор проверок:
   - `cmake --build build --target check-ttlang-mlir`
   - `cmake --build build --target check-ttlang-python-lit`
   - `cmake --build build --target check-ttlang-pytest`
3. Убедиться, что все проходят; при необходимости зафиксировать вывод или только факт pass/fail.

Опционально (при наличии устройства): прогнать минимальный perf smoke (один пример, замерить время) и записать результат для сравнения.

## Как проверить после рефакторинга

1. Запустить те же цели: `check-ttlang-mlir`, `check-ttlang-python-lit`, `check-ttlang-pytest`.
2. Правило: все тесты, которые проходили на baseline (commit main), должны проходить и после рефакторинга.
3. При наличии baseline по perf: повторить perf smoke и убедиться, что метрика не хуже (в пределах шума).

## Скрипты

- `scripts/baseline_record.sh` — записывает текущий `git rev-parse HEAD` и результат поднабора тестов.
- `scripts/baseline_compare.sh` — по сохранённому baseline запускает те же тесты и сообщает pass/fail.

Использование: из корня репо после `source build/env/activate` (или с активированным venv).
