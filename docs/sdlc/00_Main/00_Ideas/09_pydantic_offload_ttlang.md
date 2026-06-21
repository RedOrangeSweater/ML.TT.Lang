## Контекст

Цель состоит в централизации валидации и сопоставления параметров Python API tt-lang
в небольших моделях Pydantic, а также в переносе настроек окружения в
pydantic-settings. Это снижает объем ручной проверки в декораторах и
упрощает сопровождение параметров окружения.

## SDLC (локальный трекинг)

- **Статус**: в работе.
- **Область изменений**:
  - `python/ttl/settings.py`: единый `TTLangSettings` + `get_settings()`.
  - `python/ttl/ttl_api.py`: замена `os.environ` чтений на settings.
  - `python/ttl/diagnostics.py`: замена `TTLANG_VERBOSE_ERRORS` чтения на settings.
  - `python/setup.py`, `requirements.txt`: добавить `pydantic-settings` и обновить зависимости.
- **Ограничения**:
  - Семантика по умолчанию не должна меняться.
  - Валидация должна оставаться строгой, но декларативной.
  - Комментарии в коде - на английском, документация - на русском.

## Цели

- Централизовать параметры окружения TTLANG в одном объекте настроек.
- Свести валидацию параметров `pykernel_gen` к декларативным моделям.
- Разделить тяжелую работу (парсинг AST, извлечение кода) и валидацию.

## Границы

- Изменения не должны менять семантику выполнения или формат MLIR по умолчанию.
- Валидация остается строгой и проверяет только параметры ввода.
- Инварианты исполнения и бизнес-логика остаются в явных проверках кода.

## Риски

- Ошибки валидации могут изменить тип исключений и тексты сообщений.
- Некорректная обработка пустых значений переменных окружения.

## План

- Ввести `TTLangSettings` и использовать его в `ttl_api.py` и `diagnostics.py`.
- Вынести опции `pykernel_gen` в Pydantic-модель и нормализовать значения.
- Добавить `pydantic-settings` в зависимости и обновить типизацию в правках.

## Было / стало

| Было | Стало |
| --- | --- |
| `validate_*` и `os.environ` разбросаны по `ttl_api.py`, `diagnostics.py` | `TTLangSettings` + Pydantic request models |
| Ad-hoc проверки loop params в AST visitor | Pydantic models в `visit_For` (`f6515298`) |
| Fused YAML в tt-llk без typed contract | `fuser_yaml_models.py` — parse-only validation (`7f60248e`) |
| Scheduler config без schema | `AbstractEngineConfig` (Pydantic) + JSON loader (`3735e9fa`) |

## Evidence (commits)

| Commit | Date | Что доказывает |
| --- | --- | --- |
| `3735e9fa` | 2026-01-31 | `AbstractEngineConfig` + toy stub |
| `7a44d766` | 2026-01-31 | Pydantic-first compile/run requests |
| `f5cc9592` | 2026-02-02 | Type safety refactor |
| `f6515298` | 2026-02-02 | Loop params Pydantic |
| `7f60248e` | 2026-02-01 | tt-llk fused YAML models |
| `1292685d` | 2026-02-05 | Recovery branch tip |

Локальный demo (без устройства): `pytest test/python/test_pydantic_validation.py -q` — см. hub `docs/publishing/DEMO_pydantic_ru.md`.
