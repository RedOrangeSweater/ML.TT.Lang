# /code.validation.pydantic — валидация только в Pydantic

Убедиться, что проверки контрактов выполняются в Pydantic-моделях, а не в бизнес-логике.

## Действия

1. **Найти вызовы validate_*** в коде (ttl_api, kernel_runner, descriptor_options и т.п.) и места с явными проверками (raise ValueError/TypeError) в бизнес-логике.
2. **Предложить перенос** в соответствующие request/response модели (field_validator, model_validator).
3. **При реализации** следовать правилу 05_validation_pydantic (`.cursor/rules/05_validation_pydantic.mdc`).

## Когда вызывать

- При рефакторе валидаций.
- По запросу пользователя («валидации в Pydantic», «убери validate_ из бизнес-логики»).

## См. также

- docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md (§1, §4).
- `.cursor/rules/05_validation_pydantic.mdc`.
