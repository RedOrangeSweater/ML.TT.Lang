# TASK: Pydantic-first рефакторинг Python-слоя tt-lang

- **Статус**: Выполнен (первая итерация).
- **Контекст**: [08_PythonDialectLayersAndDataFlow.md](../02_Architecture/08_PythonDialectLayersAndDataFlow.md), правило «Validation only in Pydantic».

## Цель

Перевести точки входа и внутренние шаги компиляции/запуска на приём **Pydantic-моделей** вместо смеси `tuple/list/dict/Callable`. Вся валидация — в Pydantic; минимизировать разнородные параметры в функциях. Допускались ломающие изменения публичного API.

## Границы (breaking changes OK)

- Публичный API `ttl.run`: вместо `run(spec_or_program, *args, grid=..., options=..., **kwargs)` — один явный вход `run(req: RunRequest, engine_config_path=..., engine_config=...)`. Удобная сборка: `RunRequest.from_program(program, *args, grid=..., options=..., **kwargs)`.
- Внутренний API `_compile_kernel`: вместо набора параметров `(f, args, kwargs, request, thread_registry, engine_config)` — один объект `CompileKernelRequest`.
- Типы опций: `TTNNKernelCompileOptions.program_config` только `ProgramConfig | None` (без `dict`).
- Имена методов: убраны все `def to_*` в пользу `build_*` / `as_*` / свойств (например `to_ttnn` → `build_ttnn`, `to_ttcore` → свойство `ttcore_dtype`, `to_dict` → `as_dict`).

## Definition of Done (первая итерация)

- [x] Добавлена Pydantic request-модель для входа в compile pipeline: `CompileKernelRequest`; `_compile_kernel(req)` принимает один объект.
- [x] `ttl.run(req: RunRequest, ...)` — один явный вход; фабрика `RunRequest.from_program(...)`.
- [x] Proxy-слой: `build_ttnn()` вместо `to_ttnn()`, `ttcore_dtype` вместо `to_ttcore()`, `as_dict()` вместо `to_dict()` в ProgramConfig и scheduler; `TTNNKernelCompileOptions.program_config` только `ProgramConfig | None`.
- [x] Program layer: `build_compile_request`, `build_program_config`, `build_compile_options` вместо `to_compile_request`, `to_program_config`, `to_compile_options`.
- [x] Тесты и примеры обновлены под новую форму API; добавлен тест `test_compile_kernel_request_compile_only_path`.
- [x] SDLC TASK-док (этот файл) на русском: цель, границы, DoD, тест-план.

## Тест-план

- **test_dialect_layers.py**: `test_run_with_spec_accepts` (run(RunRequest(...)) → No threads found), `test_run_with_raw_callable_raises_not_implemented`, `test_run_with_program_no_grid_raises` (RunRequest.from_program без grid), `test_program_spec_build_compile_request`, `test_compile_kernel_request_compile_only_path` (_compile_kernel(CompileKernelRequest(...)) → No threads found).
- **test_scheduler_config.py**: вызовы `run(RunRequest(spec=..., args=..., kwargs={}), engine_config=...)`.
- **test_nn_api.py**: ProgramModule вызывает `run(RunRequest(...))` через nn/module.py — без изменений сигнатуры тестов.
- **nn_module_lit.py**, **examples/nn_single_module.py**: комментарии обновлены под новый API.

## Затрагиваемые файлы

- `python/ttl/ttl_api.py`, `python/ttl/program/__init__.py`, `python/ttl/compile/pipeline.py`
- `python/ttl/descriptor_options.py`, `python/ttl/dtype_utils.py`, `python/ttl/ttnn_proxy.py`
- `python/ttl/layouts.py`, `python/ttl/_src/ttl_ast.py`, `python/ttl/boundary/ttnn_types.py`
- `python/ttl/scheduler/op_graph.py`, `topology.py`, `scheduler_stub.py`
- `python/ttl/nn/module.py`
- Тесты и примеры: `test/python/test_dialect_layers.py`, `test_scheduler_config.py`, `nn_module_lit.py`, `examples/nn_single_module.py`

## Ссылки

- Правило валидации: только в Pydantic (`.cursor/rules/05_validation_pydantic.mdc`).
- Слои и поток данных: [08_PythonDialectLayersAndDataFlow.md](../02_Architecture/08_PythonDialectLayersAndDataFlow.md).
