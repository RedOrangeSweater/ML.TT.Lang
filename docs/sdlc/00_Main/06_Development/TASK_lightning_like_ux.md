# TASK: Lightning-like UX (Trainer, Module, Callbacks, Logger)

- **Статус**: Выполнен (MVP).
- **Контекст**: [18_ideal_ux_and_layer_responsibilities.md](../00_Ideas/18_ideal_ux_and_layer_responsibilities.md), план Lightning-like UX.

## Цель

Сделать пользовательский код в виде «модуль + Trainer»; инфраструктура берёт на себя девайс/конвертацию тензоров, режимы train/eval/predict/benchmark, компиляцию (warmup + cache), сбор метрик/профилей/артефактов.

## Scope (MVP)

- **Интерфейсы (Pydantic-first)**: `TrainerConfig`, `TrainableModule` (forward, training_step, validation_step, configure_optimizers — опционально/заглушки), `Callback` (on_fit_start, on_train_batch_end, …), `Logger` (jsonl + stdout).
- **Trainer**: `fit()` с двумя режимами — inference/benchmark loop (прогон батчей, latency/throughput/profiling) и scheduler-training loop (обёртка над SchedulerPlacementEnv, reward-based). Этапы: setup → compile_warmup → run loop → teardown.
- **Сокращение boilerplate**: `DeviceManager` (open/close, контекстный менеджер), `TensorAdapter` (torch↔ttnn, layout/memory по умолчанию); `ProgramModule` с опциональным `tensor_adapter` и методом `compile()` отдельно от `__call__()`; `Sequential.compile()` для warmup.
- **Observability**: опция `enable_profiling` в Trainer (установка TTLANG_AUTO_PROFILE), экспорт summary в `output_dir/summary.json`; structured metrics через Logger (jsonl при использовании JsonlLogger).

## Definition of Done

- [x] Интерфейсы TrainerConfig, TrainableModule, Callback, Logger зафиксированы; Trainer MVP с fit(), benchmark и scheduler_env режимами.
- [x] DeviceManager, TensorAdapter; ProgramModule и Sequential с опциональным tensor_adapter и compile().
- [x] Профилирование: config.enable_profiling → TTLANG_AUTO_PROFILE; запись summary.json в output_dir.
- [x] Примеры: `examples/nn_trainer_benchmark.py` (compile-only OK), `examples/nn_trainer_scheduler_env.py` (step/reward без tt-device).
- [x] Pytest: lifecycle hooks (on_fit_start, on_compile_start/end, on_train_batch_*, on_epoch_*), scheduler_env summary.
- [x] SDLC TASK (этот файл), раздел в 00_Quickstart «Lightning-like workflow», run configs: TTL: Trainer (benchmark), TTL: Trainer (scheduler env).

## Тест-план

- **test_nn_trainer.py**: `test_on_fit_start_end_and_compile_hooks`, `test_on_epoch_and_batch_hooks`, `test_scheduler_env_loop_summary`.
- **examples/nn_trainer_benchmark.py**: запуск с TTLANG_COMPILE_ONLY=1 (compile-only).
- **examples/nn_trainer_scheduler_env.py**: запуск без устройства (mock env).

## Затрагиваемые файлы

- `python/ttl/nn/`: config.py, callbacks.py, logger.py, module.py, sequential.py, trainer.py, device.py, __init__.py
- Примеры: examples/nn_trainer_benchmark.py, examples/nn_trainer_scheduler_env.py
- Тесты: test/python/test_nn_trainer.py
- Документация: docs/sdlc/00_Main/00_Status/00_Quickstart.md, .vscode/launch.json, .vscode/tasks.json

## Ссылки

- Идеальный UX и разделение слоёв: [18_ideal_ux_and_layer_responsibilities.md](../00_Ideas/18_ideal_ux_and_layer_responsibilities.md).
- Поток данных и модули: [20_IdealDataFlowAndModuleStructure.md](../02_Architecture/20_IdealDataFlowAndModuleStructure.md).
