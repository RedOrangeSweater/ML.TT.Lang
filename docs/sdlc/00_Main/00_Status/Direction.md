# Направление (tt-lang)

- **Что это**: tt-lang — Python DSL для kernel-подобных программ (compute + data movement), компилируемых в MLIR и далее в TTKernel/EmitC; исполнение через ttnn или симулятор.
- **Куда движемся**: явные слои (Program / Graph / Compile / Runtime), Pydantic-типы и прокси на границе с ttnn; иерархические request-модели вместо длинных списков параметров; целевой data flow и модульная структура по слоям (см. [20_IdealDataFlowAndModuleStructure.md](../02_Architecture/20_IdealDataFlowAndModuleStructure.md)); документация и статус в согласии с кодом (SDLC-first, sdlc.audit).
- **Рефакторинг 2026**:
    - **Типизация**: переход на строгую типизацию (mypy, pyright), замена `object` и `Any` на `Protocols` и конкретные типы.
    - **Валидация**: перенос всей бизнес-валидации в Pydantic `model_validator`.
    - **Pythonic API**: упрощение декораторов и фасада `ttl_api.py`.
- **Где начать**: [00_Quickstart.md](00_Quickstart.md) — технический ввод; [02_Architecture/08_PythonDialectLayersAndDataFlow.md](../02_Architecture/08_PythonDialectLayersAndDataFlow.md) — реестр слоёв и трансформаций; [02_Architecture/20_IdealDataFlowAndModuleStructure.md](../02_Architecture/20_IdealDataFlowAndModuleStructure.md) — идеальный data flow и целевая структура модулей; [docs/README.md](../../../README.md) — навигация по документации.
