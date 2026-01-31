# Направление (tt-lang)

- **Что это**: tt-lang — Python DSL для kernel-подобных программ (compute + data movement), компилируемых в MLIR и далее в TTKernel/EmitC; исполнение через ttnn или симулятор.
- **Куда движемся**: явные слои (Program / Graph / Compile / Runtime), Pydantic-типы и прокси на границе с ttnn; иерархические request-модели вместо длинных списков параметров; документация и статус в согласии с кодом (SDLC-first, sdlc.audit).
- **Где начать**: [00_Quickstart.md](00_Quickstart.md) — технический ввод; [02_Architecture/08_PythonDialectLayersAndDataFlow.md](../02_Architecture/08_PythonDialectLayersAndDataFlow.md) — реестр слоёв и трансформаций; [docs/README.md](../../../README.md) — навигация по документации.
