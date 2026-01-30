# 03: Сделать код MLIR-пайплайна читаемым как Sequential (пайплайны TTL)

## Проблема

В `tt-lang/lib/Dialect/TTL/Pipelines/TTLPipelines.cpp` бизнес-цель проста:

- понизить TTL -> TTKernel
- обеспечить порядок жизненного цикла DST
- выполнить очистку (canonicalize + CSE)
- опционально понизить до EmitC и сделать финальные очистки

Однако реализация читается как повторение механики:

- многократный `pm.addPass(createFoo())`
- условные блоки добавляют еще повторов (`if (options.lowerToEmitC) { ... }`)
- фабрики проходов в основном свободные функции с префиксом `create`, из-за чего код выглядит шумно (`create... create... create...`)

Справка (текущий код):

```cpp
pm.addPass(createTTLConvertTTLToCompute());
pm.addPass(createTTLAssignDST());
pm.addPass(createTTLInsertTileRegsSync());
pm.addPass(createTTLLowerToLoops());
pm.addPass(createTTLAnnotateCBAssociations());
pm.addPass(createTTLConvertTTLToTTKernel());
pm.addPass(createCanonicalizerPass());
pm.addPass(createCSEPass());
if (options.lowerToEmitC) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(::mlir::tt::createConvertTTKernelToEmitC());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mlir::emitc::createFormExpressionsPass());
}
```

## Почему это «нормально» в MLIR (но все еще можно улучшить)

В MLIR/LLVM построение проходов обычно представлено свободными фабричными функциями вроде
`createCanonicalizerPass()` по нескольким причинам:

- **Границы линковки и зависимостей**: библиотеки проходов могут линковаться независимо.
- **Интеграция с TableGen**: многие проходы генерируются и экспортируются как `create*Pass()`.
- **Композиция**: `OpPassManager` ожидает `std::unique_ptr<Pass>`, фабрики — самый чистый способ их создавать.

Стиль со свободными функциями является идиоматичным. Проблема не в «неправильности», а в
**читабельности на месте вызова**.

## Предлагаемое решение A: небольшой helper «строителя пайплайна» (вариадик + условные блоки)

Самый простой способ получить ощущение PyTorch `nn.Sequential` в C++:

- сделать маленькую обертку, вызывающую `pm.addPass(factory())`
- добавить вариадик-хелпер `addPasses(pm, ...)` (fold expression)
- оставить условные блоки обычными C++ `if (...) { ... }`

### Набросок helper-ов

```cpp
namespace mlir::tt::ttl::pipeline {

template <typename Factory>
void add(mlir::OpPassManager &pm, Factory &&factory) {
  pm.addPass(factory());
}

template <typename... Factories>
void addPasses(mlir::OpPassManager &pm, Factories &&...factories) {
  (add(pm, std::forward<Factories>(factories)), ...);
}

} // namespace mlir::tt::ttl::pipeline
```

### Пример «после» (читается как бизнес-логика)

```cpp
namespace p = mlir::tt::ttl::pipeline;

void createTTLToTTKernelPipeline(OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options) {
  p::addPasses(
      pm,
      createTTLConvertTTLToCompute,
      createTTLAssignDST,
      createTTLInsertTileRegsSync,
      createTTLLowerToLoops,
      createTTLAnnotateCBAssociations,
      createTTLConvertTTLToTTKernel,
      createCanonicalizerPass,
      createCSEPass);

  if (options.lowerToEmitC) {
    p::addPasses(
        pm,
        createLowerAffinePass,
        ::mlir::tt::createConvertTTKernelToEmitC,
        createCanonicalizerPass,
        mlir::emitc::createFormExpressionsPass);
  }
}
```

### Заметки / крайние случаи

- **Проходы с опциями**: передавать лямбду, чтобы это по-прежнему выглядело как «фабрика».

```cpp
p::add(pm, [&] { return ttcore::createTTCoreRegisterDevicePass(registerDeviceOptions); });
```

- **Проходы с перегруженными именами**: лямбда устраняет неоднозначность.
- **Итерация на этапе компиляции**: fold expression разворачивается на этапе компиляции; накладные расходы рантайма не меняются.

## Предлагаемое решение B: помощники фаз/групп (лучше для «бизнес-намерения»)

Если у пайплайна есть смысловые фазы, лучше сделать их явными функциями (внутри также использовать `addPasses`).

Важно: **не выделять крошечные группы из 2-4 проходов только ради названия**. Для очень небольших групп
хелперы часто скрывают больше, чем показывают. Фазовые хелперы полезны, когда верно хотя бы одно:

- группа **длинная** (условно 5+ проходов), и инлайн-лист визуально шумный
- группа **переиспользуется** в нескольких пайплайнах
- у группы есть **неочевидный инвариант**, который стоит централизовать (ограничения порядка, «должны идти подряд» и т.д.)

```cpp
namespace mlir::tt::ttl::pipeline {

inline void addDstLifecycle(OpPassManager &pm) {
  addPasses(pm, createTTLAssignDST, createTTLInsertTileRegsSync);
}

inline void addCleanup(OpPassManager &pm) {
  addPasses(pm, createCanonicalizerPass, createCSEPass);
}

inline void addEmitCBackend(OpPassManager &pm) {
  addPasses(pm,
            createLowerAffinePass,
            ::mlir::tt::createConvertTTKernelToEmitC,
            createCanonicalizerPass,
            mlir::emitc::createFormExpressionsPass);
}

} // namespace mlir::tt::ttl::pipeline
```

Тогда пайплайн становится почти самодокументируемым:

```cpp
namespace p = mlir::tt::ttl::pipeline;

void createTTLToTTKernelPipeline(OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options) {
  p::addPasses(pm,
               createTTLConvertTTLToCompute,
               p::addDstLifecycle, // хороший пример: строгий порядок + названное намерение
               createTTLLowerToLoops,
               createTTLAnnotateCBAssociations,
               createTTLConvertTTLToTTKernel,
               createCanonicalizerPass, // небольшая очистка: инлайн яснее, чем helper
               createCSEPass);
  if (options.lowerToEmitC) {
    // Небольшая backend-группа: инлайн-список читается нормально.
    p::addPasses(pm,
                 createLowerAffinePass,
                 ::mlir::tt::createConvertTTKernelToEmitC,
                 createCanonicalizerPass,
                 mlir::emitc::createFormExpressionsPass);
  }
}
```

На практике для целых фаз проще всего работает обычный `if`, и он читается лучше всего.

### Использование фазовых helper-ов внутри `addPasses(...)`

Если требуется, чтобы тело пайплайна было *единым линейным списком* (без отдельных вызовов `p::addCleanup(pm)`),
`addPasses` можно расширить для **двух типов элементов**:

- **Фабрика прохода**: вызывается с сигнатурой `() -> std::unique_ptr<mlir::Pass>`
- **Фазовый helper**: вызывается с сигнатурой `(mlir::OpPassManager &) -> void` (добавляет несколько проходов)

Минимальный helper, поддерживающий оба случая:

```cpp
namespace mlir::tt::ttl::pipeline {

template <typename Entry>
void addOne(mlir::OpPassManager &pm, Entry &&entry) {
  if constexpr (std::is_invocable_r_v<std::unique_ptr<mlir::Pass>, Entry>) {
    pm.addPass(entry());
  } else if constexpr (std::is_invocable_r_v<void, Entry, mlir::OpPassManager &>) {
    entry(pm);
  } else {
    static_assert(sizeof(Entry) == 0, "Entry must be a Pass factory or a group-adder");
  }
}

template <typename... Entries>
void addPasses(mlir::OpPassManager &pm, Entries &&...entries) {
  (addOne(pm, std::forward<Entries>(entries)), ...);
}

} // namespace mlir::tt::ttl::pipeline
```

Теперь пайплайн можно записать в виде единого списка:

```cpp
namespace p = mlir::tt::ttl::pipeline;

void createTTLToTTKernelPipeline(OpPassManager &pm,
                                 const TTLToTTKernelPipelineOptions &options) {
  p::addPasses(
      pm,
      createTTLConvertTTLToCompute,
      p::addDstLifecycle, // фазовый helper: разворачивается в createTTLAssignDST + createTTLInsertTileRegsSync
      createTTLLowerToLoops,
      createTTLAnnotateCBAssociations,
      createTTLConvertTTLToTTKernel,
      p::addCleanup);     // фазовый helper: разворачивается в canonicalizer + CSE

  if (options.lowerToEmitC) {
    p::addEmitCBackend(pm);
  }
}
```

Так сохраняется «Sequential»-читабельность, без лишних концепций, кроме `addPasses` и обычного `if`.

#### Что такое «фазовый helper» на практике?

Это обычная C++ функция, добавляющая *группу* проходов, чтобы один раз назвать намерение.
Например, в этом документе:

- `addCleanup(pm)` означает:

```cpp
pm.addPass(createCanonicalizerPass());
pm.addPass(createCSEPass());
```

- `addEmitCBackend(pm)` означает:

```cpp
pm.addPass(createLowerAffinePass());
pm.addPass(::mlir::tt::createConvertTTKernelToEmitC());
pm.addPass(createCanonicalizerPass());
pm.addPass(mlir::emitc::createFormExpressionsPass());
```

## Именование: что делать с повторением `create*`?

### Ключевая мысль: не стоит слишком сильно бороться с конвенциями MLIR

Переименование фабрик проходов, чтобы убрать `create`, обычно не оправдано, потому что:

- это часть идиоматичного API MLIR
- многие фабрики генерируются или предоставляются извне
- префикс `create` сигнализирует «создает новый Pass»

Вместо этого стоит улучшать читабельность на *месте использования*.

### Вариант 1: локальный алиас пространства имен «pass factories»

Можно создать локальное пространство имен с импортированными символами, чтобы вызовы были единообразны:

```cpp
namespace pass {
using ::mlir::createCanonicalizerPass;
using ::mlir::createCSEPass;
using ::mlir::createLowerAffinePass;
using ::mlir::emitc::createFormExpressionsPass;
using ::mlir::tt::createConvertTTKernelToEmitC;

using ::mlir::tt::ttl::createTTLAnnotateCBAssociations;
using ::mlir::tt::ttl::createTTLAssignDST;
using ::mlir::tt::ttl::createTTLConvertTTLToCompute;
using ::mlir::tt::ttl::createTTLConvertTTLToTTKernel;
using ::mlir::tt::ttl::createTTLLowerToLoops;
using ::mlir::tt::ttl::createTTLInsertTileRegsSync;
} // namespace pass
```

Тогда вызовы builder-а остаются единообразными и хорошо ищутся:

```cpp
p::addPasses(pm, pass::createTTLConvertTTLToCompute, pass::createTTLAssignDST, pass::createCanonicalizerPass);
```

Это **не** переименовывает ничего глобально; это локальная помощь для читабельности.

### Вариант 1b: алиасы `create::TTL...` (убрать повторяющийся `create` на месте вызова)

Если цель — читать пайплайны как `create::TTLConvertTTLToCompute` (без постоянного повторения `create`),
можно определить локальные алиасы, которые *указывают* на существующие фабричные функции. Это не требует
оберток или переименований.

```cpp
namespace create {
// TTL
inline constexpr auto TTLConvertTTLToCompute = ::mlir::tt::ttl::createTTLConvertTTLToCompute;
inline constexpr auto TTLAssignDST = ::mlir::tt::ttl::createTTLAssignDST;
inline constexpr auto TTLInsertTileRegsSync = ::mlir::tt::ttl::createTTLInsertTileRegsSync;
inline constexpr auto TTLLowerToLoops = ::mlir::tt::ttl::createTTLLowerToLoops;
inline constexpr auto TTLAnnotateCBAssociations = ::mlir::tt::ttl::createTTLAnnotateCBAssociations;
inline constexpr auto TTLConvertTTLToTTKernel = ::mlir::tt::ttl::createTTLConvertTTLToTTKernel;

// Generic MLIR
inline constexpr auto Canonicalizer = ::mlir::createCanonicalizerPass;
inline constexpr auto CSE = ::mlir::createCSEPass;
inline constexpr auto LowerAffine = ::mlir::createLowerAffinePass;

// TT / EmitC
inline constexpr auto TTKernelToEmitC = ::mlir::tt::createConvertTTKernelToEmitC;
inline constexpr auto EmitCFormExpressions = ::mlir::emitc::createFormExpressionsPass;
} // namespace create
```

Тогда список пайплайна выглядит так:

```cpp
p::addPasses(pm,
             create::TTLConvertTTLToCompute,
             p::addDstLifecycle,
             create::TTLLowerToLoops,
             create::TTLAnnotateCBAssociations,
             create::TTLConvertTTLToTTKernel,
             p::addCleanup);
```

### Вариант 2: скрыть `create*` за «семантическими» фазовыми функциями

Обычно это лучший вариант для больших пайплайнов: код перестает заботиться о названиях фабрик и читается как процесс:

- `addDstLifecycle(pm)`
- `addCleanup(pm)`
- `addEmitCBackend(pm)`

## Рекомендация

- Для коротких пайплайнов (как текущий `createTTLToTTKernelPipeline`): **Решение A** (вариадик `addPasses` + обычный `if`) достаточно.
- Для пайплайнов, которые разрастаются до «100 addPass вызовов»: стоит комбинировать **Решение A + B**, и тело пайплайна превращается в высокоуровневый рецепт.

Это сохраняет конвенции MLIR, улучшает повседневную эргономику для инженеров и делает будущие правки безопаснее
(фазы явно выделены, ограничения порядка остаются сгруппированными).
