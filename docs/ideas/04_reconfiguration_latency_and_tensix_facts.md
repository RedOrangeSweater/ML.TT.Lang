---
title: "04: Латентность реконфигурации и факты о Tensix (Wormhole B0)"
status: draft
audience: разработчики
---

# 04: Латентность реконфигурации и факты о Tensix (Wormhole B0)

Эта заметка собирает в одном месте две темы:

- **Авторитетные, цитируемые факты** о вычислительных ресурсах Tensix (Wormhole B0), чтобы работа не опиралась на предположения.
- **Воспроизводимый подход к измерению** “как быстро новый kernel становится исполнимым”, с разделением на host-side стадии компиляции и device-side исполнение.

## 1. Что такое “латентность реконфигурации”

В этом контексте “латентность реконфигурации” — это время от момента, когда “доступен новый исходник/IR”, до момента, когда “kernel может запускаться на устройстве”, разделенное на:

- **Host-only стадии (compiler/tooling)**:
  - Python DSL -> построение MLIR модуля;
  - MLIR pass pipeline (TTL -> TTKernel -> EmitC);
  - EmitC/TTKernel -> сгенерированные C++ kernel sources;
  - TT-Metal compile path (JIT) и взаимодействие с кэшами.
- **Device-facing стадии (runtime)**:
  - построение program и загрузка бинарей (если применимо);
  - первый enqueue / первый запуск (cold start);
  - steady-state run (warm start, кэш прогрет).

Полезны два режима измерения:

- **Cold start**: кэши очищены (in-memory kernel hash cache, program cache).
- **Warm start**: повторный вызов того же kernel без очистки кэшей.

## 2. “Source of truth” факты о Tensix (Wormhole B0)

### 2.1 Compute engines и dataflow (FPU/SFPU, unpacker/packer, registers)

Официальное описание TT-Metalium:

- “Compute Engines and Data Flow within Tensix”:
  - `https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html`

Ключевые тезисы из документа:

- Tensix имеет **matrix engine (FPU)** и **vector engine (SFPU)**, входной поток обеспечивается **unpacker**, выходной — **packer**.
- Вычислительные блоки опираются на наборы регистров:
  - **SrcA**, **SrcB** (внутренние регистры matrix engine),
  - **Dst** (набор destination регистров; используется также SFPU),
  - **LReg** (внутренние регистры SFPU).
- Compute kernel компилируется в **три отдельных бинарника**, каждый выполняется на отдельном RISC-V ядре (T0–T2) внутри Tensix, и требует явной синхронизации между компонентами.

### 2.2 Правила емкости Dst регистров (таблица “сколько tiles доступно”)

TT-Metalium фиксирует правила, определяющие, сколько **Dst tiles** доступно в зависимости от конфигурации host:

- `fp32_dest_acc_en` выбирает 16-bit vs 32-bit Dst storage.
- `dst_full_sync_en` управляет double-buffering:
  - `dst_full_sync_en=false` означает **double-buffering включен** (доступна только половина Dst за раз, чтобы перекрывать compute и pack).
  - `dst_full_sync_en=true` означает **double-buffering выключен** (доступен полный Dst, но перекрытие сокращается).

Ссылка на страницу TT-Metalium с таблицей количества Dst tiles; это и есть авторитетный источник по поведению:

- `https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html`

### 2.3 ISA ссылки (Wormhole B0)

Для ISA-level деталей:

- SFPU (Vector unit) на Wormhole B0:
  - `https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/VectorUnit.md`
- FPU (Matrix unit) на Wormhole B0:
  - `https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md`

## 3. Внутренние заметки репозитория (практика, не аппаратный spec)

Эти документы полезны для понимания “как ведет себя на практике” и вставленной синхронизации, но **не** являются hardware spec:

- `docs/ideas/03_tensix_sfpu_fpu_pipelining_dst.md` (pipeline overlap, `tile_regs_*`, hazard points)
- `docs/development/DST_Allocation.md` (DST pressure, `dst_idx` semantics)
- `docs/01_Architecture/02_LLD_CompilerPipeline.md` (точный порядок MLIR pipeline для TTL -> TTKernel -> EmitC)

Также важно: текущий TTNN interop код в `python/ttl/ttl_api.py` фиксирует:

- ровно **3 kernels** для interop (1 compute + 2 data movement);
- утверждение “each core has only 2 NOCs” как причину ограничения data-movement kernels.

Эта информация является **текущим ограничением интеграции tt-lang**, а не официальным утверждением о железе.

## 4. Измерение device-side времени (cycles)

Если цель включает “сколько времени выполняется на устройстве”, TT-Metalium предоставляет device-side profiler:

- Device Program Profiler:
  - `https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html`

Он измеряет длительности участков (в cycles since reset) на RISC-V ядрах через `DeviceZoneScopedN(...)` и формирует CSV (`profile_log_device.csv`) в `${TT_METAL_HOME}/generated/profiler/.logs/`.

## 5. Очистка кэшей для cold-start экспериментов (что безопасно, а что нет)

TTNN предоставляет очистку in-memory кэша компиляции kernels:

- `ttnn.device.ClearKernelCache()`

По документации TTNN, это очищает только in-memory hash lookup cache; disk-cached kernel binaries расположены в:

- `~/.cache/tt-metal-cache/<git_hash>/<build_id>/kernels/`

Для program cache у объекта device доступны:

- `device.clear_program_cache()`
- `device.disable_and_clear_program_cache()`

Эти методы позволяют получать воспроизводимые “cold” vs “warm” измерения без удаления файлов на диске.
