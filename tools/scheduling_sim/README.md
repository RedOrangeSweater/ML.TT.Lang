# scheduling_sim (salabim)

Небольшой дискретно-событийный симулятор, чтобы наглядно сравнить:

- **baseline**: без перекрытия (DMA -> wait -> compute)
- **ping-pong**: 2 слота (prefetch следующего слота параллельно compute)
- **RCW baseline**: reader/compute/writer последовательно фазами
- **RCW pipeline**: reader/compute/writer параллельно через страницы CB

Цель — получить “правдоподобную” картинку эффектов конвейеризации при явных допущениях по задержкам.

## Установка и окружение

Используется toolchain venv:

```bash
cd /home/kilka/Projects/ML/TT-NN/tt-lang
source build/env/activate

# Установить dev-зависимости (включая salabim)
python -m pip install -r dev-requirements.txt
```

## Запуск

```bash
source build/env/activate

# Запустить оба эксперимента
python -m tools.scheduling_sim.run --scenario all

# Только ping-pong
python -m tools.scheduling_sim.run --scenario pingpong

# Только reader/compute/writer
python -m tools.scheduling_sim.run --scenario rcw
```

## Параметры (что можно крутить)

- `--tiles N`: число тайлов
- `--dma-read-lat X`: задержка DMA read (условные циклы)
- `--dma-write-lat X`: задержка DMA write
- `--compute-lat X`: задержка compute на тайл
- `--cb-pages-in N`, `--cb-pages-out N`: емкость CB в страницах

Пример: “память медленная, compute быстрый” (pipeline должен помогать):

```bash
python -m tools.scheduling_sim.run --scenario all --tiles 64 --dma-read-lat 50 --compute-lat 5
```

Пример: “compute доминирует” (pipeline помогает меньше):

```bash
python -m tools.scheduling_sim.run --scenario all --tiles 64 --dma-read-lat 10 --compute-lat 50
```

## Модель и допущения (важно)

- Время — условные “циклы”.
- `NOC_READ`, `NOC_WRITE`, `COMPUTE` — ресурсы с capacity=1 (в текущей версии).
- Ping-pong моделирует TRID-специфический wait: compute ждёт только готовность нужного слота.
- RCW pipeline моделирует страницы CB через простой счётчик уровня (без адресации/лейаутов).

Это **не** ttsim и не цикл-точный симулятор железа. Это “микромодель”, чтобы сравнивать стратегии и дальше развивать её (в т.ч. для RL).
