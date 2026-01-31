# Scheduler Viz (doc 16)

Gymnasium environment and React UI for scheduler placement visualization and debugging. Target model is a **dynamic system** (AlphaStar-like): different placement strategies lead to different behavior over time (buffers full/empty, factories busy/idle); reward and strategy comparison should rely on **dynamics run**, not only static placement (doc 16, section 2).

## Phase 1: Placement (current)

Current env implements **placement phase only**: choose placement (node -> core); reward is a stub (minus sum of edge distances). No time evolution, no buffer/factory state.

- **State**: op graph + topology + current placement (sequential: place next node).
- **Action**: core index (place next node on that core).
- **Reward**: stub = minus sum of Manhattan distances over graph edges. Target: reward from **dynamics run** (utilization, latency, bottlenecks) after connecting simulator (doc 15 or simplified discrete model).

### Setup

From repo root (recommended: use project venv and uv):

```bash
# Install scheduler-viz Python deps into project venv (from tt-lang root)
uv pip install -r apps/scheduler-viz/requirements.txt
```

Or from `apps/scheduler-viz`:

```bash
cd apps/scheduler-viz
pip install -r requirements.txt
```

Ensure the Python used by Run config **Scheduler Viz: Backend** is the same venv (e.g. select tt-lang `.venv` as interpreter).

### Export from tt-lang

From tt-lang repo (with build env activated):

```python
from ttl.scheduler import export_scheduler_input_to_json

thread_infos = [
    ("add_compute", "compute"),
    ("dm_read", "datamovement"),
    ("dm_write", "datamovement"),
]
export_scheduler_input_to_json(thread_infos, (1, 1), "scheduler_input.json")
```

Or use the JSON from `export_scheduler_input(thread_infos, grid)`.

### Run env

```python
from scheduler_env import SchedulerPlacementEnv

env = SchedulerPlacementEnv("scheduler_input.json")
obs, info = env.reset(seed=42)
while True:
    action = env.action_space.sample()  # or policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print("Reward:", reward)
print("Full state:", env.get_full_state())
```

## React UI (MVP)

Three views: vertical op graph, resource view (FPU/SFPU/NOC), HW topology "floor". Load JSON, step/run via backend.

### Run backend and UI

```bash
# Terminal 1: from apps/scheduler-viz
cd apps/scheduler-viz
pip install -r requirements.txt
uvicorn backend:app --reload --port 8000

# Terminal 2: from apps/scheduler-viz/ui
cd apps/scheduler-viz/ui
npm install
npm run dev
```

Open http://localhost:5174. Select an **example** (toy_program_add, toy_program_broadcast, toy_program_multicore_auto) or load a custom scheduler JSON (optional). Then Step or Run. Examples are generated from tt-lang programs; reference examples match old-style and program-style programs (same grid and thread structure). Equivalence is tested in `test/python/test_scheduler_e2e.py::test_old_style_and_program_style_add_same_structure` (doc 16).

Or use Run configs: **Scheduler Viz: Backend**, **Scheduler Viz: UI** (or compound **Scheduler Viz: Backend + UI**) from `.vscode/launch.json`.

### Phase 2: Dynamics (planned)

After placement (or per step), run **dynamics simulator** (doc 15 salabim or simplified discrete model: queues, edge delays, block occupancy). Return trajectory or aggregated metrics; reward from metrics (utilization, minus latency, bottleneck penalty). UI: display **dynamics** (buffer fill, factory occupancy over time) and compare two plans by metrics after run, not only by static placement map.

### Simulator/metrics integration (stub)

Reward is computed by `_reward_stub` (minus sum of edge distances). To plug tt-sim or the salabim emulator (doc 15), pass a `reward_fn(graph, topology, placement) -> float` into `SchedulerPlacementEnv(..., reward_fn=...)`, or implement `reward_from_simulator_stub` in `scheduler_env.py` to call the simulator and return latency/throughput-based reward. Target: reward from **dynamics run** (trajectory of buffer/factory state), not static placement cost.
