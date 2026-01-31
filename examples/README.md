# Examples

## Toy examples (Phase 1.2)

Toy examples use the **`@ttl.program`** decorator. One program may compile to one or more device kernels (Reader, Compute, Writer, etc.); under the hood the current fixed scheme (e.g. 1 compute + 2 datamovement) is used until the scheduler is extended.

| Example | Description |
|--------|-------------|
| [toy_program_add.py](toy_program_add.py) | Simple elementwise add, single core, `grid=(1, 1)`. |
| [toy_program_broadcast.py](toy_program_broadcast.py) | Single-core broadcast: `bcast(c) + (a * b)`, `grid=(1, 1)`. |
| [toy_program_multicore_auto.py](toy_program_multicore_auto.py) | Multicore elementwise `a * b + c`, `grid="auto"`. |

Run from repo root (with tt-lang env active). With a device: execution on hardware; without: compile-only (output unchanged).
