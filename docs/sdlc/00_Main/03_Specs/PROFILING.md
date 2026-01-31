# Profiling (host Tracy + device profiler)

This document describes recommended profiling knobs for measuring tt-lang reconfiguration latency (cold vs warm) and correlating host-side stage boundaries with device-side execution.

## Host-side stage boundaries (Tracy)

tt-lang can emit **host-side Tracy zones** around major stages (Python build, thread compile, MLIR pipeline, program enqueue/run).

- **Enable**: `TTLANG_TRACY=1`
- **Source**: `python/ttl/profiling.py` (`span(...)` wrapper) and its use in `python/ttl/ttl_api.py`.

This is intended to be the primary “semantic timeline” for the host side.

## Device-side execution (cycles)

TT-Metalium provides a device program profiler that captures device-side zones and writes `profile_log_device.csv`.

- **Enable at runtime**: `TT_METAL_DEVICE_PROFILER=1`
- **Collection**: call `ttnn.device.ReadDeviceProfiler(device)` after runs.

See TT-Metalium docs:
- `https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html`

## Cold vs warm measurements

For “cold start” measurements, it is useful to clear:\n
- In-memory kernel compilation hash cache: `ttnn.device.ClearKernelCache()`\n
- Device program cache: `device.disable_and_clear_program_cache()` (fallback to `device.clear_program_cache()`)\n

## Practical harness

Use:

- `tools/profile_reconfiguration_latency.py`

Example:

```bash
source build/env/activate

# Host stage boundaries (Tracy)
python tools/profile_reconfiguration_latency.py --device-id 0 --tracy

# Add device-side profiler capture
python tools/profile_reconfiguration_latency.py --device-id 0 --tracy --device-profiler
```
