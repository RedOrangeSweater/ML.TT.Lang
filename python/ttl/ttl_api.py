# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Main API for the TTL dialect Python DSL.

Ideal UX (no constraints): ttl.run(add_kernel, lhs, rhs, out, grid=(2,2))
or ttl.run(Spec(program=..., grid=...), *tensors). User thinks only "what to compute"
and "run params"; framework maps to KernelCompileRequest; engine does compile + run.
See docs/sdlc/00_Main/00_Ideas/18_ideal_ux_and_layer_responsibilities.md.
"""

from __future__ import annotations

import ast
import functools
import inspect
import os
import random
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import CellType
from typing import Callable, Generator, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

try:
    import ttnn
except (ModuleNotFoundError, ImportError):
    ttnn = None

import ttl._mlir_libs._ttlang  # Register tt-lang passes
from pykernel._src.utils import _cleanup_source_code
from ttmlir.dialects import ttkernel
from ttmlir.ir import *
from ttmlir.passes import (
    get_ttkernel_arg_spec,
    get_ttkernel_names,
    ttkernel_to_cpp_by_name,
)
from ttmlir.passmanager import PassManager

from ._src.auto_profile import (
    build_cb_wait_to_dma_map,
    build_dma_producer_to_cb_map,
    get_line_mapper,
    is_auto_profile_enabled,
    load_cb_flow_graph,
    parse_device_profile_csv,
    print_profile_report,
)
from ._src.tensor_registry import (
    get_tensor_global_index,
    get_tensor_source,
    register_tensor_name,
    register_tensor_source,
)
from ._src.ttl_ast import TTLCompilerConfig, TTLGenericCompiler
from .circular_buffer import CircularBuffer, get_cb_count
from .constants import SUPPORTED_MEMORY_SPACES
from .diagnostics import (
    TTLangCompileError,
    find_variable_assignment,
    format_mlir_error,
    format_python_error,
)
from .dtype_utils import (
    TensorDtype,
    is_ttnn_tensor,
    tile_bytes_from_dtype,
)
from .kernel_runner import (
    KernelSpec,
    RunKernelRequest,
    run_kernel_on_device,
)
from .operators import CopyTransferHandler, TensorBlock, copy
from .program import (
    KernelCompileRequest,
    ProgramOptions,
    ProgramSpec,
    _resolve_grid,
)
from .descriptor_options import (
    CoreRangeSetOptions,
    KernelWriteRequest,
    NocConfigOptions,
    ProgramConfig,
    ProgramRunConfig,
    ReaderConfigOptions,
    ThreadConfigBuildRequest,
    TTNNKernelCompileOptions,
    TTNNKernelCompileRequest,
)
from .settings import get_settings
from .ttl_utils import get_thread_type_string
from .verbose_context import verbose_compilation, verbose_print
from .config import HAS_TT_DEVICE


# For kernel body: TensorAccessor and dma (alias for copy) used in examples
# TensorAccessor(tensor) returns the tensor so it is captured; compiler treats
# subscript access (accessor[i,j]) as tensor accessor in MLIR (see ttl_ast).
def TensorAccessor(tensor):
    """Wrap a tensor for use as accessor in DM threads (e.g. accessor[i, j] in copy)."""
    return tensor


dma = copy  # Alias used in examples (DMA = copy for data movement)

class ThreadRegistry:
    """Registry for automatic collection of @compute and @datamovement threads."""

    __slots__ = ("_threads",)

    def __init__(self) -> None:
        self._threads: list[Callable[..., object]] = []

    def register(self, thread_fn: Callable[..., object]) -> None:
        """Register a thread function during decoration."""
        self._threads.append(thread_fn)

    def clear(self) -> None:
        """Clear the registry before kernel execution."""
        self._threads.clear()

    def get_and_clear(self) -> list[Callable[..., object]]:
        """Return all registered threads and clear the registry."""
        threads = list(self._threads)
        self._threads.clear()
        return threads


_thread_registry = ThreadRegistry()


def _get_tensor_cache_info(tensor) -> tuple:
    """Extract cache-relevant info from a tensor: (shape, dtype, memory_space, layout)."""
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    mem_config = tensor.memory_config()
    memory_space = (
        str(mem_config.buffer_type) if hasattr(mem_config, "buffer_type") else "unknown"
    )
    layout = str(tensor.layout) if hasattr(tensor, "layout") else "unknown"
    return (shape, dtype, memory_space, layout)


def _make_cache_key(
    args: tuple,
    fp32_dest_acc_en: bool | None,
    dst_full_sync_en: bool | None,
) -> tuple:
    """Create cache key from tensor properties and runtime compute config parameters."""
    tensor_key = tuple(
        _get_tensor_cache_info(arg) for arg in args if is_ttnn_tensor(arg)
    )
    return (tensor_key, fp32_dest_acc_en, dst_full_sync_en)


def _should_execute() -> bool:
    """Check if kernel execution should proceed (not compile-only mode)."""
    return not get_settings().compile_only


def _run_profiling_pipeline(
    tensors: tuple,
    all_source_lines: dict[str, list[str]],
    thread_to_kernel: dict[str, str],
    kernel_line_offsets: dict[str, int] | None = None,
):
    """
    Read device profiler data and display profile report.

    Called after kernel execution when auto-profiling is enabled.

    Args:
        tensors: Tuple of tensor arguments passed to the kernel
        all_source_lines: Dict mapping kernel name to source lines
        thread_to_kernel: Dict mapping RISC thread name to kernel name
    """
    if not is_auto_profile_enabled():
        return

    if ttnn is None:
        print("[Auto-profile] ttnn not available, skipping profiling")
        return

    from pathlib import Path

    # Get device from first ttnn tensor
    device = None
    for tensor in tensors:
        if is_ttnn_tensor(tensor) and hasattr(tensor, "device"):
            device = tensor.device()
            break

    if device is None:
        print("[Auto-profile] No device found in tensors, skipping profiling")
        return

    # Read profiler data from device
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception as e:
        print(f"[Auto-profile] Failed to read device profiler: {e}")
        return

    # Find the profile CSV - default location is $TT_METAL_HOME/generated/profiler/.logs/
    settings = get_settings()
    if settings.profile_csv:
        csv_path = Path(settings.profile_csv)
    else:
        tt_metal_home = settings.tt_metal_home
        if not tt_metal_home:
            print("[Auto-profile] TT_METAL_HOME not set, cannot find profile CSV")
            return
        csv_path = (
            Path(tt_metal_home) / "generated/profiler/.logs/profile_log_device.csv"
        )

    if not csv_path.exists():
        print(f"[Auto-profile] Profile CSV not found at {csv_path}")
        print("[Auto-profile] Ensure TT_METAL_DEVICE_PROFILER=1 is set before running")
        return

    # Parse and display results
    line_mapper = get_line_mapper()

    # Load CB flow graph for DMA attribution
    cb_flow = load_cb_flow_graph(csv_path)
    cb_wait_to_dma = build_cb_wait_to_dma_map(cb_flow)
    dma_producer_to_cb = build_dma_producer_to_cb_map(cb_flow)

    try:
        results = parse_device_profile_csv(csv_path, line_mapper)
        if results:
            print_profile_report(
                results,
                all_source_lines,
                thread_to_kernel,
                line_mapper,
                cb_wait_to_dma,
                dma_producer_to_cb,
                kernel_line_offsets,
            )
        else:
            print("[Auto-profile] No signpost results found in profile CSV")
    except Exception as e:
        print(f"[Auto-profile] Failed to parse profile CSV: {e}")


def _detect_memory_space_from_tensor(tensor, default: str) -> str:
    """Detect memory space (L1/DRAM) from a ttnn tensor's buffer type."""
    mem_config = tensor.memory_config()
    if hasattr(mem_config, "buffer_type"):
        buffer_type_str = str(mem_config.buffer_type)
        if "L1" in buffer_type_str:
            return "L1"
        elif "DRAM" in buffer_type_str:
            return "DRAM"
    return default


def _is_interleaved_tensor(tensor) -> bool:
    """Check if a ttnn tensor has interleaved memory layout."""
    mem_config = tensor.memory_config()
    if hasattr(mem_config, "memory_layout"):
        return "INTERLEAVED" in str(mem_config.memory_layout)
    return False


def _has_float32_args(args) -> bool:
    """
    Check if any input tensor uses float32 dtype.

    Inspects the tensor arguments to detect float32. This is used to
    automatically enable fp32_dest_acc_en configuration for compute kernels.

    Args:
        args: List of tensor arguments (torch or ttnn)

    Returns:
        True if any tensor uses float32 dtype, False otherwise
    """
    try:
        for tensor in args:
            if tensor is None:
                continue

            # Check ttnn tensor
            if is_ttnn_tensor(tensor):
                tensor_dtype = tensor.dtype
                # ttnn.float32
                if (
                    hasattr(tensor_dtype, "name")
                    and "float32" in str(tensor_dtype.name).lower()
                ):
                    return True
                if "float32" in str(tensor_dtype).lower():
                    return True
            # Check torch tensor
            elif hasattr(tensor, "dtype"):
                import torch

                if tensor.dtype == torch.float32:
                    return True
    except (AttributeError, TypeError, ImportError):
        pass

    return False


def _get_source_line_offset(f) -> int:
    """Get the line offset to convert parsed AST line numbers to actual file lines."""
    try:
        raw_lines, start_lineno = inspect.getsourcelines(f)
        # Count only leading decorator lines (before the def)
        num_decorator_lines = 0
        for line in raw_lines:
            stripped = line.strip()
            if stripped.startswith("@"):
                num_decorator_lines += 1
            elif stripped.startswith("def ") or stripped.startswith("async def "):
                break
        return start_lineno + num_decorator_lines - 1
    except (TypeError, OSError):
        return 0


class CompilationSourceContext(BaseModel):
    """Pydantic context for a single kernel compilation: source file, lines, and debug flags.

    Built once from the decorated function so call sites do not manually pass
    _source_file, _source_lines, _line_offset, debug_locations through kwargs.
    """

    model_config = ConfigDict(frozen=True)

    source_file: str = "<unknown>"
    source_lines: list[str] = Field(default_factory=list)
    line_offset: int = 0
    debug_locations: bool = True
    verbose: bool = False
    source_code: str = ""

    @classmethod
    def from_function(cls, f, *, verbose: bool = False) -> "CompilationSourceContext":
        """Build context from a decorated function (captures file, source, line offset)."""
        try:
            source_file = inspect.getfile(f)
        except (TypeError, OSError):
            source_file = "<unknown>"
        source_code = _cleanup_source_code(f)
        source_lines = source_code.splitlines()
        line_offset = _get_source_line_offset(f)
        return cls(
            source_file=source_file,
            source_lines=source_lines,
            line_offset=line_offset,
            debug_locations=True,
            verbose=verbose,
            source_code=source_code,
        )


def _track_tensor_sources(f_params, args, source_file: str) -> None:
    """Track source locations for tensor arguments.

    Searches backwards from the kernel call site to find where each
    tensor variable was assigned, then registers that location.
    """
    if source_file == "<unknown>":
        return

    try:
        with open(source_file, "r") as sf:
            source_lines = sf.read().splitlines()
    except (IOError, OSError):
        return

    call_line = None
    for frame_info in inspect.stack():
        if frame_info.filename == source_file:
            call_line = frame_info.lineno
            break

    if not call_line:
        return

    for param_name, arg in zip(f_params, args):
        if not is_ttnn_tensor(arg):
            continue
        assign_line = find_variable_assignment(source_lines, param_name, call_line)
        if assign_line:
            register_tensor_source(arg, source_file, assign_line)


class CompiledTTNNKernel(BaseModel):
    """
    A compiled tt-lang kernel ready for execution via ttnn.generic_op.

    Caches compilation artifacts (kernel paths, CB descriptors) so the kernel
    can be executed multiple times with different tensors without recompiling.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kernel_paths: list[tuple[str, str]] = Field(
        ...,
        description="List of (path, thread_type) tuples for each kernel",
    )
    kernel_configs: list[object] = Field(
        ...,
        description="List of config descriptors matching kernel_paths",
    )
    kernel_arg_specs: list[object] = Field(
        ...,
        description="List of arg specs (rt_args list) for each kernel",
    )
    num_tensors: int = Field(..., ge=0, description="Number of input/output tensors")
    core_ranges: object = Field(..., description="CoreRangeSet for kernel execution")
    kernel_tensor_indices: list[list[int]] = Field(
        ...,
        description="List of global tensor indices used by each kernel",
    )
    cb_configs: list[object] = Field(
        default_factory=list,
        description="CircularBuffer configs per CB index",
    )
    program_hash: object | None = Field(
        default=None, description="Hash for tt-metal program cache"
    )
    source_lines: object | None = Field(
        default=None, description="Source lines (deprecated)"
    )
    all_source_lines: dict[str, object] = Field(
        default_factory=dict,
        description="Dict mapping kernel name to source lines",
    )
    thread_to_kernel: dict[str, str] = Field(
        default_factory=dict,
        description="Dict mapping RISC thread name to kernel name",
    )
    kernel_line_offsets: dict[str, object] = Field(
        default_factory=dict,
        description="Dict mapping kernel name to line offset",
    )
    program_config: dict[str, object] = Field(
        default_factory=dict,
        description="Grid, objective, placement, etc.",
    )
    thread_names: list[str] = Field(
        default_factory=list,
        description="Thread names in same order as kernel_paths (for scheduler export)",
    )

    @field_validator("cb_configs", "thread_names", mode="before")
    @classmethod
    def _none_to_list(cls, v: object) -> object:
        """Coerce None to [] for optional list fields (call site may pass None)."""
        return v if v is not None else []

    @field_validator(
        "all_source_lines",
        "thread_to_kernel",
        "kernel_line_offsets",
        "program_config",
        mode="before",
    )
    @classmethod
    def _none_to_dict(cls, v: object) -> object:
        """Coerce None to {} for optional dict fields (call site may pass None)."""
        return v if v is not None else {}

    def __call__(self, *args: object) -> object:
        """Execute the kernel with the given tensors."""
        if len(args) != self.num_tensors:
            raise ValueError(f"Expected {self.num_tensors} tensors, got {len(args)}")

        # Validate grid against device's compute grid.
        device = args[0].device()
        device_grid = device.compute_with_storage_grid_size()
        kernel_grid = self.core_ranges.bounding_box().grid_size()
        if kernel_grid.x > device_grid.x or kernel_grid.y > device_grid.y:
            raise ValueError(
                f"Kernel grid ({kernel_grid.x}, {kernel_grid.y}) exceeds device "
                f"compute grid ({device_grid.x}, {device_grid.y}). "
                f"Reduce grid size to fit within available cores."
            )

        # Build kernel specs from stored kernel info.
        kernel_specs = []
        for kernel_idx, (kernel_path, thread_type) in enumerate(self.kernel_paths):
            tensor_indices = self.kernel_tensor_indices[kernel_idx]
            config = self.kernel_configs[kernel_idx]
            spec = KernelSpec(
                path=kernel_path,
                thread_type=thread_type,
                tensor_indices=tensor_indices,
                config=config,
            )
            kernel_specs.append(spec)

        run_req = RunKernelRequest(
            kernel_specs=kernel_specs,
            tensors=list(args),
            cb_configs=self.cb_configs,
            core_ranges=self.core_ranges,
            program_hash=self.program_hash,
        )
        return run_kernel_on_device(run_req)

    def get_scheduler_input(self) -> dict:
        """
        Return scheduler input (op_graph, topology, plan) for use by scheduler-viz.

        Requires thread_names to have been set at construction (newer kernels).
        Grid is taken from program_config or derived from core_ranges.
        """
        from .scheduler import export_scheduler_input

        if not self.thread_names or len(self.thread_names) != len(self.kernel_paths):
            raise ValueError(
                "get_scheduler_input requires thread_names (same length as kernel_paths). "
                "Recompile the kernel to get scheduler export support."
            )
        grid = self.program_config.get("grid")
        if grid is None and self.core_ranges is not None:
            box = self.core_ranges.bounding_box()
            g = box.grid_size()
            grid = (g.x, g.y)
        if grid is None:
            raise ValueError(
                "Cannot derive grid for scheduler input (no program_config.grid or core_ranges)."
            )
        thread_infos = list(zip(self.thread_names, [ty for _, ty in self.kernel_paths]))
        return export_scheduler_input(thread_infos, grid, self.program_config)


def validate_ttnn_tensors(args: tuple) -> None:
    """Validate tensor types and TTNN tensor properties. Raises ValueError on invalid."""
    ttnn_count = sum(1 for arg in args if is_ttnn_tensor(arg))
    if ttnn_count > 0 and ttnn_count < len(args):
        raise ValueError(
            f"TTNN interop requires all tensors to be the same type. "
            f"Got {ttnn_count} TTNN tensors and {len(args) - ttnn_count} host tensors. "
            f"Mixed tensor types would generate extra bounce kernels."
        )
    for i, arg in enumerate(args):
        if not is_ttnn_tensor(arg):
            continue
        mem_space = _detect_memory_space_from_tensor(arg, "unknown")
        if mem_space not in ("L1", "DRAM"):
            raise ValueError(
                f"TTNN interop requires L1 or DRAM memory space, but tensor {i} is in {mem_space}."
            )
        if not _is_interleaved_tensor(arg):
            raise ValueError(
                f"TTNN interop requires interleaved tensors, but tensor {i} is not. "
                f"Use ttnn.DRAM_MEMORY_CONFIG or ttnn.L1_MEMORY_CONFIG for interleaved tensors."
            )
        if hasattr(arg, "layout") and "TILE" not in str(arg.layout):
            raise ValueError(
                f"TTNN interop requires tilized tensors, but tensor {i} has layout {arg.layout}. "
                f"Use ttnn.to_layout(tensor, ttnn.TILE_LAYOUT) to convert."
            )


def validate_kernel_count(kernel_info: list) -> None:
    """Validate kernel count (exactly 3: 1 compute + 2 data movement). Raises ValueError if not."""
    if len(kernel_info) != 3:
        compute_count = sum(1 for _, t in kernel_info if t == "compute")
        dm_count = sum(1 for _, t in kernel_info if t == "noc")
        raise ValueError(
            f"TTNN interop requires exactly 3 kernels (1 compute + 2 data movement), "
            f"got {len(kernel_info)} kernels ({compute_count} compute, {dm_count} data movement). "
            f"Each core has only 2 NOCs, so more than 2 DM kernels causes NOC conflicts."
        )


def _build_config_for_thread(request: ThreadConfigBuildRequest) -> tuple:
    """Build (config, thread_to_kernel_entries) from a single Pydantic request."""
    return request.build_config_and_entries()


@contextmanager
def tmp_kernel_dir() -> Generator[Path, None, None]:
    """
    Context manager: create a dedicated directory for kernel sources for this compilation.

    Yields the directory path. Does not delete on exit so CompiledTTNNKernel paths
    remain valid. Use for grouping kernel files under one dir instead of /tmp/{user}/.
    """
    user = get_settings().user
    base = Path(f"/tmp/{user}")
    base.mkdir(parents=True, exist_ok=True)
    dir_path = base / f"ttlang_kernels_{uuid.uuid4().hex[:12]}"
    dir_path.mkdir(parents=False, exist_ok=False)
    try:
        yield dir_path
    finally:
        pass  # Keep dir so kernel paths in CompiledTTNNKernel remain valid


def _write_kernel_to_tmp(req: KernelWriteRequest) -> str:
    """Write kernel source to req.base_dir or /tmp/{user} and return the file path."""
    import hashlib
    import os

    content_hash = hashlib.md5(req.source.encode()).hexdigest()[:8]
    if req.base_dir is not None:
        req.base_dir.mkdir(parents=True, exist_ok=True)
        path = req.base_dir / f"ttlang_kernel_{req.name}_{content_hash}.cpp"
    else:
        user = get_settings().user
        path = Path(f"/tmp/{user}/ttlang_kernel_{req.name}_{content_hash}.cpp")
        os.makedirs(f"/tmp/{user}", exist_ok=True)
    with path.open("w") as f:
        f.write(req.source)
    print(f"=== {req.name} kernel written to {path} ===")
    print(req.source)
    print("=" * 60)
    return str(path)


def _compile_ttnn_kernel(req: TTNNKernelCompileRequest):
    """
    Compile kernel to CompiledTTNNKernel for execution via ttnn.generic_op.

    Builds kernel paths, configs, and CB descriptors from compiled MLIR module.
    All inputs are carried by the single Pydantic request object.
    """
    opts = req.compile_options or TTNNKernelCompileOptions()
    kernel_info = get_ttkernel_names(req.module)
    validate_ttnn_tensors(req.args)
    validate_kernel_count(kernel_info)

    if opts.verbose:
        print("=" * 60)
        print("TTNN INTEROP: Compiling kernel")
        print("=" * 60)
        print(f"Found {len(kernel_info)} kernels:")
        for name, thread_type in kernel_info:
            print(f"  - {name} ({thread_type})")

    if ttnn is None:
        print("\nttnn not available - cannot compile for ttnn.generic_op")
        return None

    core_ranges = CoreRangeSetOptions(grid=req.grid).build_ttnn_core_range_set()
    if opts.verbose:
        print(f"\nCore range: {core_ranges}")

    kernel_paths = []
    kernel_configs = []
    kernel_arg_specs = []
    noc_kernel_idx = 0
    has_f32 = _has_float32_args(req.args)
    thread_to_kernel: dict[str, str] = {}

    with tmp_kernel_dir() as base_dir:
        for name, thread_type in kernel_info:
            cpp_source = ttkernel_to_cpp_by_name(req.module, name)
            kernel_path = _write_kernel_to_tmp(
                KernelWriteRequest(name=name, source=cpp_source, base_dir=base_dir)
            )
            kernel_paths.append((kernel_path, thread_type))

            thread_req = ThreadConfigBuildRequest(
                thread_type=thread_type,
                name=name,
                noc_kernel_idx=noc_kernel_idx,
                compute_opts=opts.compute_config_options(),
                has_f32=has_f32,
                verbose=opts.verbose,
            )
            config, entries = _build_config_for_thread(thread_req)
            kernel_configs.append(config)
            thread_to_kernel.update(entries)
            if thread_type == "noc":
                noc_kernel_idx += 1

            arg_spec = get_ttkernel_arg_spec(req.module, name)
            if arg_spec is not None:
                arg_spec = ttkernel.ir.ArgSpecAttr.maybe_downcast(arg_spec)
                kernel_arg_specs.append(arg_spec.rt_args if arg_spec else [])
            else:
                kernel_arg_specs.append([])

    thread_names = [name for name, _ in kernel_info]
    raw_cfg = opts.program_config
    if isinstance(raw_cfg, ProgramConfig):
        cfg = raw_cfg.to_dict()
    else:
        cfg = dict(raw_cfg or {})
    cfg.setdefault("grid", req.grid)
    compiled_kernel = CompiledTTNNKernel(
        kernel_paths=kernel_paths,
        kernel_configs=kernel_configs,
        kernel_arg_specs=kernel_arg_specs,
        num_tensors=len(req.args),
        core_ranges=core_ranges,
        kernel_tensor_indices=req.thread_tensor_indices,
        cb_configs=req.cb_configs,
        program_hash=req.program_hash,
        source_lines=req.source_lines,
        all_source_lines=req.all_source_lines,
        thread_to_kernel=thread_to_kernel,
        kernel_line_offsets=req.kernel_line_offsets,
        program_config=cfg,
        thread_names=thread_names,
    )

    if opts.verbose:
        print(f"\nCompiled kernel ready (compiled {len(kernel_paths)} threads)")
        print("=" * 60)

    return compiled_kernel


def _collect_captures(
    f: Callable,
) -> dict[str, int | CircularBuffer]:
    """
    Collect and convert captured variables from function closure.

    Args:
        f: Function with closure to inspect

    Returns:
        Dictionary mapping variable names to converted values

    Raises:
        TypeError: If closure contains unsupported variable types
    """
    if f.__closure__ is None:
        return {}

    def convert(name, val):
        if isinstance(val, int):
            return val
        elif is_ttnn_tensor(val):
            return val
        elif isinstance(val, CircularBuffer):
            return val
        else:
            # Allow torch.Tensor for compile-only path (e.g. scheduler export fixtures)
            try:
                import torch

                if isinstance(val, torch.Tensor):
                    return val
            except ImportError:
                pass
            raise TypeError(f"Unhandled capture for vars of type({type(val)})")

    return {
        n: convert(n, c.cell_contents)
        for n, c in zip(f.__code__.co_freevars, f.__closure__)
    }


class ThreadWrapperView(BaseModel):
    """Typed view of a decorated thread: wrapped callable and its closure.

    Built from a callable so getattr for __wrapped__ and __closure__ is done
    once in from_callable(); call sites work with this structure instead of
    repeated getattr.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    wrapped: Callable[..., object] | None = None
    closure: tuple[CellType, ...] | None = None

    @classmethod
    def from_callable(cls, thread_fn: Callable[..., object]) -> "ThreadWrapperView":
        wrapped = getattr(thread_fn, "__wrapped__", None)
        closure: tuple[CellType, ...] | None = (
            getattr(wrapped, "__closure__", None) if wrapped else None
        )
        if closure is not None:
            closure = tuple(closure)
        return cls(wrapped=wrapped, closure=closure)


def _collect_cb_configs(
    threads: list[Callable[..., object]],
) -> list[CircularBuffer | None]:
    """Extract CircularBuffer objects from thread closures, indexed by cb_index.

    Returns a list of CircularBuffer objects indexed by cb_index. Each CB has
    shape, buffer_factor, tensor (for dtype), and _cb_index attributes.
    """
    cb_configs_dict: dict[int, CircularBuffer] = {}
    for thread_fn in threads:
        view = ThreadWrapperView.from_callable(thread_fn)
        if not view.closure:
            continue
        for cell in view.closure:
            val = cell.cell_contents
            if isinstance(val, CircularBuffer):
                cb_configs_dict[val._cb_index] = val

    if not cb_configs_dict:
        return []
    max_idx = max(cb_configs_dict.keys())
    return [cb_configs_dict.get(i) for i in range(max_idx + 1)]


def _compile(
    kernel_type: str | None = None,
    verbose: bool = False,
) -> Callable[..., object]:
    """
    Internal decorator for compiling kernel threads.

    Args:
        kernel_type: Type of kernel ("compute" or "datamovement")
        verbose: Enable verbose compilation output

    Returns:
        Decorator function for kernel compilation
    """

    def _decorator(f):
        # Capture source file at decoration time
        try:
            source_file = inspect.getfile(f)
        except (TypeError, OSError):
            source_file = "<unknown>"

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            ctx = CompilationSourceContext.from_function(f, verbose=verbose)
            compiler_config = TTLCompilerConfig.model_validate(
                {
                    **kwargs,
                    "_source_file": ctx.source_file,
                    "_source_lines": ctx.source_lines,
                    "_line_offset": ctx.line_offset,
                    "debug_locations": ctx.debug_locations,
                    "_globals": f.__globals__,
                }
            )

            m = ast.parse(ctx.source_code)

            b = TTLGenericCompiler(
                f.__name__,
                kernel_type,
                _collect_captures(f),
                *args,
                config=compiler_config,
            )

            with verbose_compilation(ctx.verbose):
                verbose_print(ast.dump(m, indent=4) + "\n")
                b.visit(m)
                verbose_print(b.module)

            try:
                b.module.operation.verify()
            except Exception as e:
                formatted = format_mlir_error(str(e), ctx.source_lines, ctx.source_file)
                raise RuntimeError(formatted) from None

            return b

        _wrapper._decorator_name = kernel_type + "_thread"
        _wrapper._source_file = source_file
        # Register thread for automatic collection
        _thread_registry.register(_wrapper)
        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator


def compute(verbose: bool = False) -> Callable:
    """
    Decorator for compute thread functions.

    Compute threads execute on Tensix cores and perform mathematical operations.

    Args:
        verbose: Enable verbose compilation output

    Returns:
        Decorator for compute kernel compilation
    """
    return _compile(
        kernel_type="compute",
        verbose=verbose,
    )


def datamovement(verbose: bool = False) -> Callable:
    """
    Decorator for data movement thread functions.

    Data movement threads handle DMA operations between memory hierarchies.

    Args:
        verbose: Enable verbose compilation output

    Returns:
        Decorator for data movement kernel compilation
    """
    return _compile(
        kernel_type="datamovement",
        verbose=verbose,
    )


class Program:
    """
    Immutable container for kernel threads and their arguments.

    A Program encapsulates compute and data movement threads along with
    the arguments to be passed during execution. After construction, all
    fields should be treated as read-only.
    """

    def __init__(
        self,
        *threads,
        args=(),
        run_config: ProgramRunConfig | None = None,
        kwargs: dict[str, object] | None = None,
    ):
        self._threads = threads
        self._args = args
        if run_config is not None:
            self._kwargs = run_config.model_dump()
        else:
            self._kwargs = kwargs if kwargs is not None else {}

    @property
    def threads(self) -> tuple:
        return self._threads

    @property
    def args(self) -> tuple:
        return self._args

    @property
    def kwargs(self) -> dict[str, object]:
        return self._kwargs

    def __call__(self, *args, **kwargs):
        return Program(*self.threads, args=args, kwargs={**self.kwargs, **kwargs})


def _collect_source_info_from_threads(
    threads: list[TTLGenericCompiler],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    """Build all_source_files, all_source_lines, kernel_line_offsets from compiled threads."""
    all_source_files: dict[str, str] = {}
    all_source_lines: dict[str, list[str]] = {}
    kernel_line_offsets: dict[str, int] = {}
    for ct in threads:
        info = ct.source_info
        all_source_files[ct.name] = info.source_file
        all_source_lines[ct.name] = info.source_lines
        kernel_line_offsets[ct.name] = info.line_offset
    return all_source_files, all_source_lines, kernel_line_offsets


# -----------------------------------------------------------------------------
# Compile layer: KernelCompileRequest -> (threads, module) -> TTNNKernelCompileRequest
# -> CompiledTTNNKernel. One focus: spec to compilation artifacts.
# -----------------------------------------------------------------------------


def _compile_kernel(
    f: Callable,
    args: tuple,
    kwargs: dict,
    request: KernelCompileRequest,
) -> CompiledTTNNKernel | None:
    """
    Compile kernel function to MLIR and return CompiledTTNNKernel.

    Args:
        f: User kernel function
        args: Positional arguments for the kernel
        kwargs: Keyword arguments for the kernel
        request: KernelCompileRequest with grid, program_hash, indexing_maps, iterator_types,
            and nested options (ProgramOptions: num_outs, memory_space, tiled, fp32_dest_acc_en,
            dst_full_sync_en, objective, placement). See KernelCompileRequest and ProgramOptions.

    Returns:
        CompiledTTNNKernel ready for execution
    """
    grid = request.grid
    program_hash = request.program_hash
    num_outs = request.options.num_outs
    memory_space = request.options.memory_space
    tiled = request.options.tiled
    fp32_dest_acc_en = request.options.fp32_dest_acc_en
    dst_full_sync_en = request.options.dst_full_sync_en
    program_config = request.options.program_config_dict() or {}
    f_params = inspect.signature(f).parameters

    # Get kernel source location for error reporting
    try:
        kernel_source_file = inspect.getfile(f)
        kernel_line_offset = _get_source_line_offset(f)
    except (TypeError, OSError):
        kernel_source_file = "<unknown>"
        kernel_line_offset = 0

    has_ttnn_tensors = any(is_ttnn_tensor(arg) for arg in args)

    # For TTNN tensors, detect memory space from tensor's buffer type.
    # L1 tensors use simple NOC addressing, DRAM uses bank-aware addressing.
    # TODO: Check all tensors and handle mixed memory spaces.
    if has_ttnn_tensors:
        first_ttnn_tensor = next((arg for arg in args if is_ttnn_tensor(arg)), None)
        if first_ttnn_tensor is not None:
            memory_space = _detect_memory_space_from_tensor(
                first_ttnn_tensor, memory_space
            )
            print(f"[TTNN interop] Detected {memory_space} memory space")

    for idx, (param_name, arg) in enumerate(zip(f_params, args)):
        register_tensor_name(arg, param_name, index=idx)

    # For pretty error printing only:
    _track_tensor_sources(f_params, args, kernel_source_file)

    run_config = ProgramRunConfig(
        grid=list(grid),
        memory_space=memory_space,
        tiled=tiled,
        debug_locations=True,  # Always generate locations for error messages
    )
    run_config.inject_into_kwargs(kwargs, set(f_params))

    from .circular_buffer import _reset_cb_counter, CircularBuffer
    from .operators import _set_current_grid

    _reset_cb_counter()
    _set_current_grid(grid)

    _thread_registry.clear()
    f(*args, **kwargs)
    threads = _thread_registry.get_and_clear()

    if not threads:
        raise ValueError(
            "No threads found. Define at least one @ttl.compute() or "
            "@ttl.datamovement() function inside your kernel."
        )

    cb_configs = _collect_cb_configs(threads)

    program = Program(*threads, args=args, run_config=run_config)

    # Always generate source locations for error messages
    # TTLANG_DEBUG_LOCATIONS only controls whether locations are printed in MLIR output
    print_debug_locations = get_settings().debug_locations

    ctx = Context()
    loc = Location.unknown(ctx)
    with ctx, loc:
        compiled_threads: list[TTLGenericCompiler] = []
        # Track which global tensor indices each thread uses (for building common_runtime_args)
        thread_tensor_indices: list[list[int]] = []

        for compile_thread in program.threads:
            try:
                ct = compile_thread(*program.args, **program.kwargs)
            except TTLangCompileError as e:
                # Thread-level error with embedded source location - use it
                raise type(e)(e.format()) from None
            except (ValueError, TypeError) as e:
                # Kernel-level error (no embedded location) - use kernel decorator
                formatted = format_python_error(
                    e, kernel_source_file, kernel_line_offset
                )
                raise type(e)(formatted) from None
            compiled_threads.append(ct)
            thread_tensor_indices.append(ct._tensor_accessor_global_indices)

            # Set TensorAccessor indexing attributes for C++ lowering
            base_cta = get_cb_count()
            ct.func_entry.attributes["ttl.base_cta_index"] = IntegerAttr.get(
                IntegerType.get_signless(32, ctx), base_cta
            )
            crta_indices = ct._tensor_accessor_global_indices
            ct.func_entry.attributes["ttl.crta_indices"] = ArrayAttr.get(
                [
                    IntegerAttr.get(IntegerType.get_signless(32, ctx), idx)
                    for idx in crta_indices
                ],
                ctx,
            )

        all_source_files, all_source_lines, kernel_line_offsets = (
            _collect_source_info_from_threads(compiled_threads)
        )

        # Optional: build op graph and run scheduler stub (Phase 3-4)
        if get_settings().use_scheduler:
            from .scheduler import (
                build_op_graph_from_threads,
                build_topology_from_grid,
                schedule_stub,
            )

            thread_infos: list[tuple[str, str]] = [
                (ct.name, ct.kernel_type or "dm") for ct in compiled_threads
            ]
            op_graph = build_op_graph_from_threads(thread_infos)
            topology = build_topology_from_grid(grid)
            plan = schedule_stub(op_graph, topology, program_config)
            # Verify plan matches current grid (stub assigns all to (0,0))
            assert (
                plan.grid_cols == topology.grid_cols
                and plan.grid_rows == topology.grid_rows
            )
            for nid in op_graph.node_ids_in_order():
                assert plan.core_for(nid) == (0, 0), f"stub plan mismatch for {nid}"

        module = Module.create(loc)

        # Insert standalone thread functions directly into module
        with InsertionPoint(module.body):
            for ct in compiled_threads:
                ct.func_entry.operation.detach_from_parent()
                module.body.append(ct.func_entry)

        initial_mlir_path = get_settings().initial_mlir
        if initial_mlir_path:
            with open(initial_mlir_path, "w") as fd:
                module.operation.print(
                    file=fd,
                    enable_debug_info=print_debug_locations,
                    print_generic_op_form=False,
                )
            print(f"SAVED INITIAL TO {initial_mlir_path}")

        verify = True

        # fmt: off
        set_compute_config_pass = "func.func(ttl-set-compute-kernel-config)"
        config_options = []
        if fp32_dest_acc_en is not None:
            config_options.append(
                f"fp32-dest-acc-en={1 if fp32_dest_acc_en else 0}"
            )
        if dst_full_sync_en is not None:
            config_options.append(
                f"dst-full-sync-en={1 if dst_full_sync_en else 0}"
            )
        if config_options:
            set_compute_config_pass = (
                "func.func(ttl-set-compute-kernel-config{"
                + " ".join(config_options)
                + "})"
            )

        pipeline_passes = [
            "func.func(convert-ttl-to-compute)",
            set_compute_config_pass,
            "func.func(ttl-assign-dst)",
            "func.func(ttl-insert-tile-regs-sync)",
            "func.func(ttl-lower-to-loops)",
            "func.func(ttl-annotate-cb-associations)",
        ]

        # Add auto-profiling passes if enabled
        if is_auto_profile_enabled():
            st = get_settings()
            if st.profile_csv:
                cb_flow_json = str(Path(st.profile_csv).parent / "cb_flow_graph.json")
            else:
                tt_metal_home = st.tt_metal_home
                if not tt_metal_home:
                    raise ValueError("TTLANG_AUTO_PROFILE=1 requires TT_METAL_HOME or TTLANG_PROFILE_CSV to be set")
                cb_flow_json = f"{tt_metal_home}/generated/profiler/.logs/cb_flow_graph.json"
            pipeline_passes.append(f'ttl-dump-cb-flow-graph{{output="{cb_flow_json}"}}')

        pipeline_passes += [
            "convert-ttl-to-ttkernel",
        ]

        if is_auto_profile_enabled():
            pipeline_passes.append("ttl-lower-signpost-to-emitc")

        pipeline_passes += [
            "canonicalize",
            "cse",
            "lower-affine",
            "convert-ttkernel-to-emitc",
            "symbol-dce",
        ]

        if HAS_TT_DEVICE:
            pipeline_passes.insert(0, "ttcore-register-device")

        pipeline = ",".join(pipeline_passes)

        pipeline_str = f"builtin.module({pipeline})"
        # fmt: on
        pm = PassManager.parse(pipeline_str)
        pm.enable_verifier(verify)

        try:
            from ttmlir._mlir_libs._ttmlir import enable_pretty_stack_traces

            enable_pretty_stack_traces(pm._CAPIPtr)
        except Exception:
            # Pretty stack traces are optional, silently continue if unavailable
            pass

        if get_settings().verbose_passes:
            print("Running custom pipeline:", pm)
            ctx.enable_multithreading(False)
            pm.enable_ir_printing(
                print_after_all=True,
                print_before_all=True,
                print_after_failure=True,
                enable_debug_info=True,
            )

        # Run the pass manager with error handling for source-aware diagnostics
        try:
            pm.run(module.operation)
        except Exception as e:
            error_msg = str(e)
            # Try to format error with source context
            # Use the first thread's source as fallback
            source_lines = None
            source_file = None
            if all_source_lines:
                first_thread = next(iter(all_source_lines.keys()))
                source_lines = all_source_lines[first_thread]
                source_file = all_source_files.get(first_thread)
            formatted = format_mlir_error(error_msg, source_lines, source_file)
            raise RuntimeError(formatted) from None

        final_mlir_path = get_settings().final_mlir
        if final_mlir_path:
            with open(final_mlir_path, "w") as fd:
                module.operation.print(
                    file=fd,
                    enable_debug_info=print_debug_locations,
                    print_generic_op_form=False,
                )
            print(f"SAVED FINAL TO {final_mlir_path}")

        # Extract source lines for auto-profiling (use first thread's source)
        profile_source_lines = None
        if all_source_lines:
            first_thread = next(iter(all_source_lines.keys()))
            profile_source_lines = all_source_lines[first_thread]

        # Compile to CompiledTTNNKernel for ttnn.generic_op
        compile_req = TTNNKernelCompileRequest(
            module=module,
            args=args,
            grid=grid,
            num_outs=num_outs,
            thread_tensor_indices=thread_tensor_indices,
            cb_configs=cb_configs,
            program_hash=program_hash,
            compile_options=TTNNKernelCompileOptions(
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=dst_full_sync_en,
                program_config=program_config,
            ),
            source_lines=profile_source_lines,
            all_source_lines=all_source_lines,
            kernel_line_offsets=kernel_line_offsets,
        )
        compiled_kernel = _compile_ttnn_kernel(compile_req)
        return compiled_kernel


OBJECTIVE_VALUES = ("latency", "throughput", "balanced")
PLACEMENT_VALUES = ("auto", "manual")


# -----------------------------------------------------------------------------
# Runtime entry: ProgramSpec + args -> compile -> run_kernel_on_device (kernel_runner).
# Program layer types live in ttl.program; see kernel_runner for Runtime layer.
# -----------------------------------------------------------------------------

# Marker set on @ttl.program-decorated wrappers so run() can accept (program, *args, grid=...).
_TTL_PROGRAM_ATTR = "_ttl_program"


def run(
    spec_or_program: ProgramSpec | Callable[..., object],
    *args: object,
    grid: (tuple[int, ...] | list[int] | Callable[..., object]) | None = None,
    options: ProgramOptions | None = None,
    **kwargs: object,
) -> object | None:
    """
    Facade: compile and run kernel from a spec or program. Ideal UX entry point.

    Two forms:
    - run(spec, *args, **kwargs) — spec is ProgramSpec (program + grid + options).
    - run(program, *args, grid=..., options=..., **kwargs) — program is @ttl.program-decorated;
      grid is required; options default to ProgramOptions().

    Lambda / raw callable: passing a lambda (e.g. lambda lhs, rhs: lhs + rhs) is planned;
    inference from parameters and return type is not yet implemented. Use @ttl.program for now.
    See 20_IdealDataFlowAndModuleStructure.md (lambda + inference).
    """
    if isinstance(spec_or_program, ProgramSpec):
        spec = spec_or_program
    else:
        if not callable(spec_or_program):
            raise TypeError(
                "first argument must be ProgramSpec or @ttl.program-decorated callable"
            )
        program_fn = spec_or_program
        if not getattr(program_fn, _TTL_PROGRAM_ATTR, False):
            raise NotImplementedError(
                "run() with a raw callable (e.g. lambda) is not yet implemented: "
                "inference from parameters and return type is planned. "
                "Use @ttl.program to define the kernel and pass it to run(program, *args, grid=...). "
                "See docs/sdlc/00_Main/02_Architecture/20_IdealDataFlowAndModuleStructure.md "
                "(lambda + inference)."
            )
        if grid is None:
            raise ValueError(
                "grid= is required when passing a program callable; "
                "e.g. run(add_kernel, lhs, rhs, out, grid=(2, 2))"
            )
        opts = options if options is not None else ProgramOptions()
        spec = ProgramSpec(program=program_fn, grid=grid, options=opts)

    if spec.options.num_outs != 1:
        raise ValueError(f"num_outs must be 1, got {spec.options.num_outs}")
    cache_key = _make_cache_key(
        args,
        fp32_dest_acc_en=spec.options.fp32_dest_acc_en,
        dst_full_sync_en=spec.options.dst_full_sync_en,
    )
    program_hash = hash((id(spec.program), cache_key))
    request = spec.to_compile_request(args, kwargs, program_hash)
    compiled = _compile_kernel(spec.program, args, dict(kwargs), request)
    if compiled is None:
        return None
    if _should_execute():
        return compiled(*args)
    return None


def pykernel_gen(
    grid: (tuple[int, ...] | Callable[..., object]) | None = None,
    indexing_maps: list[Callable[..., object]] | None = None,
    iterator_types: list[str] | None = None,
    num_outs: int = 1,
    memory_space: str = "L1",
    tiled: bool = True,
    fp32_dest_acc_en: bool | None = None,
    dst_full_sync_en: bool | None = None,
    objective: str | None = None,
    placement: str | None = None,
) -> Callable:
    """
    Decorator for generating TTL kernels from Python functions.

    This decorator compiles Python functions into TTL dialect operations,
    handling thread compilation, stream creation, and pipeline execution.
    Kernels are compiled to C++ for execution via ttnn.generic_op.

    Args:
        grid: Grid dimensions as tuple (e.g., (2, 2)) or callable
        indexing_maps: List of lambda functions for indexing (optional)
        iterator_types: List of iterator types ("parallel", "reduction")
        num_outs: Number of output arguments
        memory_space: "L1" or "DRAM"
        tiled: Whether to use tiled layout
        fp32_dest_acc_en: Optional override for fp32_dest_acc_en
        dst_full_sync_en: Optional override for dst_full_sync_en
        objective: Optional policy "latency" | "throughput" | "balanced" (stored, not used yet)
        placement: Optional policy "auto" | "manual" (stored, not used yet)

    Returns:
        Decorated function that compiles and executes the kernel

    Raises:
        AssertionError: If required parameters are missing or invalid
    """
    if grid is None:
        raise ValueError("grid parameter is required")
    if iterator_types is not None and indexing_maps is None:
        raise ValueError("indexing_maps must be set when iterator_types is set")

    try:
        options = ProgramOptions(
            num_outs=num_outs,
            memory_space=memory_space,
            tiled=tiled,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            objective=objective,
            placement=placement,
        )
    except ValidationError as e:
        raise ValueError(str(e)) from e

    if options.num_outs != 1:
        raise ValueError(f"num_outs must be 1, got {options.num_outs}")

    if indexing_maps is None:
        indexing_maps = []

    if indexing_maps:
        for indexing_map in indexing_maps:
            num_dims = list(tuple(inspect.signature(indexing_map).parameters))
            if iterator_types is not None:
                if num_dims != len(iterator_types):
                    raise ValueError(
                        f"Number of dimensions ({num_dims}) must match iterator_types length ({len(iterator_types)})"
                    )

    if iterator_types is None:
        iterator_types = []

    def _decorator(f):
        # Per-kernel state: random ID and cache
        kernel_id = random.getrandbits(64)
        cache: dict[tuple[object, ...], CompiledTTNNKernel] = {}

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            resolved_grid = _resolve_grid(grid, args, kwargs)

            # Build cache key from tensor properties
            cache_key = _make_cache_key(
                args,
                fp32_dest_acc_en=options.fp32_dest_acc_en,
                dst_full_sync_en=options.dst_full_sync_en,
            )

            # Check cache for previously compiled kernel
            if cache_key in cache:
                compiled_kernel = cache[cache_key]
            else:
                # Compute program_hash for tt-metal cache
                program_hash = hash((kernel_id, cache_key))

                # Compile kernel
                compile_request = KernelCompileRequest(
                    grid=resolved_grid,
                    program_hash=program_hash,
                    indexing_maps=indexing_maps,
                    iterator_types=iterator_types,
                    options=options,
                )
                compiled_kernel = _compile_kernel(f, args, kwargs, compile_request)

                if compiled_kernel is not None:
                    cache[cache_key] = compiled_kernel

            # Expose last compiled kernel for scheduler export (e.g. get_scheduler_input)
            _wrapper._last_compiled_kernel = compiled_kernel

            # Execute (unless compile-only mode)
            if compiled_kernel is not None and _should_execute():
                result = compiled_kernel(*args)

                # Run auto-profiling after execution
                if is_auto_profile_enabled() and compiled_kernel.all_source_lines:
                    _run_profiling_pipeline(
                        args,
                        compiled_kernel.all_source_lines,
                        compiled_kernel.thread_to_kernel,
                        compiled_kernel.kernel_line_offsets,
                    )

                return result

        setattr(_wrapper, _TTL_PROGRAM_ATTR, True)
        return _wrapper

    return _decorator


# Alias for backward compatibility
kernel = pykernel_gen

# Preferred name: one program may compile to one or more device kernels (Reader/Compute/Writer etc.)
program = pykernel_gen


__all__ = [
    "pykernel_gen",
    "kernel",
    "program",
    "Program",
    "compute",
    "datamovement",
    "TensorBlock",
    "CircularBuffer",
    "CopyTransferHandler",
    "copy",
    "dma",
    "TensorAccessor",
    "CompiledTTNNKernel",
]
