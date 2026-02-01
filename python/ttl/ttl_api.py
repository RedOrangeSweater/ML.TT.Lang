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
from pathlib import Path
from types import CellType
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from .scheduler import AbstractEngineConfig

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

try:
    import ttnn
except (ModuleNotFoundError, ImportError):
    ttnn = None

import ttl._mlir_libs._ttlang  # Register tt-lang passes
from pykernel._src.utils import _cleanup_source_code

from ._src.auto_profile import (
    build_cb_wait_to_dma_map,
    build_dma_producer_to_cb_map,
    get_line_mapper,
    is_auto_profile_enabled,
    load_cb_flow_graph,
    parse_device_profile_csv,
    print_profile_report,
)
from ._src.tensor_registry import get_tensor_global_index, get_tensor_source
from ._src.ttl_ast import TTLCompilerConfig, TTLGenericCompiler
from .circular_buffer import CircularBuffer
from .constants import MemorySpace, SUPPORTED_MEMORY_SPACES
from .diagnostics import (
    TTLangCompileError,
    format_mlir_error,
    format_python_error,
)
from .dtype_utils import (
    TensorDtype,
    TTNNMemoryConfigProxy,
    is_ttnn_tensor,
    tile_bytes_from_dtype,
)
from .kernel_runner import (
    KernelSpec,
    RunKernelRequest,
    run_kernel_on_device,
)
from .operators import CopyTransferHandler, TensorBlock, copy
from .compile.pipeline import (
    _compile_kernel as _compile_kernel_impl,
    _get_source_line_offset,
)
from .program import (
    KernelCompileRequest,
    Program,
    ProgramDecoratorParams,
    ProgramOptions,
    ProgramSpec,
    RunRequest,
    _resolve_grid,
)
from .descriptor_options import (
    CompiledKernelArtifacts,
    CompiledProfilingSource,
    CompiledRuntimeContext,
    NocConfigOptions,
    ProgramConfig,
    ProgramRunConfig,
    ReaderConfigOptions,
    TTNNCompileCacheAndCb,
    TTNNCompileInput,
    TTNNKernelCompileOptions,
    TTNNKernelCompileRequest,
    TTNNProfilingInput,
)
from .settings import settings_ttlang
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


def _get_tensor_cache_info(tensor: object) -> tuple[tuple[int, ...], str, MemorySpace, str]:
    """Extract cache-relevant info from a tensor: (shape, dtype, memory_space, layout)."""
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    proxy = TTNNMemoryConfigProxy(tensor=tensor, default="unknown")
    memory_space = proxy.memory_space
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
    return not settings_ttlang.compile_only


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
    settings = settings_ttlang
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


class CompiledTTNNKernel(BaseModel):
    """
    A compiled tt-lang kernel ready for execution via ttnn.generic_op.

    Caches compilation artifacts (kernel paths, CB descriptors) so the kernel
    can be executed multiple times with different tensors without recompiling.
    Structured as artifacts (per-kernel output), runtime (execution context),
    and optional profiling (source lines).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifacts: CompiledKernelArtifacts = Field(
        ...,
        description="Per-kernel compilation output (paths, configs, thread mapping)",
    )
    runtime: CompiledRuntimeContext = Field(
        ...,
        description="Execution context (num_tensors, core_ranges, cb_configs, etc.)",
    )
    profiling: CompiledProfilingSource | None = Field(
        default=None, description="Source lines for profiling/debugging"
    )

    # Backward-compatibility aliases (read from nested models).
    @property
    def kernel_paths(self) -> list[tuple[str, str]]:
        return self.artifacts.kernel_paths

    @property
    def kernel_configs(self) -> list[object]:
        return self.artifacts.kernel_configs

    @property
    def kernel_arg_specs(self) -> list[object]:
        return self.artifacts.kernel_arg_specs

    @property
    def kernel_tensor_indices(self) -> list[list[int]]:
        return self.artifacts.kernel_tensor_indices

    @property
    def thread_to_kernel(self) -> dict[str, str]:
        return self.artifacts.thread_to_kernel

    @property
    def thread_names(self) -> list[str]:
        return self.artifacts.thread_names

    @property
    def num_tensors(self) -> int:
        return self.runtime.num_tensors

    @property
    def core_ranges(self) -> object:
        return self.runtime.core_ranges

    @property
    def cb_configs(self) -> list[object]:
        return self.runtime.cb_configs

    @property
    def program_hash(self) -> object | None:
        return self.runtime.program_hash

    @property
    def program_config(self) -> dict[str, object]:
        return self.runtime.program_config

    @property
    def source_lines(self) -> object | None:
        return self.profiling.source_lines if self.profiling else None

    @property
    def all_source_lines(self) -> dict[str, object]:
        return self.profiling.all_source_lines if self.profiling else {}

    @property
    def kernel_line_offsets(self) -> dict[str, object]:
        return self.profiling.kernel_line_offsets if self.profiling else {}

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


# -----------------------------------------------------------------------------
# Compile layer: KernelCompileRequest -> (threads, module) -> TTNNKernelCompileRequest
# -> CompiledTTNNKernel. Implementation in .compile.pipeline; facade delegates.
# -----------------------------------------------------------------------------

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
    engine_config_path: str | Path | None = None,
    engine_config: AbstractEngineConfig | None = None,
    **kwargs: object,
) -> object | None:
    """
    Facade: compile and run kernel from a spec or program. Ideal UX entry point.

    Two forms:
    - run(spec, *args, **kwargs) — spec is ProgramSpec (program + grid + options).
    - run(program, *args, grid=..., options=..., **kwargs) — program is @ttl.program-decorated;
      grid is required; options default to ProgramOptions().

    Optional engine_config_path or engine_config: abstract engine config for scheduler
    (backend, topology, objective). When use_scheduler is True, tenstorrent backend uses
    topology from config if topology_grid is set; toy backend runs schedule_toy_stub.
    See docs/sdlc/00_Main/00_Ideas/20_nickel_mlir_config_abstract_engine.md.

    Lambda / raw callable: passing a lambda (e.g. lambda lhs, rhs: lhs + rhs) is planned;
    inference from parameters and return type is not yet implemented. Use @ttl.program for now.
    See 20_IdealDataFlowAndModuleStructure.md (lambda + inference).
    """
    if engine_config_path is not None:
        from .scheduler import load_abstract_engine_config

        engine_config = load_abstract_engine_config(engine_config_path)
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

    req = RunRequest(spec=spec, args=args, kwargs=kwargs)
    cache_key = _make_cache_key(
        req.args,
        fp32_dest_acc_en=req.spec.options.fp32_dest_acc_en,
        dst_full_sync_en=req.spec.options.dst_full_sync_en,
    )
    program_hash = hash((id(req.spec.program), cache_key))
    request = req.spec.to_compile_request(req.args, req.kwargs, program_hash)
    compiled = _compile_kernel_impl(
        req.spec.program,
        req.args,
        dict(req.kwargs),
        request,
        _thread_registry,
        engine_config,
    )
    if compiled is None:
        return None
    if _should_execute():
        return compiled(*req.args)
    return None


def pykernel_gen(
    grid: (tuple[int, ...] | Callable[..., object]) | None = None,
    indexing_maps: list[Callable[..., object]] | None = None,
    iterator_types: list[str] | None = None,
    num_outs: int = 1,
    memory_space: MemorySpace = "L1",
    tiled: bool = True,
    fp32_dest_acc_en: bool | None = None,
    dst_full_sync_en: bool | None = None,
    objective: Literal["latency", "throughput", "balanced"] | None = None,
    placement: Literal["auto", "manual"] | None = None,
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
        memory_space: MemorySpace (L1 or DRAM)
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
    params = ProgramDecoratorParams(
        grid=grid,
        indexing_maps=indexing_maps,
        iterator_types=iterator_types,
        num_outs=num_outs,
        memory_space=memory_space,
        tiled=tiled,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
        objective=objective,
        placement=placement,
    )
    options = params.to_program_options()
    indexing_maps = params.indexing_maps if params.indexing_maps is not None else []
    iterator_types = params.iterator_types if params.iterator_types is not None else []

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
                    indexing_maps=indexing_maps_list,
                    iterator_types=iterator_types_list,
                    options=options,
                )
                compiled_kernel = _compile_kernel_impl(
                    f, args, kwargs, compile_request, _thread_registry, None
                )

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
