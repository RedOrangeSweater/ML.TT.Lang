# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile pipeline: KernelCompileRequest -> (threads, module) -> TTNNKernelCompileRequest
-> CompiledTTNNKernel.

One focus: spec to compilation artifacts. _compile_kernel and _compile_ttnn_kernel
live here; ttl_api remains facade and passes thread_registry to avoid circular imports.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path
from types import CellType
from typing import Callable, Protocol

import ttl._mlir_libs._ttlang  # noqa: F401  # Register tt-lang passes
from pydantic import BaseModel, ConfigDict, Field
from ttmlir.ir import (
    ArrayAttr,
    Context,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    PassManager,
)

from .._src.ttl_ast import TTLGenericCompiler
from ..circular_buffer import CircularBuffer, get_cb_count
from ..constants import SUPPORTED_MEMORY_SPACES
from ..descriptor_options import (
    CompiledKernelArtifacts,
    CompiledProfilingSource,
    CompiledRuntimeContext,
    CoreRangeSetOptions,
    KernelWriteRequest,
    ProgramConfig,
    ProgramRunConfig,
    ThreadConfigBuildRequest,
    TTNNCompileCacheAndCb,
    TTNNCompileInput,
    TTNNKernelCompileOptions,
    TTNNKernelCompileRequest,
    TTNNProfilingInput,
)
from ..diagnostics import (
    TTLangCompileError,
    find_variable_assignment,
    format_mlir_error,
    format_python_error,
)
from ..dtype_utils import TTNNMemoryConfigProxy, is_ttnn_tensor
from ..program import KernelCompileRequest, Program
from ..settings import settings_ttlang
from ..ttl_utils import tmp_dir
try:
    import ttnn  # type: ignore[import-untyped]
except (ModuleNotFoundError, ImportError):
    ttnn = None

from .._src.auto_profile import is_auto_profile_enabled
from .._src.tensor_registry import register_tensor_name, register_tensor_source
from ..config import HAS_TT_DEVICE

# Optional: get_ttkernel_names, ttkernel_to_cpp_by_name, get_ttkernel_arg_spec
from ttmlir.dialects import ttkernel
from ttmlir.passes import get_ttkernel_arg_spec, get_ttkernel_names, ttkernel_to_cpp_by_name


class ThreadRegistryLike(Protocol):
    """Protocol for thread registry: clear and get_and_clear for compile pipeline."""

    def clear(self) -> None: ...
    def get_and_clear(self) -> list[Callable[..., object]]: ...


def _get_source_line_offset(f: Callable[..., object]) -> int:
    """Get the line offset to convert parsed AST line numbers to actual file lines."""
    try:
        raw_lines, start_lineno = inspect.getsourcelines(f)
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


def _has_float32_args(args: tuple[object, ...]) -> bool:
    """Check if any input tensor uses float32 dtype."""
    try:
        for tensor in args:
            if tensor is None:
                continue
            if is_ttnn_tensor(tensor):
                tensor_dtype = getattr(tensor, "dtype", None)
                if tensor_dtype is not None:
                    if (
                        hasattr(tensor_dtype, "name")
                        and "float32" in str(tensor_dtype.name).lower()
                    ):
                        return True
                    if "float32" in str(tensor_dtype).lower():
                        return True
            elif hasattr(tensor, "dtype"):
                import torch  # noqa: PLC0415

                if tensor.dtype == torch.float32:
                    return True
    except (AttributeError, TypeError, ImportError):
        pass
    return False


def _build_config_for_thread(request: ThreadConfigBuildRequest) -> tuple[object, dict[str, str]]:
    """Build (config, thread_to_kernel_entries) from a single Pydantic request."""
    return request.build_config_and_entries()


def _write_kernel_to_tmp(req: KernelWriteRequest) -> str:
    """Write kernel source to req.base_dir or /tmp/{user} and return the file path."""
    import hashlib
    import os

    content_hash = hashlib.md5(req.source.encode()).hexdigest()[:8]
    if req.base_dir is not None:
        req.base_dir.mkdir(parents=True, exist_ok=True)
        path = req.base_dir / f"ttlang_kernel_{req.name}_{content_hash}.cpp"
    else:
        user = settings_ttlang.user
        path = Path(f"/tmp/{user}/ttlang_kernel_{req.name}_{content_hash}.cpp")
        os.makedirs(f"/tmp/{user}", exist_ok=True)
    with path.open("w") as f:
        f.write(req.source)
    print(f"=== {req.name} kernel written to {path} ===")
    print(req.source)
    print("=" * 60)
    return str(path)


class ThreadWrapperView(BaseModel):
    """Typed view of a decorated thread: wrapped callable and its closure."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    wrapped: Callable[..., object] | None = None
    closure: tuple[CellType, ...] | None = None

    @classmethod
    def from_callable(cls, thread_fn: Callable[..., object]) -> ThreadWrapperView:
        wrapped = getattr(thread_fn, "__wrapped__", None)
        closure = getattr(wrapped, "__closure__", None) if wrapped else None
        if closure is not None:
            closure = tuple(closure)
        return cls(wrapped=wrapped, closure=closure)


def _collect_cb_configs(
    threads: list[Callable[..., object]],
) -> list[CircularBuffer | None]:
    """Extract CircularBuffer objects from thread closures, indexed by cb_index."""
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


def _track_tensor_sources(
    f_params: object,
    args: tuple[object, ...],
    source_file: str,
) -> None:
    """Track source locations for tensor arguments."""
    if source_file == "<unknown>":
        return
    try:
        with open(source_file) as sf:
            source_lines = sf.read().splitlines()
    except (OSError, IOError):
        return
    call_line = None
    for frame_info in inspect.stack():
        if frame_info.filename == source_file:
            call_line = frame_info.lineno
            break
    if call_line is None:
        return
    param_names = list(f_params) if hasattr(f_params, "__iter__") and not isinstance(f_params, (str, bytes)) else []
    for param_name, arg in zip(param_names, args):
        if not is_ttnn_tensor(arg):
            continue
        assign_line = find_variable_assignment(source_lines, param_name, call_line)
        if assign_line:
            register_tensor_source(arg, source_file, assign_line)


def _compile_ttnn_kernel(req: TTNNKernelCompileRequest) -> object | None:
    """
    Compile kernel to CompiledTTNNKernel for execution via ttnn.generic_op.

    Builds kernel paths, configs, and CB descriptors from compiled MLIR module.
    """
    from ..descriptor_options import (
        CompiledTTNNKernel,
    )

    # Validation runs in TTNNKernelCompileRequest model_validator (Pydantic).
    kernel_info = get_ttkernel_names(req.input.module)
    opts = req.compile_options or TTNNKernelCompileOptions()

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

    core_ranges = CoreRangeSetOptions(grid=req.input.grid).build_ttnn_core_range_set()
    if opts.verbose:
        print(f"\nCore range: {core_ranges}")

    kernel_paths: list[tuple[str, str]] = []
    kernel_configs: list[object] = []
    kernel_arg_specs: list[object] = []
    noc_kernel_idx = 0
    has_f32 = _has_float32_args(req.input.args)
    thread_to_kernel: dict[str, str] = {}

    base = Path(f"/tmp/{settings_ttlang.user}")
    base.mkdir(parents=True, exist_ok=True)
    with tmp_dir(base, lambda: f"ttlang_kernels_{uuid.uuid4().hex[:12]}") as base_dir:
        for name, thread_type in kernel_info:
            cpp_source = ttkernel_to_cpp_by_name(req.input.module, name)
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

            arg_spec = get_ttkernel_arg_spec(req.input.module, name)
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
    cfg.setdefault("grid", req.input.grid)
    cb = req.cache_and_cb
    prof = req.profiling
    artifacts = CompiledKernelArtifacts(
        kernel_paths=kernel_paths,
        kernel_configs=kernel_configs,
        kernel_arg_specs=kernel_arg_specs,
        kernel_tensor_indices=req.input.thread_tensor_indices,
        thread_to_kernel=thread_to_kernel,
        thread_names=thread_names,
    )
    runtime = CompiledRuntimeContext(
        num_tensors=len(req.input.args),
        core_ranges=core_ranges,
        cb_configs=cb.cb_configs if cb is not None else [],
        program_hash=cb.program_hash if cb is not None else None,
        program_config=cfg,
    )
    profiling = None
    if prof and (
        prof.source_lines is not None
        or prof.all_source_lines
        or prof.kernel_line_offsets
    ):
        profiling = CompiledProfilingSource(
            source_lines=prof.source_lines,
            all_source_lines=prof.all_source_lines or {},
            kernel_line_offsets=prof.kernel_line_offsets or {},
        )
    compiled_kernel = CompiledTTNNKernel(
        artifacts=artifacts, runtime=runtime, profiling=profiling
    )

    if opts.verbose:
        print(f"\nCompiled kernel ready (compiled {len(kernel_paths)} threads)")
        print("=" * 60)

    return compiled_kernel


def _compile_kernel(
    f: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    request: KernelCompileRequest,
    thread_registry: ThreadRegistryLike,
    engine_config: object | None = None,
) -> object | None:
    """
    Compile kernel function to MLIR and return CompiledTTNNKernel.

    thread_registry is passed from ttl_api (where @compute/@datamovement register).
    """
    memory_space = request.options.memory_space  # may be updated from tensor
    f_params = inspect.signature(f).parameters

    try:
        kernel_source_file = inspect.getfile(f)
        kernel_line_offset = _get_source_line_offset(f)
    except (TypeError, OSError):
        kernel_source_file = "<unknown>"
        kernel_line_offset = 0

    has_ttnn_tensors = any(is_ttnn_tensor(arg) for arg in args)
    if has_ttnn_tensors:
        first_ttnn_tensor = next((arg for arg in args if is_ttnn_tensor(arg)), None)
        if first_ttnn_tensor is not None:
            proxy = TTNNMemoryConfigProxy(tensor=first_ttnn_tensor, default=memory_space)
            detected = proxy.memory_space
            if detected in SUPPORTED_MEMORY_SPACES:
                memory_space = detected
                print(f"[TTNN interop] Detected {memory_space} memory space")

    for idx, (param_name, arg) in enumerate(zip(f_params, args)):
        register_tensor_name(arg, param_name, index=idx)
    _track_tensor_sources(f_params, args, kernel_source_file)

    run_config = ProgramRunConfig(
        grid=list(request.grid),
        memory_space=memory_space,
        tiled=request.options.tiled,
        debug_locations=True,
    )
    run_config.inject_into_kwargs(kwargs, set(f_params))

    from ..circular_buffer import _reset_cb_counter
    from ..operators import _set_current_grid

    _reset_cb_counter()
    _set_current_grid(request.grid)

    thread_registry.clear()
    f(*args, **kwargs)
    threads = thread_registry.get_and_clear()

    if not threads:
        raise ValueError(
            "No threads found. Define at least one @ttl.compute() or "
            "@ttl.datamovement() function inside your kernel."
        )

    cb_configs = _collect_cb_configs(threads)
    program = Program(*threads, args=args, run_config=run_config)
    print_debug_locations = settings_ttlang.debug_locations

    ctx = Context()
    loc = Location.unknown(ctx)
    with ctx, loc:
        compiled_threads: list[TTLGenericCompiler] = []
        thread_tensor_indices: list[list[int]] = []

        for compile_thread in program.threads:
            try:
                ct = compile_thread(*program.args, **program.kwargs)
            except TTLangCompileError as e:
                raise type(e)(e.format()) from None
            except (ValueError, TypeError) as e:
                formatted = format_python_error(
                    e, kernel_source_file, kernel_line_offset
                )
                raise type(e)(formatted) from None
            compiled_threads.append(ct)
            thread_tensor_indices.append(ct._tensor_accessor_global_indices)

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

        if settings_ttlang.use_scheduler:
            from ..scheduler import (
                build_op_graph_from_threads,
                build_topology_from_grid,
                schedule_stub,
                schedule_toy_stub,
                validate_topology_connectivity,
            )

            engine_cfg = engine_config
            if engine_cfg is None:
                thread_infos = [
                    (ct.name, ct.kernel_type or "dm") for ct in compiled_threads
                ]
                op_graph = build_op_graph_from_threads(thread_infos)
                topology = build_topology_from_grid(request.grid)
                plan = schedule_stub(op_graph, topology, request.options.program_config_dict)
                assert (
                    plan.grid_cols == topology.grid_cols
                    and plan.grid_rows == topology.grid_rows
                )
                for nid in op_graph.node_ids_in_order():
                    assert plan.core_for(nid) == (0, 0), f"stub plan mismatch for {nid}"
            elif getattr(engine_cfg, "backend", None) == "tenstorrent":
                thread_infos = [
                    (ct.name, ct.kernel_type or "dm") for ct in compiled_threads
                ]
                op_graph = build_op_graph_from_threads(thread_infos)
                topo_grid = (
                    engine_cfg.get_topology_grid()
                    if hasattr(engine_cfg, "get_topology_grid")
                    else None
                )
                grid_for_topology = topo_grid if topo_grid is not None else request.grid
                topology = build_topology_from_grid(grid_for_topology)
                plan = schedule_stub(op_graph, topology, request.options.program_config_dict)
                assert (
                    plan.grid_cols == topology.grid_cols
                    and plan.grid_rows == topology.grid_rows
                )
                for nid in op_graph.node_ids_in_order():
                    assert plan.core_for(nid) == (0, 0), f"stub plan mismatch for {nid}"
            else:
                validate_topology_connectivity(engine_cfg)
                schedule_toy_stub(engine_cfg)

        module = Module.create(loc)
        with InsertionPoint(module.body):
            for ct in compiled_threads:
                ct.func_entry.operation.detach_from_parent()
                module.body.append(ct.func_entry)

        initial_mlir_path = settings_ttlang.initial_mlir
        if initial_mlir_path:
            with open(initial_mlir_path, "w") as fd:
                module.operation.print(
                    file=fd,
                    enable_debug_info=print_debug_locations,
                    print_generic_op_form=False,
                )
            print(f"SAVED INITIAL TO {initial_mlir_path}")

        verify = True
        set_compute_config_pass = "func.func(ttl-set-compute-kernel-config)"
        config_options = []
        if request.options.fp32_dest_acc_en is not None:
            config_options.append(
                f"fp32-dest-acc-en={1 if request.options.fp32_dest_acc_en else 0}"
            )
        if request.options.dst_full_sync_en is not None:
            config_options.append(
                f"dst-full-sync-en={1 if request.options.dst_full_sync_en else 0}"
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
        if is_auto_profile_enabled():
            st = settings_ttlang
            if st.profile_csv:
                cb_flow_json = str(Path(st.profile_csv).parent / "cb_flow_graph.json")
            else:
                tt_metal_home = st.tt_metal_home
                if not tt_metal_home:
                    raise ValueError(
                        "TTLANG_AUTO_PROFILE=1 requires TT_METAL_HOME or TTLANG_PROFILE_CSV to be set"
                    )
                cb_flow_json = f"{tt_metal_home}/generated/profiler/.logs/cb_flow_graph.json"
            pipeline_passes.append(f'ttl-dump-cb-flow-graph{{output="{cb_flow_json}"}}')
        pipeline_passes += ["convert-ttl-to-ttkernel"]
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

        pipeline_str = f"builtin.module({','.join(pipeline_passes)})"
        pm = PassManager.parse(pipeline_str)
        pm.enable_verifier(verify)
        try:
            from ttmlir._mlir_libs._ttmlir import enable_pretty_stack_traces
            enable_pretty_stack_traces(pm._CAPIPtr)
        except Exception:
            pass

        if settings_ttlang.verbose_passes:
            print("Running custom pipeline:", pm)
            ctx.enable_multithreading(False)
            pm.enable_ir_printing(
                print_after_all=True,
                print_before_all=True,
                print_after_failure=True,
                enable_debug_info=True,
            )

        try:
            pm.run(module.operation)
        except Exception as e:
            error_msg = str(e)
            source_lines = None
            source_file = None
            if all_source_lines:
                first_thread = next(iter(all_source_lines.keys()))
                source_lines = all_source_lines[first_thread]
                source_file = all_source_files.get(first_thread)
            formatted = format_mlir_error(error_msg, source_lines, source_file)
            raise RuntimeError(formatted) from None

        final_mlir_path = settings_ttlang.final_mlir
        if final_mlir_path:
            with open(final_mlir_path, "w") as fd:
                module.operation.print(
                    file=fd,
                    enable_debug_info=print_debug_locations,
                    print_generic_op_form=False,
                )
            print(f"SAVED FINAL TO {final_mlir_path}")

        profile_source_lines = None
        if all_source_lines:
            first_thread = next(iter(all_source_lines.keys()))
            profile_source_lines = all_source_lines[first_thread]

        compile_input = TTNNCompileInput(
            module=module,
            args=args,
            grid=request.grid,
            num_outs=request.options.num_outs,
            thread_tensor_indices=thread_tensor_indices,
        )
        cache_and_cb = TTNNCompileCacheAndCb(
            cb_configs=cb_configs, program_hash=request.program_hash
        )
        profiling_input = TTNNProfilingInput(
            source_lines=profile_source_lines,
            all_source_lines=all_source_lines or None,
            kernel_line_offsets=kernel_line_offsets or None,
        )
        compile_req = TTNNKernelCompileRequest(
            input=compile_input,
            compile_options=TTNNKernelCompileOptions(
                fp32_dest_acc_en=request.options.fp32_dest_acc_en,
                dst_full_sync_en=request.options.dst_full_sync_en,
                program_config=request.options.program_config_dict,
            ),
            cache_and_cb=cache_and_cb,
            profiling=profiling_input,
        )
        return _compile_ttnn_kernel(compile_req)


__all__ = [
    "_compile_kernel",
    "_compile_ttnn_kernel",
    "_get_source_line_offset",
]
