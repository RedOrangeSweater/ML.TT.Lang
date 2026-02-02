# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile pipeline: KernelCompileRequest -> (threads, module) -> TTNNKernelCompileRequest
-> CompiledTTNNKernel.

Orchestration only: delegates to thread_compiler, ttnn_compiler, source_collector,
stages.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from __future__ import annotations

from collections.abc import Sequence

from ttmlir.ir import Context, Location, PassManager

import ttl._mlir_libs._ttlang  # noqa: F401  # Register tt-lang passes

from .._src.auto_profile import is_auto_profile_enabled
from ..boundary import MlirModuleLike
from ..config import HAS_TT_DEVICE
from ..descriptor_options import (
    CompiledTTNNKernel,
    ProgramConfig,
    TTNNCompileCacheAndCb,
    TTNNCompileInput,
    TTNNKernelCompileOptions,
    TTNNKernelCompileRequest,
    TTNNProfilingInput,
)
from ..diagnostics import format_mlir_error
from ..program import CompileKernelRequest, KernelCompileRequest
from ..scheduler import AbstractEngineConfig
from ..settings import AutoProfileConfig, settings_ttlang
from .source_collector import get_source_line_offset
from .thread_compiler import (
    CompiledThreadsResult,
    build_module_from_threads,
    compile_program_to_threads,
    resolve_memory_space_and_register_tensors,
)
from .ttnn_compiler import compile_ttnn_kernel


def build_ttnn_compile_request(
    module: MlirModuleLike,
    args: tuple[object, ...],
    request: KernelCompileRequest,
    thread_tensor_indices: list[list[int]],
    cb_configs: list[object] | None,
    all_source_lines: dict[str, list[str]] | None,
    kernel_line_offsets: dict[str, int] | None,
) -> TTNNKernelCompileRequest:
    """Build TTNNKernelCompileRequest from pipeline artifacts."""
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
    profile_source_lines = None
    if all_source_lines:
        first_thread = next(iter(all_source_lines.keys()))
        profile_source_lines = all_source_lines[first_thread]
    profiling_input = TTNNProfilingInput(
        source_lines=profile_source_lines,
        all_source_lines=all_source_lines,
        kernel_line_offsets=kernel_line_offsets,
    )
    program_config = ProgramConfig(
        grid=request.grid,
        objective=request.options.objective,
        placement=request.options.placement,
    )
    return TTNNKernelCompileRequest(
        input=compile_input,
        compile_options=TTNNKernelCompileOptions(
            fp32_dest_acc_en=request.options.fp32_dest_acc_en,
            dst_full_sync_en=request.options.dst_full_sync_en,
            program_config=program_config,
        ),
        cache_and_cb=cache_and_cb,
        profiling=profiling_input,
    )


def _run_scheduler_if_enabled(
    *,
    compiled_threads: Sequence[object],
    request: KernelCompileRequest,
    engine_config: AbstractEngineConfig | None,
) -> None:
    """Run scheduler stub logic when settings_ttlang.use_scheduler is enabled."""
    if not settings_ttlang.use_scheduler:
        return

    from ..scheduler import (
        build_op_graph_from_threads,
        build_topology_from_grid,
        schedule_stub,
        schedule_toy_stub,
        validate_topology_connectivity,
    )

    if engine_config is None:
        thread_infos: list[tuple[str, str]] = []
        for ct in compiled_threads:
            name = str(getattr(ct, "name", "<unknown>"))
            kernel_type = getattr(ct, "kernel_type", None)
            thread_infos.append((name, str(kernel_type or "dm")))
        op_graph = build_op_graph_from_threads(thread_infos)
        topology = build_topology_from_grid(request.grid)
        plan = schedule_stub(op_graph, topology, request.options.program_config_dict)
        assert plan.grid_cols == topology.grid_cols
        assert plan.grid_rows == topology.grid_rows
        for nid in op_graph.node_ids_in_order():
            assert plan.core_for(nid) == (0, 0), f"stub plan mismatch for {nid}"
        return

    validate_topology_connectivity(engine_config)
    schedule_toy_stub(engine_config)


def _build_pipeline_passes(request: KernelCompileRequest) -> list[str]:
    """Build MLIR pass pipeline list based on request options and settings."""
    set_compute_config_pass = "func.func(ttl-set-compute-kernel-config)"
    config_options: list[str] = []
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
            "func.func(ttl-set-compute-kernel-config{" + " ".join(config_options) + "})"
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
        cfg = AutoProfileConfig(
            profile_csv=settings_ttlang.profile_csv,
            tt_metal_home=settings_ttlang.tt_metal_home or None,
        )
        cb_flow_json = cfg.cb_flow_graph_json_path()
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

    return pipeline_passes


def _run_pass_pipeline(
    *,
    module: MlirModuleLike,
    ctx: Context,
    pipeline_passes: list[str],
    all_source_files: dict[str, str],
    all_source_lines: dict[str, list[str]] | None,
) -> None:
    """Run MLIR pass pipeline with optional IR printing and formatted errors."""
    pipeline_str = f"builtin.module({','.join(pipeline_passes)})"
    pm = PassManager.parse(pipeline_str)
    pm.enable_verifier(True)
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


def _compile_kernel(req: CompileKernelRequest) -> CompiledTTNNKernel | None:
    """
    Compile kernel function to MLIR and return CompiledTTNNKernel.

    Single request object bundles program, args/kwargs, compile params, and
    thread registry.
    Orchestrates thread_compiler, pass manager, stages, and ttnn_compiler.
    """
    request = req.compile_request
    engine_config = req.engine_config

    memory_space, kernel_source_file, kernel_line_offset = (
        resolve_memory_space_and_register_tensors(req)
    )

    ctx = Context()
    loc = Location.unknown(ctx)
    with ctx, loc:
        result: CompiledThreadsResult = compile_program_to_threads(
            req,
            memory_space,
            ctx,
            loc,
            kernel_source_file,
            kernel_line_offset,
        )

        _run_scheduler_if_enabled(
            compiled_threads=result.compiled_threads,
            request=request,
            engine_config=engine_config,
        )

        module = build_module_from_threads(result.compiled_threads, loc, ctx)
        print_debug_locations = settings_ttlang.debug_locations

        from .stages import CompileStageContext, get_initial_stages, run_stages

        stage_ctx = CompileStageContext(
            initial_mlir_path=settings_ttlang.initial_mlir,
            module=module,
            print_debug_locations=print_debug_locations,
        )
        run_stages(get_initial_stages(), stage_ctx, settings_ttlang)

        pipeline_passes = _build_pipeline_passes(request)
        _run_pass_pipeline(
            module=module,
            ctx=ctx,
            pipeline_passes=pipeline_passes,
            all_source_files=result.all_source_files,
            all_source_lines=result.all_source_lines,
        )

        from .stages import CompileStageContext, get_final_stages, run_stages

        final_stage_ctx = CompileStageContext(
            module=module,
            final_mlir_path=settings_ttlang.final_mlir,
            print_debug_locations=print_debug_locations,
        )
        run_stages(get_final_stages(), final_stage_ctx, settings_ttlang)

        ttnn_req = build_ttnn_compile_request(
            module=module,
            args=req.args,
            request=request,
            thread_tensor_indices=result.thread_tensor_indices,
            cb_configs=result.cb_configs,
            all_source_lines=result.all_source_lines,
            kernel_line_offsets=result.kernel_line_offsets,
        )
    return compile_ttnn_kernel(ttnn_req)


# Backward compatibility: same names as before refactor
_compile_ttnn_kernel = compile_ttnn_kernel
_get_source_line_offset = get_source_line_offset


__all__ = [
    "_compile_kernel",
    "_compile_ttnn_kernel",
    "_get_source_line_offset",
]
