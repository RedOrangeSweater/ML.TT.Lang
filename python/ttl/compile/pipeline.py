# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile pipeline: KernelCompileRequest -> (threads, module) -> TTNNKernelCompileRequest
-> CompiledTTNNKernel.

Orchestration only: delegates to thread_compiler, ttnn_compiler, source_collector, stages.
See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ttmlir.ir import Context, Location, PassManager

import ttl._mlir_libs._ttlang  # noqa: F401  # Register tt-lang passes

from .._src.auto_profile import is_auto_profile_enabled
from ..config import HAS_TT_DEVICE
from ..descriptor_options import (
    ProgramConfig,
    TTNNCompileCacheAndCb,
    TTNNCompileInput,
    TTNNKernelCompileOptions,
    TTNNKernelCompileRequest,
    TTNNProfilingInput,
)
from ..diagnostics import format_mlir_error
from ..program import CompileKernelRequest
from ..settings import settings_ttlang

from .source_collector import get_source_line_offset
from .thread_compiler import (
    build_module_from_threads,
    compile_program_to_threads,
    resolve_memory_space_and_register_tensors,
)
from .ttnn_compiler import compile_ttnn_kernel


class ThreadRegistryLike(Protocol):
    """Protocol for thread registry: clear and get_and_clear for compile pipeline."""

    def clear(self) -> None: ...
    def get_and_clear(self) -> list[Callable[..., object]]: ...


def _compile_kernel(req: CompileKernelRequest) -> object | None:
    """
    Compile kernel function to MLIR and return CompiledTTNNKernel.

    Single request object bundles program, args/kwargs, compile params, and thread registry.
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
        (
            compiled_threads,
            thread_tensor_indices,
            all_source_files,
            all_source_lines,
            kernel_line_offsets,
            cb_configs,
            program,
        ) = compile_program_to_threads(
            req,
            memory_space,
            ctx,
            loc,
            kernel_source_file,
            kernel_line_offset,
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
                plan = schedule_stub(
                    op_graph, topology, request.options.program_config_dict
                )
                assert (
                    plan.grid_cols == topology.grid_cols
                    and plan.grid_rows == topology.grid_rows
                )
                for nid in op_graph.node_ids_in_order():
                    assert plan.core_for(nid) == (0, 0), (
                        f"stub plan mismatch for {nid}"
                    )
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
                grid_for_topology = (
                    topo_grid if topo_grid is not None else request.grid
                )
                topology = build_topology_from_grid(grid_for_topology)
                plan = schedule_stub(
                    op_graph, topology, request.options.program_config_dict
                )
                assert (
                    plan.grid_cols == topology.grid_cols
                    and plan.grid_rows == topology.grid_rows
                )
                for nid in op_graph.node_ids_in_order():
                    assert plan.core_for(nid) == (0, 0), (
                        f"stub plan mismatch for {nid}"
                    )
            else:
                validate_topology_connectivity(engine_cfg)
                schedule_toy_stub(engine_cfg)

        module = build_module_from_threads(compiled_threads, loc, ctx)
        print_debug_locations = settings_ttlang.debug_locations

        from .stages import CompileStageContext, get_initial_stages, run_stages

        stage_ctx = CompileStageContext(
            module=module,
            initial_mlir_path=settings_ttlang.initial_mlir,
            print_debug_locations=print_debug_locations,
        )
        run_stages(get_initial_stages(), stage_ctx, settings_ttlang)

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
                cb_flow_json = str(
                    Path(st.profile_csv).parent / "cb_flow_graph.json"
                )
            else:
                tt_metal_home = st.tt_metal_home
                if not tt_metal_home:
                    raise ValueError(
                        "TTLANG_AUTO_PROFILE=1 requires TT_METAL_HOME or "
                        "TTLANG_PROFILE_CSV to be set"
                    )
                cb_flow_json = (
                    f"{tt_metal_home}/generated/profiler/.logs/cb_flow_graph.json"
                )
            pipeline_passes.append(
                f'ttl-dump-cb-flow-graph{{output="{cb_flow_json}"}}'
            )
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
            formatted = format_mlir_error(
                error_msg, source_lines, source_file
            )
            raise RuntimeError(formatted) from None

        from .stages import CompileStageContext, get_final_stages, run_stages

        final_stage_ctx = CompileStageContext(
            module=module,
            final_mlir_path=settings_ttlang.final_mlir,
            print_debug_locations=print_debug_locations,
        )
        run_stages(get_final_stages(), final_stage_ctx, settings_ttlang)

        profile_source_lines = None
        if all_source_lines:
            first_thread = next(iter(all_source_lines.keys()))
            profile_source_lines = all_source_lines[first_thread]

        compile_input = TTNNCompileInput(
            module=module,
            args=req.args,
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
        grid_tuple: tuple[int, int] | None = None
        if isinstance(request.grid, (tuple, list)) and len(request.grid) >= 2:
            grid_tuple = (int(request.grid[0]), int(request.grid[1]))
        program_config = ProgramConfig(
            grid=grid_tuple,
            objective=request.options.objective,
            placement=request.options.placement,
        )
        compile_req = TTNNKernelCompileRequest(
            input=compile_input,
            compile_options=TTNNKernelCompileOptions(
                fp32_dest_acc_en=request.options.fp32_dest_acc_en,
                dst_full_sync_en=request.options.dst_full_sync_en,
                program_config=program_config,
            ),
            cache_and_cb=cache_and_cb,
            profiling=profiling_input,
        )
        return compile_ttnn_kernel(compile_req)


# Backward compatibility: same names as before refactor
_compile_ttnn_kernel = compile_ttnn_kernel
_get_source_line_offset = get_source_line_offset


__all__ = [
    "_compile_kernel",
    "_compile_ttnn_kernel",
    "_get_source_line_offset",
]
