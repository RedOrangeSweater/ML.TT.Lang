# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Thread compiler: run program to get threads, compile each to TTLGenericCompiler, build module.

Compiles program to list of TTLGenericCompiler instances and assembles MLIR module.
Scheduler validation runs in pipeline; pass manager and stages stay in pipeline.
"""

from __future__ import annotations

import inspect

from ttmlir.ir import (
    ArrayAttr,
    Context,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
)

from .._src.tensor_registry import register_tensor_name
from .._src.ttl_ast import TTLGenericCompiler
from ..circular_buffer import get_cb_count
from ..constants import SUPPORTED_MEMORY_SPACES
from ..descriptor_options import ProgramRunConfig
from ..diagnostics import TTLangCompileError, format_python_error
from ..dtype_utils import TTNNMemoryConfigProxy, is_ttnn_tensor
from ..program import CompileKernelRequest, Program
from .source_collector import (
    collect_cb_configs,
    collect_source_info_from_threads,
    get_source_line_offset,
    track_tensor_sources,
)


def compile_program_to_threads(
    req: CompileKernelRequest,
    memory_space: str,
    ctx: Context,
    loc: Location,
    kernel_source_file: str,
    kernel_line_offset: int,
) -> tuple[
    list[TTLGenericCompiler],
    list[list[int]],
    dict[str, str],
    dict[str, list[str]],
    dict[str, int],
    list[object],
    Program,
]:
    """
    Run program to collect threads, compile each to TTLGenericCompiler, collect source info.

    Call resolve_memory_space_and_register_tensors(req) first to get memory_space and
    register tensor names / track sources. Returns (compiled_threads, thread_tensor_indices,
    all_source_files, all_source_lines, kernel_line_offsets, cb_configs, program).
    """
    f = req.program
    args = req.args
    kwargs = dict(req.kwargs)
    request = req.compile_request
    thread_registry = req.thread_registry
    f_params = inspect.signature(f).parameters

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

    cb_configs = collect_cb_configs(threads)
    program = Program(*threads, args=args, run_config=run_config)

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
        collect_source_info_from_threads(compiled_threads)
    )

    return (
        compiled_threads,
        thread_tensor_indices,
        all_source_files,
        all_source_lines,
        kernel_line_offsets,
        cb_configs,
        program,
    )


def build_module_from_threads(
    compiled_threads: list[TTLGenericCompiler],
    loc: Location,
    ctx: Context,
) -> Module:
    """Assemble MLIR module from compiled thread func_entries."""
    module = Module.create(loc)
    with InsertionPoint(module.body):
        for ct in compiled_threads:
            ct.func_entry.operation.detach_from_parent()
            module.body.append(ct.func_entry)
    return module


def resolve_memory_space_and_register_tensors(
    req: CompileKernelRequest,
) -> tuple[str, str, int]:
    """
    Resolve memory_space from request and tensors; register tensor names and track sources.

    Returns (memory_space, kernel_source_file, kernel_line_offset).
    """
    f = req.program
    args = req.args
    request = req.compile_request
    memory_space = request.options.memory_space
    f_params = inspect.signature(f).parameters

    try:
        kernel_source_file = inspect.getfile(f)
        kernel_line_offset = get_source_line_offset(f)
    except (TypeError, OSError):
        kernel_source_file = "<unknown>"
        kernel_line_offset = 0

    has_ttnn_tensors = any(is_ttnn_tensor(arg) for arg in args)
    if has_ttnn_tensors:
        first_ttnn_tensor = next((arg for arg in args if is_ttnn_tensor(arg)), None)
        if first_ttnn_tensor is not None:
            proxy = TTNNMemoryConfigProxy(
                tensor=first_ttnn_tensor, default=memory_space
            )
            detected = proxy.memory_space
            if detected in SUPPORTED_MEMORY_SPACES:
                memory_space = detected
                print(f"[TTNN interop] Detected {memory_space} memory space")

    for idx, (param_name, arg) in enumerate(zip(f_params, args, strict=False)):
        register_tensor_name(arg, param_name, index=idx)
    track_tensor_sources(f_params, args, kernel_source_file)

    return memory_space, kernel_source_file, kernel_line_offset


__all__ = [
    "compile_program_to_threads",
    "build_module_from_threads",
    "resolve_memory_space_and_register_tensors",
]
