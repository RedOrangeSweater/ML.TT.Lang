# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN compiler: TTNNKernelCompileRequest -> CompiledTTNNKernel.

Builds kernel paths, configs, and CB descriptors from compiled MLIR module.
Single focus: request to CompiledTTNNKernel for execution via ttnn.generic_op.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from ttmlir.dialects import ttkernel
from ttmlir.passes import (
    get_ttkernel_arg_spec,
    get_ttkernel_names,
    ttkernel_to_cpp_by_name,
)

try:
    import ttnn  # type: ignore[import-untyped]
except (ModuleNotFoundError, ImportError):
    ttnn = None

from ..descriptor_options import (
    CompiledKernelArtifacts,
    CompiledProfilingSource,
    CompiledRuntimeContext,
    CompiledTTNNKernel,
    CoreRangeSetOptions,
    KernelWriteRequest,
    ThreadConfigBuildRequest,
    TTNNKernelCompileOptions,
    TTNNKernelCompileRequest,
)
from ..dtype_utils import is_ttnn_tensor
from ..settings import settings_ttlang
from ..ttl_utils import tmp_dir
from .kernel_writer import write_kernel_to_tmp


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


def compile_ttnn_kernel(req: TTNNKernelCompileRequest) -> CompiledTTNNKernel | None:
    """
    Compile kernel to CompiledTTNNKernel for execution via ttnn.generic_op.

    Builds kernel paths, configs, and CB descriptors from compiled MLIR module.
    """
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
            kernel_path = write_kernel_to_tmp(
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
    cfg = raw_cfg.model_dump(exclude_none=False) if raw_cfg is not None else {}
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
            all_source_lines=(
                prof.all_source_lines if prof.all_source_lines is not None else {}
            ),
            kernel_line_offsets=(
                prof.kernel_line_offsets if prof.kernel_line_offsets is not None else {}
            ),
        )
    compiled_kernel = CompiledTTNNKernel(
        artifacts=artifacts, runtime=runtime, profiling=profiling
    )

    if opts.verbose:
        print(f"\nCompiled kernel ready (compiled {len(kernel_paths)} threads)")
        print("=" * 60)

    return compiled_kernel


__all__ = ["compile_ttnn_kernel"]
