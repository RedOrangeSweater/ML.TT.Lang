# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Python dialect layers (program, descriptor_options, ttnn_proxy, kernel_runner).

Verifies Pydantic request models and layer boundaries: Program/Compile/Runtime
dialects and transformations. See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

import pytest
from pydantic import ValidationError

from ttl import run
from ttl.descriptor_options import (
    ComputeConfigOptions,
    CoreRangeSetOptions,
    ThreadConfigBuildRequest,
)
from ttl.kernel_runner import (
    CBDescriptorBuildRequest,
    KernelDescriptorBuildRequest,
    RunKernelRequest,
    build_cb_descriptors,
    build_kernel_descriptors,
)
from ttl.program import (
    CompileKernelRequest,
    KernelCompileRequest,
    ProgramDecoratorParams,
    ProgramOptions,
    ProgramSpec,
    RunRequest,
    _resolve_grid,
)
from ttl.ttnn_proxy import (
    ComputeConfigProxy,
    ComputeConfigResolved,
    ComputeDescriptorBuildContext,
    CoreRangeSetProxy,
    ReaderConfigProxy,
    WriterConfigProxy,
)

# -----------------------------------------------------------------------------
# Program layer (no ttnn required)
# -----------------------------------------------------------------------------


def test_program_decorator_params_indexing_maps_dims_match_iterator_types():
    """ProgramDecoratorParams raises when indexing_map params count != len(iterator_types)."""
    # Two-arg map but three iterator_types -> validation error
    map_2 = lambda i, j: (i, j)
    with pytest.raises(ValidationError, match="Number of dimensions.*must match iterator_types"):
        ProgramDecoratorParams(
            grid=(1, 1),
            indexing_maps=[map_2],
            iterator_types=["parallel", "parallel", "reduction"],
            options=ProgramOptions(),
        )
    # Two-arg map and two iterator_types -> ok
    params = ProgramDecoratorParams(
        grid=(1, 1),
        indexing_maps=[map_2],
        iterator_types=["parallel", "parallel"],
        options=ProgramOptions(),
    )
    assert len(params.indexing_maps) == 1
    assert len(params.iterator_types) == 2


def test_program_spec_build_compile_request():
    """ProgramSpec.build_compile_request(args, kwargs, program_hash) returns KernelCompileRequest."""

    def _dummy_program(_x):
        pass

    options = ProgramOptions()
    spec = ProgramSpec(program=_dummy_program, grid=(2, 2), options=options)
    req = spec.build_compile_request((), {}, program_hash=42)
    assert isinstance(req, KernelCompileRequest)
    assert req.program_hash == 42
    assert req.grid == (2, 2)
    assert req.options.memory_space == "L1"


def test_resolve_grid_callable():
    """_resolve_grid(callable, args, kwargs) returns result of callable."""
    grid = _resolve_grid(lambda a, b: (a, b), (3, 4), {})
    assert grid == (3, 4)


def test_resolve_grid_tuple():
    """_resolve_grid((cols, rows), args, kwargs) returns grid as-is."""
    grid = _resolve_grid((2, 3), (), {})
    assert grid == (2, 3)


def test_run_with_raw_callable_raises_not_implemented():
    """run(RunRequest(spec=ProgramSpec(program=lambda...))) raises NotImplementedError (lambda inference planned)."""
    req = RunRequest(
        spec=ProgramSpec(
            program=lambda lhs, rhs: lhs + rhs,
            grid=(1, 1),
            options=ProgramOptions(),
        ),
        args=(None, None),
        kwargs={},
    )
    with pytest.raises(NotImplementedError, match="raw callable.*not yet implemented"):
        run(req)


def test_run_with_program_no_grid_raises():
    """RunRequest.from_program(program, *args) without grid= raises ValueError."""

    def _fake_program(_x):
        pass

    _fake_program._ttl_program = True  # simulate @ttl.program-decorated
    with pytest.raises(ValueError, match="grid= is required"):
        RunRequest.from_program(_fake_program, None)


def test_run_with_spec_accepts():
    """run(RunRequest(spec=ProgramSpec(...), args=...)) builds and runs (no device)."""

    def _dummy_program(_x):
        pass

    spec = ProgramSpec(
        program=_dummy_program,
        grid=(1, 1),
        options=ProgramOptions(),
    )
    req = RunRequest(spec=spec, args=(None,), kwargs={})
    # Will fail at _compile_kernel (no threads) but run() accepts req and proceeds
    with pytest.raises(ValueError, match="No threads found"):
        run(req)


def test_program_cache_key_empty_args():
    """make_cache_key((), None, None) returns key tuple (empty tensor_key, None, None)."""
    from ttl.program.cache_key import make_cache_key

    key = make_cache_key((), fp32_dest_acc_en=None, dst_full_sync_en=None)
    assert isinstance(key, tuple)
    assert len(key) == 3
    assert key[0] == ()
    assert key[1] is None
    assert key[2] is None


def test_compile_registry_get_and_clear():
    """get_thread_registry() returns registry with clear() and get_and_clear()."""
    from ttl.compile.registry import get_thread_registry

    reg = get_thread_registry()
    reg.clear()
    threads = reg.get_and_clear()
    assert threads == []


def test_compile_stages_save_initial_mlir_disabled():
    """run_stages with SaveInitialMlirStage when settings.initial_mlir is None does not write."""
    from ttl.compile.stages import (
        CompileStageContext,
        get_initial_stages,
        run_stages,
    )

    class SettingsNoInitial:
        initial_mlir = None

    ctx = CompileStageContext(module=None, initial_mlir_path=None)
    run_stages(get_initial_stages(), ctx, SettingsNoInitial())
    # No file written; no exception


def test_compile_stages_context_and_run():
    """CompileStageContext and run_stages accept context; stages run when enabled."""
    from ttl.compile.stages import (
        CompileStageContext,
        get_initial_stages,
        run_stages,
    )

    stages = get_initial_stages()
    assert len(stages) >= 1
    assert stages[0].name == "save_initial_mlir"
    ctx = CompileStageContext(module=None, initial_mlir_path=None)
    run_stages(stages, ctx, type("S", (), {"initial_mlir": None})())
    # No file (path None); no exception


def test_compile_stages_save_initial_mlir_enabled_writes_file(tmp_path):
    """When initial_mlir is set, SaveInitialMlirStage runs and writes module to file."""
    from ttl.compile.stages import (
        CompileStageContext,
        get_initial_stages,
        run_stages,
    )

    out_path = tmp_path / "initial.mlir"

    class MockOp:
        def print(self, *, file, enable_debug_info=False, print_generic_op_form=True):
            file.write("module { }\n")
            file.flush()

    class MockModule:
        operation = MockOp()

    settings = type("S", (), {"initial_mlir": str(out_path)})()
    ctx = CompileStageContext(
        module=MockModule(),
        initial_mlir_path=str(out_path),
    )
    run_stages(get_initial_stages(), ctx, settings)
    assert out_path.exists()
    assert out_path.read_text().strip() == "module { }"


def test_compile_kernel_request_compile_only_path():
    """_compile_kernel(CompileKernelRequest(...)) accepts single request and proceeds to No threads found."""
    from ttl.compile.pipeline import _compile_kernel

    def _dummy_program(_x):
        pass

    class _EmptyRegistry:
        def clear(self) -> None:
            pass

        def get_and_clear(self) -> list:
            return []

    compile_request = KernelCompileRequest(
        grid=(1, 1),
        program_hash=42,
        indexing_maps=[],
        iterator_types=[],
        options=ProgramOptions(),
    )
    req = CompileKernelRequest(
        program=_dummy_program,
        args=(None,),
        kwargs={},
        compile_request=compile_request,
        thread_registry=_EmptyRegistry(),
        engine_config=None,
    )
    with pytest.raises(ValueError, match="No threads found"):
        _compile_kernel(req)


# -----------------------------------------------------------------------------
# Pydantic validation (no ttnn required)
# -----------------------------------------------------------------------------


def test_core_range_set_options_invalid_grid_raises():
    """CoreRangeSetOptions(grid=(0, 1)) raises ValueError."""
    with pytest.raises(ValueError, match="grid.*>= 1"):
        CoreRangeSetOptions(grid=(0, 1))


def test_kernel_descriptor_build_request_validation():
    """KernelDescriptorBuildRequest requires grid_cols/grid_rows >= 1."""
    with pytest.raises(ValidationError):
        KernelDescriptorBuildRequest(
            kernel_specs=[],
            tensors=[],
            tensor_accessor_args=[],
            core_ranges=None,
            grid_cols=0,
            grid_rows=1,
            num_cbs=0,
        )


def test_run_kernel_request_construction():
    """RunKernelRequest accepts required fields; program_hash optional."""
    req = RunKernelRequest(
        kernel_specs=[],
        tensors=[],
        cb_configs=[],
        core_ranges=None,
    )
    assert req.program_hash is None
    req2 = RunKernelRequest(
        kernel_specs=[],
        tensors=[],
        cb_configs=[],
        core_ranges=None,
        program_hash=42,
    )
    assert req2.program_hash == 42


def test_thread_config_build_request_compute_noc_other():
    """ThreadConfigBuildRequest builds config and entries for compute, noc, other."""
    # Only test that the request builds and has expected structure; actual
    # build_config_and_entries() needs ttnn for descriptor creation.
    req_compute = ThreadConfigBuildRequest(
        thread_type="compute",
        name="k",
        compute_opts=ComputeConfigOptions(),
    )
    assert req_compute.thread_type == "compute"
    assert req_compute.name == "k"

    req_noc = ThreadConfigBuildRequest(thread_type="noc", name="dm", noc_kernel_idx=1)
    assert req_noc.thread_type == "noc"
    assert req_noc.noc_kernel_idx == 1


# -----------------------------------------------------------------------------
# descriptor_options and ttnn_proxy (require ttnn)
# -----------------------------------------------------------------------------


@pytest.mark.requires_device
def test_compute_config_resolved_to_ttnn():
    """ComputeConfigResolved.from_proxy_and_context(...).to_ttnn() returns descriptor."""
    pytest.importorskip("ttnn")
    proxy = ComputeConfigProxy(fp32_dest_acc_en=True, dst_full_sync_en=False)
    context = ComputeDescriptorBuildContext(
        kernel_name="k", has_f32=False, verbose=False
    )
    resolved = ComputeConfigResolved.from_proxy_and_context(proxy, context)
    config = resolved.to_ttnn()
    assert config is not None
    assert config.fp32_dest_acc_en is True
    assert config.dst_full_sync_en is False


@pytest.mark.requires_device
def test_reader_writer_config_proxy_to_ttnn():
    """ReaderConfigProxy and WriterConfigProxy .to_ttnn() return descriptors."""
    pytest.importorskip("ttnn")
    reader = ReaderConfigProxy().to_ttnn()
    writer = WriterConfigProxy().to_ttnn()
    assert reader is not None
    assert writer is not None


@pytest.mark.requires_device
def test_core_range_set_proxy_to_ttnn():
    """CoreRangeSetProxy(grid=(2, 2)).to_ttnn() returns CoreRangeSet."""
    pytest.importorskip("ttnn")
    proxy = CoreRangeSetProxy(grid=(2, 2))
    core_range_set = proxy.to_ttnn()
    assert core_range_set is not None
    assert hasattr(core_range_set, "bounding_box")


@pytest.mark.requires_device
def test_core_range_set_options_build_ttnn_core_range_set():
    """CoreRangeSetOptions(grid=(1, 1)).build_ttnn_core_range_set() returns CoreRangeSet."""
    pytest.importorskip("ttnn")
    opts = CoreRangeSetOptions(grid=(1, 1))
    core_range_set = opts.build_ttnn_core_range_set()
    assert core_range_set is not None


@pytest.mark.requires_device
def test_thread_config_build_request_build_config_and_entries():
    """ThreadConfigBuildRequest(compute/noc).build_config_and_entries() returns (config, entries)."""
    pytest.importorskip("ttnn")
    req_compute = ThreadConfigBuildRequest(
        thread_type="compute",
        name="add_k",
        has_f32=False,
    )
    config, entries = req_compute.build_config_and_entries()
    assert config is not None
    assert "TRISC_0" in entries
    assert entries["TRISC_0"] == "add_k"

    req_noc = ThreadConfigBuildRequest(
        thread_type="noc",
        name="dm_r",
        noc_kernel_idx=0,
    )
    config_noc, entries_noc = req_noc.build_config_and_entries()
    assert config_noc is not None
    assert "NCRISC" in entries_noc
    assert entries_noc["NCRISC"] == "dm_r"


# -----------------------------------------------------------------------------
# kernel_runner request-based API (require ttnn)
# -----------------------------------------------------------------------------


@pytest.mark.requires_device
def test_build_kernel_descriptors_empty_specs():
    """build_kernel_descriptors(KernelDescriptorBuildRequest with empty specs) returns []."""
    pytest.importorskip("ttnn")
    core_ranges = CoreRangeSetProxy(grid=(1, 1)).to_ttnn()
    req = KernelDescriptorBuildRequest(
        kernel_specs=[],
        tensors=[],
        tensor_accessor_args=[],
        core_ranges=core_ranges,
        grid_cols=1,
        grid_rows=1,
        num_cbs=0,
    )
    result = build_kernel_descriptors(req)
    assert result == []


@pytest.mark.requires_device
def test_build_cb_descriptors_empty_configs():
    """build_cb_descriptors(CBDescriptorBuildRequest with empty cb_configs) returns []."""
    pytest.importorskip("ttnn")
    core_ranges = CoreRangeSetProxy(grid=(1, 1)).to_ttnn()
    req = CBDescriptorBuildRequest(
        tensors=[],
        cb_configs=[],
        core_ranges=core_ranges,
    )
    result = build_cb_descriptors(req)
    assert result == []
