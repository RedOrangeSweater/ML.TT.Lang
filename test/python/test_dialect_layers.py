# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Python dialect layers (descriptor_options, ttnn_proxy, kernel_runner).

Verifies Pydantic request models and layer boundaries: Program/Compile/Runtime
dialects and transformations. See docs/sdlc/00_Main/02_Architecture/08_PythonDialectLayersAndDataFlow.md.
"""

import pytest
from pydantic import ValidationError

from ttl.descriptor_options import (
    ComputeConfigOptions,
    CoreRangeSetOptions,
    NocConfigOptions,
    ThreadConfigBuildRequest,
)
from ttl.kernel_runner import (
    CBDescriptorBuildRequest,
    KernelDescriptorBuildRequest,
    KernelSpec,
    RunKernelRequest,
    build_kernel_descriptors,
    build_cb_descriptors,
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
    context = ComputeDescriptorBuildContext(kernel_name="k", has_f32=False, verbose=False)
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
