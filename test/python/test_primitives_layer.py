# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for primitives layer: composable DM/compute/sync units and registry.

Smoke tests: primitives package imports; ComposablePrimitive.compose(); PrimitiveRegistry.
See docs/sdlc/00_Main/02_Architecture/20_IdealDataFlowAndModuleStructure.md (§7).
"""


from ttl.primitives import (
    BuildContext,
    ComposablePrimitive,
    DMPrimitive,
    ElementwiseOp,
    PrimitiveRegistry,
    Reader,
    Semaphore,
    Writer,
)


def test_primitives_import():
    """Primitives package exports base, dm, compute, sync, registry types."""
    assert BuildContext is not None
    assert ComposablePrimitive is not None
    assert DMPrimitive is not None
    assert Reader is not None
    assert Writer is not None
    assert ElementwiseOp is not None
    assert Semaphore is not None
    assert PrimitiveRegistry is not None


def test_build_context_model():
    """BuildContext is a Pydantic model with grid and memory_space."""
    ctx = BuildContext(grid=(2, 2), memory_space="L1")
    assert ctx.grid == (2, 2)
    assert ctx.memory_space == "L1"


def test_composable_primitive_compose():
    """ComposablePrimitive.compose() returns new instance with appended children."""
    base = ComposablePrimitive()
    r = Reader(name="r")
    w = Writer(name="w")
    composed = base.compose(r, w)
    assert len(composed.children) == 2
    assert composed.children[0].name == "r"
    assert composed.children[1].name == "w"


def test_primitive_registry_register_and_get():
    """PrimitiveRegistry.register(name) and get(name) round-trip."""
    @PrimitiveRegistry.register("test_dummy_prim")
    class DummyPrim(DMPrimitive):
        pass

    got = PrimitiveRegistry.get("test_dummy_prim")
    assert got is DummyPrim
    assert "test_dummy_prim" in PrimitiveRegistry.list_names()
