# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stage: save final MLIR to file when TTLANG_FINAL_MLIR is set."""

from __future__ import annotations

from . import CompileStageContext


class SaveFinalMlirStage:
    """Stage that writes the lowered MLIR module to final_mlir_path when enabled."""

    @property
    def name(self) -> str:
        return "save_final_mlir"

    def enabled(self, settings: object) -> bool:
        path = getattr(settings, "final_mlir", None)
        return path is not None and path != ""

    def run(self, context: CompileStageContext) -> None:
        if context.final_mlir_path is None or context.module is None:
            return
        with open(context.final_mlir_path, "w") as fd:
            context.module.operation.print(
                file=fd,
                enable_debug_info=context.print_debug_locations,
                print_generic_op_form=False,
            )
        print(f"SAVED FINAL TO {context.final_mlir_path}")
