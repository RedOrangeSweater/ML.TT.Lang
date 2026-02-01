# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared exceptions for middleware pipelines."""


class MiddlewareError(RuntimeError):
    """Base error for middleware pipelines."""


__all__ = [
    "MiddlewareError",
]

