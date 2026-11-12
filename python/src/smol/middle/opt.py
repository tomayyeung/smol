# Copyright (C) 2026 Mehmet Emre
# SPDX-License-Identifier: GPL-3.0-only
#
# This file is part of smol.
#
# smol is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 3 as published by the Free
# Software Foundation.
#
# smol is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# smol. If not, see <https://www.gnu.org/licenses/>.

"""Tiny IR optimizations."""

from smol.middle import const, tir


def optimize(program: tir.Program) -> tir.Program:
    """Replace instructions whose results are constant at the analysis fixpoint."""
    pre = const.analyze(program)
    blocks = {}
    for label, block in program.blocks.items():
        if label not in pre:
            blocks[label] = block
            continue

        _, _, posts = const.analyze_bb(pre[label], block)
        instructions = []
        for instruction, post in zip(block.instructions, posts, strict=True):
            match instruction:
                case tir.Copy(dst, _) | tir.Arith(_, dst, _, _):
                    match post[dst]:
                        case const.Const(value):
                            instruction = tir.Const(dst, value)
            instructions.append(instruction)
        blocks[label] = tir.Block(tuple(instructions), block.terminator)

    return tir.Program(program.declarations, blocks)
