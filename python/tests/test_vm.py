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

"""Execution tests for the tiny IR VM."""

import pytest

from smol.front.ast import BOp
from smol.middle import tir
from smol.vm import VM, run


def test_vm_state_and_basic_blocks() -> None:
    program = tir.Program(
        frozenset({"x", "y"}),
        {
            "entry": tir.Block((tir.Read("x"),), tir.Branch("x", "yes", "no")),
            "yes": tir.Block((tir.Copy("y", "x"), tir.Print("y")), tir.Exit()),
            "no": tir.Block((tir.Print("y"),), tir.Jump("yes")),
        },
    )
    inputs = [0, 9]
    vm = VM(program, inputs)

    assert vm.env == {"x": 0, "y": 0}
    assert vm.inputs == inputs
    assert vm.outputs == []
    assert vm.run_bb("entry") == "no"
    assert vm.inputs == [9]
    assert inputs == [0, 9]
    assert vm.run_bb("no") == "yes"
    assert vm.run_bb("yes") is None
    assert vm.outputs == [0, 0]


def test_run_inst_updates_environment_and_output() -> None:
    vm = VM(tir.Program(frozenset({"x"}), {}), [])
    vm.run_inst(tir.Const("x", 7))
    vm.run_inst(tir.Print("x"))

    assert vm.env == {"x": 7}
    assert vm.outputs == [7]


@pytest.mark.parametrize(("guard", "expected"), [(0, 2), (7, 1)])
def test_branch_selects_block_from_guard(guard: int, expected: int) -> None:
    program = tir.Program(
        frozenset({"guard", "result"}),
        {
            "entry": tir.Block((tir.Read("guard"),), tir.Branch("guard", "yes", "no")),
            "yes": tir.Block((tir.Const("result", 1),), tir.Jump("done")),
            "no": tir.Block((tir.Const("result", 2),), tir.Jump("done")),
            "done": tir.Block((tir.Print("result"),), tir.Exit()),
        },
    )

    assert run(program, [guard]) == [expected]


@pytest.mark.parametrize(
    ("op", "left", "right", "expected"),
    [
        (BOp.ADD, 2, 3, 5),
        (BOp.SUB, 2, 3, -1),
        (BOp.MUL, -3, 4, -12),
        (BOp.DIV, -7, 3, -3),
        (BOp.DIV, 7, -3, -3),
        (BOp.DIV, -7, -3, 2),
        (BOp.DIV, 7, 3, 2),
        (BOp.DIV, 1, 0, -1),
        (BOp.DIV, -(1 << 63), -1, -(1 << 63)),
        (BOp.LT, -1, 0, 1),
        (BOp.LT, 1, 0, 0),
        (BOp.ADD, (1 << 63) - 1, 1, -(1 << 63)),
    ],
)
def test_arithmetic_uses_signed_64_bit_values(
    op: BOp, left: int, right: int, expected: int
) -> None:
    program = tir.Program(
        frozenset({"left", "right", "result"}),
        {
            "entry": tir.Block(
                (
                    tir.Const("left", left),
                    tir.Const("right", right),
                    tir.Arith(op, "result", "left", "right"),
                    tir.Print("result"),
                ),
                tir.Exit(),
            )
        },
    )

    assert run(program, []) == [expected]


def test_read_wraps_to_signed_64_bit_and_preserves_unused_input() -> None:
    program = tir.Program(
        frozenset({"x"}),
        {"entry": tir.Block((tir.Read("x"), tir.Print("x")), tir.Exit())},
    )
    inputs = [1 << 64, 12]

    assert run(program, inputs) == [0]
    assert inputs == [1 << 64, 12]


def test_read_requires_input() -> None:
    program = tir.Program(
        frozenset({"x"}),
        {"entry": tir.Block((tir.Read("x"),), tir.Exit())},
    )

    with pytest.raises(ValueError, match="requires another input"):
        run(program, [])
