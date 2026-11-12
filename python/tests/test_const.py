"""Constant analysis and propagation preserve tiny IR behavior."""

import pytest

from smol.cli import _get_ir
from smol.front.ast import BOp
from smol.middle import const, tir
from smol.middle.opt import optimize
from smol.vm import run


def test_analyze_bb_returns_independent_instruction_post_states() -> None:
    initial: const.State = {"x": const.Const(0), "y": const.Const(0)}
    block = tir.Block(
        (tir.Const("x", 2), tir.Copy("y", "x"), tir.Read("x"), tir.Print("y")),
        tir.Branch("x", "yes", "no"),
    )

    post, successors, posts = const.analyze_bb(initial, block)

    assert successors == ["yes", "no"]
    assert posts == [
        {"x": const.Const(2), "y": const.Const(0)},
        {"x": const.Const(2), "y": const.Const(2)},
        {"x": const.Top(), "y": const.Const(2)},
        {"x": const.Top(), "y": const.Const(2)},
    ]
    assert post == posts[-1]
    assert initial == {"x": const.Const(0), "y": const.Const(0)}


def test_analyze_empty_block() -> None:
    initial: const.State = {"x": const.Const(4)}
    assert const.analyze_bb(initial, tir.Block((), tir.Jump("next"))) == (
        initial,
        ["next"],
        [],
    )


def test_propagation_uses_state_at_each_instruction() -> None:
    program = tir.Program(
        frozenset({"x", "y"}),
        {
            "entry": tir.Block(
                (
                    tir.Const("x", 3),
                    tir.Copy("y", "x"),
                    tir.Arith(BOp.ADD, "x", "x", "y"),
                    tir.Copy("y", "x"),
                    tir.Read("x"),
                    tir.Copy("y", "x"),
                    tir.Print("y"),
                ),
                tir.Exit(),
            )
        },
    )

    optimized = optimize(program)

    assert optimized.blocks["entry"].instructions == (
        tir.Const("x", 3),
        tir.Const("y", 3),
        tir.Const("x", 6),
        tir.Const("y", 6),
        tir.Read("x"),
        tir.Copy("y", "x"),
        tir.Print("y"),
    )
    assert isinstance(program.blocks["entry"].instructions[1], tir.Copy)
    assert run(optimized, [9]) == run(program, [9]) == [9]


@pytest.mark.parametrize("right_value", [5, 6])
def test_propagation_joins_paths_at_fixpoint(right_value: int) -> None:
    program = tir.Program(
        frozenset({"guard", "x", "y"}),
        {
            "entry": tir.Block((tir.Read("guard"),), tir.Branch("guard", "a", "b")),
            "a": tir.Block((tir.Const("x", 5),), tir.Jump("done")),
            "b": tir.Block((tir.Const("x", right_value),), tir.Jump("done")),
            "done": tir.Block((tir.Copy("y", "x"), tir.Print("y")), tir.Exit()),
        },
    )

    optimized = optimize(program)
    expected = tir.Const("y", 5) if right_value == 5 else tir.Copy("y", "x")
    assert optimized.blocks["done"].instructions[0] == expected
    for guard in [0, 1]:
        assert run(optimized, [guard]) == run(program, [guard])


def test_unreachable_blocks_are_preserved() -> None:
    program = tir.Program(
        frozenset({"x", "y"}),
        {
            "entry": tir.Block((tir.Const("x", 1),), tir.Branch("x", "a", "b")),
            "a": tir.Block((), tir.Jump("done")),
            "b": tir.Block((tir.Read("x"),), tir.Jump("done")),
            "done": tir.Block((tir.Copy("y", "x"),), tir.Exit()),
        },
    )
    optimized = optimize(program)
    assert optimized.blocks["b"] == program.blocks["b"]
    assert optimized.blocks["entry"].terminator == program.blocks["entry"].terminator
    assert optimized.blocks["done"].instructions == (tir.Const("y", 1),)


def test_loop_uses_fixpoint_rather_than_first_iteration() -> None:
    program = tir.Program(
        frozenset({"x", "one", "guard", "y"}),
        {
            "entry": tir.Block((tir.Const("one", 1),), tir.Jump("loop")),
            "loop": tir.Block(
                (tir.Arith(BOp.ADD, "x", "x", "one"), tir.Read("guard")),
                tir.Branch("guard", "loop", "done"),
            ),
            "done": tir.Block((tir.Copy("y", "x"), tir.Print("y")), tir.Exit()),
        },
    )
    optimized = optimize(program)
    assert optimized.blocks["loop"] == program.blocks["loop"]
    assert optimized.blocks["done"] == program.blocks["done"]
    assert run(optimized, [1, 0]) == run(program, [1, 0]) == [2]


@pytest.mark.parametrize(
    ("op", "left", "right", "expected"),
    [
        (BOp.ADD, (1 << 63) - 1, 1, -(1 << 63)),
        (BOp.SUB, -(1 << 63), 1, (1 << 63) - 1),
        (BOp.MUL, 1 << 62, 4, 0),
        (BOp.DIV, -7, 3, -3),
        (BOp.DIV, 7, -3, -3),
        (BOp.DIV, -7, -3, 2),
        (BOp.DIV, 7, 3, 2),
        (BOp.DIV, 1, 0, -1),
        (BOp.DIV, -(1 << 63), -1, -(1 << 63)),
        (BOp.LT, 1 << 63, 0, 1),
    ],
)
def test_folded_arithmetic_matches_vm(
    op: BOp, left: int, right: int, expected: int
) -> None:
    program = tir.Program(
        frozenset({"x", "y", "z"}),
        {
            "entry": tir.Block(
                (
                    tir.Const("x", left),
                    tir.Const("y", right),
                    tir.Arith(op, "z", "x", "y"),
                    tir.Print("z"),
                ),
                tir.Exit(),
            )
        },
    )
    optimized = optimize(program)
    assert optimized.blocks["entry"].instructions[2] == tir.Const("z", expected)
    assert run(optimized, []) == run(program, []) == [expected]


@pytest.mark.parametrize("guard", [0, 1])
def test_zero_initialization_is_respected_at_joins(guard: int) -> None:
    source = "$read guard $if guard {:= x 5} {} := y x $print y"
    assert run(_get_ir(source, True), [guard]) == run(_get_ir(source, False), [guard])


def test_cli_optimization_folds_lowered_ir() -> None:
    program = _get_ir(":= x 2 := y + x 3 $print y", True)
    assert all(
        not isinstance(instruction, (tir.Copy, tir.Arith))
        for block in program.blocks.values()
        for instruction in block.instructions
    )
    assert run(program, []) == [5]
