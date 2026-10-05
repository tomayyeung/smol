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

"""Lowering from the AST to tiny IR."""

from dataclasses import dataclass

from smol.front import ast
from smol.middle import tir
from smol.middle.tir import *


@dataclass(frozen=True, slots=True)
class Label:
    name: str


@dataclass(frozen=True, slots=True)
class Inner:
    instruction: tir.Instruction


@dataclass(frozen=True, slots=True)
class Term:
    terminator: tir.Terminator


type TvEntry = Label | Inner | Term


def lower(program: ast.Program) -> tir.Program:
    return Lower().lower_program(program)


def construct_cfg(entries: list[TvEntry]) -> dict[str, tir.Block]:
    """Convert a translation vector into a basic-block map."""
    entries.reverse()
    blocks = {}

    while entries:
        label = entries[-1]
        entries.pop()
        assert isinstance(label, Label)
        label = label.name
        assert label not in blocks
        insns: list[tir.Instruction] = []
        while isinstance(entries[-1], Inner):
            insns.append(entries[-1].instruction)
            entries.pop()
        term = entries[-1]
        entries.pop()
        assert isinstance(term, Term)
        blocks[label] = tir.Block(tuple(insns), term.terminator)

    return blocks


class Lower:
    def __init__(self) -> None:
        self.declarations: set[str] = set()
        self.translation: list[TvEntry] = []
        self.fresh_counter = 0
        self.block_counter = 0

    def add_declaration(self, name: str) -> None:
        self.declarations.add(name)

    def emit(self, insn: tir.Instruction) -> None:
        self.translation.append(Inner(insn))

    def emit_term(self, term: tir.Terminator) -> None:
        self.translation.append(Term(term))

    def emit_label(self, lbl: str):
        self.translation.append(Label(lbl))

    def lower_program(self, program: ast.Program) -> tir.Program:
        self.translation.append(Label("entry"))
        for statement in program.statements:
            self.lower_stmt(statement)
        self.translation.append(Term(tir.Exit()))
        return tir.Program(
            declarations=frozenset(self.declarations),
            blocks=construct_cfg(self.translation),
        )

    def lower_stmt(self, statement: ast.Stmt) -> None:
        match statement:
            case ast.Assign(name, value):
                rhs = self.lower_expr(value)
                self.add_declaration(name)
                self.emit(tir.Copy(dst=name, src=rhs))
            case ast.Print(value):
                rhs = self.lower_expr(value)
                self.emit(tir.Print(rhs))
            case ast.Read(name):
                self.add_declaration(name)
                self.emit(tir.Read(name))
            case ast.If(guard, tt, ff):
                cond = self.lower_expr(guard)
                b_tt = self.mk_label()
                b_ff = self.mk_label()
                join = self.mk_label()
                self.emit_term(tir.Branch(cond, b_tt, b_ff))
                self.emit_label(b_tt)
                for stmt in tt:
                    self.lower_stmt(stmt)
                self.emit_term(tir.Jump(join))
                self.emit_label(b_ff)
                for stmt in ff:
                    self.lower_stmt(stmt)
                self.emit_term(tir.Jump(join))
                self.emit_label(join)


    def lower_expr(self, expression: ast.Expr) -> str:
        match expression:
            case ast.Var(x):
                self.add_declaration(x)
                return x
            case ast.Const(n):
                lhs = self.mk_var("_const")
                self.emit(tir.Const(dst=lhs, src=n))
                return lhs
            case ast.BinOp(op, lhs, rhs):
                lhs = self.lower_expr(lhs)
                rhs = self.lower_expr(rhs)
                tmp = self.mk_var("_t")
                self.emit(tir.Arith(op=op, dst=tmp, lhs=lhs, rhs=rhs))
                return tmp
            case ast.Negate(inner):
                inner = self.lower_expr(inner)
                zero = self.lower_expr(ast.Const(0))
                tmp = self.mk_var("_t")
                self.emit(tir.Arith(op=ast.BOp.SUB, dst=tmp, lhs=zero, rhs=inner))
                return tmp

    def mk_var(self, prefix: str) -> str:
        x = f"{prefix}{self.fresh_counter}"
        self.fresh_counter += 1
        self.add_declaration(x)
        return x

    def mk_label(self) -> str:
        l = f"bb{self.block_counter}"
        self.block_counter += 1
        return l
